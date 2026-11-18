#pragma once

#include <cooperative_groups/reduce.h>
#include <torch/extension.h>

namespace rbf4_iq_sine {

constexpr int inputs = 22;
constexpr int hidden = 64;
constexpr int channels = 4;
constexpr int b1 = hidden * inputs;
constexpr int w2 = b1 + hidden;
constexpr int b2 = w2 + 3 * hidden;
constexpr int parameters = b2 + 3;

struct Field {
    const int64_t* keys = nullptr;
    const float* features = nullptr;
    const float* scales = nullptr;
    const float* phase = nullptr;
    const float* mlp = nullptr;
    const float* frequencies = nullptr;
};

struct Gradients {
    float* features = nullptr;
    float* scales = nullptr;
    float* phase = nullptr;
    float* mlp = nullptr;
};

inline Field make_field(const std::vector<torch::Tensor>& tensors, const torch::Tensor& rgbs)
{
    if (tensors.empty())
        return {};
    TORCH_CHECK(tensors.size() == 6, "RBF4 requires six field tensors");
    for (size_t i = 0; i < tensors.size(); ++i) {
        const auto& t = tensors[i];
        TORCH_CHECK(t.device() == rgbs.device() && t.is_cuda() && t.is_contiguous(),
                    "RBF4 tensors must be contiguous on the raster device");
        TORCH_CHECK(t.scalar_type() == (i == 0 ? torch::kInt64 : torch::kFloat32),
                    "RBF4 tensor dtype mismatch");
    }

    const auto n = rgbs.size(0);
    TORCH_CHECK(tensors[0].sizes() == at::IntArrayRef({n, 8}) &&
                tensors[1].dim() == 2 && tensors[1].size(1) == channels &&
                tensors[2].sizes() == at::IntArrayRef({tensors[1].size(0), 3}) &&
                tensors[3].sizes() == at::IntArrayRef({channels}) &&
                tensors[4].sizes() == at::IntArrayRef({parameters}) &&
                tensors[5].sizes() == at::IntArrayRef({channels}), "RBF4 tensor shape mismatch");
    return {tensors[0].data_ptr<int64_t>(), tensors[1].data_ptr<float>(),
            tensors[2].data_ptr<float>(), tensors[3].data_ptr<float>(),
            tensors[4].data_ptr<float>(), tensors[5].data_ptr<float>()};
}

struct Sample {
    float phi[8];
    float inv_sum;
    float x[inputs];
    float h[hidden];
    float rgb[3];  // Before the non-negative clamp.
};

__device__ __forceinline__ float offset_squared(float coordinate, int corner, int axis)
{
    const float d = coordinate - ((corner >> (2 - axis)) & 1);
    return d * d;
}

__device__ __forceinline__ float inverse_scale_squared(float eta)
{
    return expf(-2.f * fminf(2.f, fmaxf(-2.f, eta)));
}

__device__ __forceinline__ void direction_encoding(float3 rd, float* encoding)
{
    const float3 d = rd * rsqrtf(dot(rd, rd));
    for (int k = 0; k < 3; ++k) {
        const float frequency = 1 << k;
        sincosf(frequency * d.x, encoding + 6*k, encoding + 6*k + 3);
        sincosf(frequency * d.y, encoding + 6*k + 1, encoding + 6*k + 4);
        sincosf(frequency * d.z, encoding + 6*k + 2, encoding + 6*k + 5);
    }
}

static __device__ __noinline__ void evaluate(
    Field field, int voxel, float3 t, const float* direction, float3 sh0, Sample& sample)
{
    const auto keys = field.keys + 8 * voxel;
    const float coordinates[] = {t.x, t.y, t.z};
    float sum = 0.f;
    for (int i = 0; i < 8; ++i) {
        float denominator = 1.f;
        for (int a = 0; a < 3; ++a)
            denominator += offset_squared(coordinates[a], i, a)
                * inverse_scale_squared(field.scales[3*keys[i] + a]);
        sample.phi[i] = 1.f / denominator;
        sum += sample.phi[i];
    }
    sample.inv_sum = 1.f / sum;

    for (int c = 0; c < channels; ++c) {
        float x = 0.f;
        for (int i = 0; i < 8; ++i) {
            const float angle = sample.phi[i] * sample.inv_sum * field.frequencies[c]
                + field.phase[c];
            x += field.features[channels*keys[i] + c] * sinf(angle);
        }
        sample.x[c] = x;
    }
    for (int c = 0; c < 18; ++c)
        sample.x[channels + c] = direction[c];

    for (int j = 0; j < hidden; ++j) {
        float z = field.mlp[b1 + j];
        for (int k = 0; k < inputs; ++k)
            z += field.mlp[j*inputs + k] * sample.x[k];
        sample.h[j] = fmaxf(0.f, z);
    }

    const float base[] = {sh0.x, sh0.y, sh0.z};
    for (int c = 0; c < 3; ++c) {
        float residual = field.mlp[b2 + c];
        for (int j = 0; j < hidden; ++j)
            residual += field.mlp[w2 + c*hidden + j] * sample.h[j];
        sample.rgb[c] = base[c] + residual;
    }
}

__device__ __forceinline__ float3 color(const Sample& sample)
{
    return {fmaxf(0.f, sample.rgb[0]), fmaxf(0.f, sample.rgb[1]), fmaxf(0.f, sample.rgb[2])};
}

static __device__ __noinline__ float3 evaluate_color(
    Field field, int voxel, float3 t, const float* direction, float3 sh0)
{
    Sample sample;
    evaluate(field, voxel, t, direction, sh0, sample);
    return color(sample);
}

__device__ __forceinline__ void warp_atomic(
    const cooperative_groups::coalesced_group& warp, float* target, float value)
{
    // The same global coefficient is addressed by every participating lane.
    const float sum = cooperative_groups::reduce(warp, value, cooperative_groups::plus<float>());
    if (warp.thread_rank() == 0)
        atomicAdd(target, sum);
}

static __device__ __noinline__ void backward(
    Field field, Gradients grad, int voxel, float3 t, const Sample& sample, float* rgb_grad)
{
    const auto warp = cooperative_groups::coalesced_threads();
    for (int c = 0; c < 3; ++c) {
        rgb_grad[c] *= sample.rgb[c] > 0.f;
        warp_atomic(warp, grad.mlp + b2 + c, rgb_grad[c]);
        for (int j = 0; j < hidden; ++j)
            warp_atomic(warp, grad.mlp + w2 + c*hidden + j, rgb_grad[c] * sample.h[j]);
    }
    float spatial_grad[channels] = {};
    for (int j = 0; j < hidden; ++j) {
        float g = 0.f;
        for (int c = 0; c < 3; ++c)
            g += rgb_grad[c] * field.mlp[w2 + c*hidden + j];
        g *= sample.h[j] > 0.f;
        warp_atomic(warp, grad.mlp + b1 + j, g);
        for (int k = 0; k < inputs; ++k)
            warp_atomic(warp, grad.mlp + j*inputs + k, g * sample.x[k]);
        for (int c = 0; c < channels; ++c)
            spatial_grad[c] += g * field.mlp[j*inputs + c];
    }

    const auto keys = field.keys + 8 * voxel;
    float alpha_grad[8] = {};
    for (int c = 0; c < channels; ++c) {
        float phase_grad = 0.f;
        for (int i = 0; i < 8; ++i) {
            const auto key = channels*keys[i] + c;
            const float angle = sample.phi[i] * sample.inv_sum * field.frequencies[c]
                + field.phase[c];
            float sine, cosine;
            sincosf(angle, &sine, &cosine);
            atomicAdd(grad.features + key, spatial_grad[c] * sine);
            const float g = spatial_grad[c] * field.features[key] * cosine;
            phase_grad += g;
            alpha_grad[i] += g * field.frequencies[c];
        }
        warp_atomic(warp, grad.phase + c, phase_grad);
    }
    float mean = 0.f;
    for (int i = 0; i < 8; ++i)
        mean += sample.phi[i] * sample.inv_sum * alpha_grad[i];

    const float coordinates[] = {t.x, t.y, t.z};
    for (int i = 0; i < 8; ++i) {
        const float g = (alpha_grad[i] - mean) * sample.inv_sum;
        for (int a = 0; a < 3; ++a) {
            const auto key = 3*keys[i] + a;
            const float eta = field.scales[key];
            if (eta >= -2.f && eta <= 2.f) {
                const float derivative = 2.f * sample.phi[i] * sample.phi[i]
                    * offset_squared(coordinates[a], i, a) * inverse_scale_squared(eta);
                atomicAdd(grad.scales + key, g * derivative);
            }
        }
    }
}

}  // namespace rbf4_iq_sine
