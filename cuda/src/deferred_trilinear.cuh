#pragma once

namespace deferred_trilinear {

constexpr int rgb = 3;
constexpr int output = 11;
constexpr int harmonic = 35;
constexpr int linear = 67;

__device__ inline float3 midpoint(
    float3 ro, float3 rd, float a, float b, float3 center, float side)
{
    const float distance = a + 0.5f * (b - a);
    return (ro + distance * rd - (center - 0.5f * side)) * (1.f / side);
}

template<int width>
__device__ inline void interpolate(const float* values, const float* weights, float* z)
{
    for (int c = 0; c < width; ++c) {
        z[c] = 0.f;
        for (int j = 0; j < 8; ++j)
            z[c] += weights[j] * values[j * width + c];
    }
}

template<int width>
__device__ inline void accumulate(
    const float* values, const float* weights, float weight, float* out)
{
    float z[width];
    interpolate<width>(values, weights, z);
    for (int c = 0; c < width; ++c) {
        if constexpr (width == 4) {
            float sine, cosine;
            sincosf(z[c], &sine, &cosine);
            out[c] += weight * sine;
            out[c + 4] += weight * cosine;
        } else {
            out[c] += weight * z[c];
        }
    }
}

template<int width>
__device__ inline void backward(const float* values, const float* weights,
    float weight, const float* pixel_grad, int pixels, float* out)
{
    float z[width];
    if constexpr (width == 4)
        interpolate<width>(values, weights, z);
    for (int c = 0; c < width; ++c) {
        float grad = pixel_grad[c * pixels];
        if constexpr (width == 4) {
            float sine, cosine;
            sincosf(z[c], &sine, &cosine);
            grad = grad * cosine - pixel_grad[(c + 4) * pixels] * sine;
        }
        for (int j = 0; j < 8; ++j)
            atomicAdd(out + j * width + c, weight * weights[j] * grad);
    }
}

}  // namespace deferred_trilinear
