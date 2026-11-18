#pragma once
// Local inverse-quadratic modulation. Density weights N are read-only.
__forceinline__ __device__ void rbf_weights(float3 t, const float N[8],
    const float* eta, const int64_t* keys, float w[8], float metric[24]) {
    float sum=0.f;
    for(int k=0;k<8;++k) {
        float delta[3]={t.x-float(k>>2),t.y-float((k>>1)&1),t.z-float(k&1)};
        float D=0.f,D0=0.f, parts[3];
        for(int a=0;a<3;++a) {
            float e=fminf(2.f,fmaxf(-2.f,eta[keys[k]*3+a]));
            float square=delta[a]*delta[a];
            parts[a]=square*expf(-2.f*e); D+=parts[a]; D0+=square;
        }
        w[k]=N[k]*(1.f+D0)/(1.f+D); sum+=w[k];
        for(int a=0;a<3;++a) metric[k*3+a]=2.f*parts[a]/(1.f+D);
    }
    for(int k=0;k<8;++k) w[k]/=sum;
}
