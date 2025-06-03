#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>


#define IDX(i, j, N) ((i) * (N) + (j))

__host__ __device__ float norm_cdf(float x) {
    return 0.5f * erfcf(-x * M_SQRT1_2);
}

__device__ float bs_call_price(float S, float K, float T, float r, float sigma) {
    if (T <= 0.0f) return fmaxf(S - K, 0.0f); // Handle near-zero expiry

    float d1 = (logf(S / K) + (r + 0.5f * sigma * sigma) * T) / (sigma * sqrtf(T));
    float d2 = d1 - sigma * sqrtf(T);

    return S * norm_cdf(d1) - K * expf(-r * T) * norm_cdf(d2);
}

__global__ void implied_vol_kernel(
    float* strikes, float* expirations, float* market_prices,
    float S, float r,
    float* out_vols,
    int num_strikes, int num_exps
) {
    int i = blockIdx.y * blockDim.y + threadIdx.y; // expiration index
    int j = blockIdx.x * blockDim.x + threadIdx.x; // strike index

    if (i < num_exps && j < num_strikes) {
        float K = strikes[j];
        float T = expirations[i];
        float P_market = market_prices[IDX(i, j, num_strikes)];

        float low = 0.01f, high = 3.0f, mid;
        float price;
        for (int k = 0; k < 100; ++k) {
            mid = 0.5f * (low + high);
            price = bs_call_price(S, K, T, r, mid);
            if (fabsf(price - P_market) < 1e-4f) break;
            if (price > P_market) high = mid;
            else low = mid;
        }

        out_vols[IDX(i, j, num_strikes)] = mid;
    }
}


int main() {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const int num_strikes = 11;
    const int num_exps = 20;
    float spot = 100.0f;
    float rate = 0.05f;

    float* h_strikes = new float[num_strikes];
    float* h_exps = new float[num_exps];
    float* h_market_prices = new float[num_strikes * num_exps];
    float* h_output_vols = new float[num_strikes * num_exps];

    for (int j = 0; j < num_strikes; ++j) {
        h_strikes[j] = 75.0f + j * 5.0f;
    }

    // Generate expirations: 7d to 365d, evenly spaced
    for (int i = 0; i < num_exps; ++i) {
        h_exps[i] = (7.0f + i * ((365.0f - 7.0f) / (num_exps - 1))) / 365.0f;
    }

    // Generate synthetic market prices using BS with smile
    for (int i = 0; i < num_exps; ++i) {
        float T = h_exps[i];
        for (int j = 0; j < num_strikes; ++j) {
            float K = h_strikes[j];
            float moneyness = fabsf((K - spot) / spot);
            float vol = 0.2f + 0.1f * moneyness;  // smile shape

            float d1 = (logf(spot / K) + (rate + 0.5f * vol * vol) * T) / (vol * sqrtf(T));
            float d2 = d1 - vol * sqrtf(T);
            float price = spot * norm_cdf(d1) - K * expf(-rate * T) * norm_cdf(d2);

            h_market_prices[i * num_strikes + j] = price;
        }
    }

    float *d_strikes, *d_exps, *d_market_prices, *d_output_vols;

    cudaMalloc(&d_strikes, sizeof(float) * num_strikes);
    cudaMalloc(&d_exps, sizeof(float) * num_exps);
    cudaMalloc(&d_market_prices, sizeof(float) * num_strikes * num_exps);
    cudaMalloc(&d_output_vols, sizeof(float) * num_strikes * num_exps);

    cudaMemcpy(d_strikes, h_strikes, sizeof(float) * num_strikes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_exps, h_exps, sizeof(float) * num_exps, cudaMemcpyHostToDevice);
    cudaMemcpy(d_market_prices, h_market_prices, sizeof(float) * num_strikes * num_exps, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((num_strikes + 15) / 16, (num_exps + 15) / 16);

    cudaEventRecord(start);
    
    implied_vol_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_strikes, d_exps, d_market_prices, spot, rate,
        d_output_vols, num_strikes, num_exps
    );
    
    cudaEventRecord(stop);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output_vols, d_output_vols, sizeof(float) * num_strikes * num_exps, cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < num_exps; ++i) {
        for (int j = 0; j < num_strikes; ++j) {
            printf("IV[T=%.2f, K=%.0f] = %.4f\n", h_exps[i], h_strikes[j], h_output_vols[IDX(i, j, num_strikes)]);
        }
    }

    float ms = 0;
    
    cudaEventElapsedTime(&ms, start, stop);
    printf("GPU compute time: %.3f ms\n", ms);
    
    cudaFree(d_strikes);
    cudaFree(d_exps);
    cudaFree(d_market_prices);
    cudaFree(d_output_vols);
    
    delete[] h_strikes;
    delete[] h_exps;
    delete[] h_market_prices;
    delete[] h_output_vols;

}
