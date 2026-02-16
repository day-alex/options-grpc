#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <numeric>

#define IDX(i, j, N) ((i) * (N) + (j))

// Check CUDA errors
#define CUDA_CHECK(call) do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                  << " - " << cudaGetErrorString(error) << std::endl; \
        exit(1); \
    } \
} while(0)

__host__ __device__ float norm_cdf(float x) {
    return 0.5f * erfcf(-x * M_SQRT1_2);
}

__host__ __device__ float  bs_call_price(float S, float K, float T, float r, float sigma) {
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

void cpu_implied_vol(
    float* strikes, float* expirations, float* market_prices,
    float S, float r,
    float* out_vols,
    int num_strikes, int num_exps
) {
    for (int i = 0; i < num_exps; ++i) {
        for (int j = 0; j < num_strikes; ++j) {
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
}

void print_gpu_info() {
    cudaDeviceProp prop;
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    
    std::cout << "\n========== GPU INFORMATION ==========\n";
    std::cout << "Device: " << prop.name << "\n";
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << "\n";
    std::cout << "Clock Rate: " << prop.clockRate / 1000 << " MHz\n";
    std::cout << "Memory Clock: " << prop.memoryClockRate / 1000 << " MHz\n";
    std::cout << "Memory Bus Width: " << prop.memoryBusWidth << " bits\n";
    std::cout << "Peak Memory Bandwidth: " << 
        2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6 << " GB/s\n";
}

int main() {
    print_gpu_info();
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    const int num_strikes = 1000;
    const int num_exps = 1000;
    float spot = 100.0f;
    float rate = 0.05f;

    // [Keep your existing allocation and initialization code]
    float* h_cpu_output_vols = new float[num_strikes * num_exps];
    float* h_strikes = new float[num_strikes];
    float* h_exps = new float[num_exps];
    float* h_market_prices = new float[num_strikes * num_exps];
    float* h_output_vols = new float[num_strikes * num_exps];

    // [Keep your existing initialization loops]
    for (int j = 0; j < num_strikes; ++j) {
        h_strikes[j] = 75.0f + j * 5.0f;
    }

    for (int i = 0; i < num_exps; ++i) {
        h_exps[i] = (7.0f + i * ((365.0f - 7.0f) / (num_exps - 1))) / 365.0f;
    }

    // [Keep your market price generation]
    for (int i = 0; i < num_exps; ++i) {
        float T = h_exps[i];
        for (int j = 0; j < num_strikes; ++j) {
            float K = h_strikes[j];
            float moneyness = fabsf((K - spot) / spot);
            float vol = 0.2f + 0.1f * moneyness;

            float d1 = (logf(spot / K) + (rate + 0.5f * vol * vol) * T) / (vol * sqrtf(T));
            float d2 = d1 - vol * sqrtf(T);
            float price = spot * norm_cdf(d1) - K * expf(-rate * T) * norm_cdf(d2);

            h_market_prices[i * num_strikes + j] = price;
        }
    }

    // GPU allocation
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

    // ============ WARM-UP GPU ============
    std::cout << "\nWarming up GPU...\n";
    const int warmup_runs = 100;
    for (int i = 0; i < warmup_runs; ++i) {
        implied_vol_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            d_strikes, d_exps, d_market_prices, spot, rate,
            d_output_vols, num_strikes, num_exps
        );
    }
    cudaDeviceSynchronize();

    // ============ DETAILED GPU TIMING ============
    const int num_runs = 1000;
    std::vector<float> gpu_times;
    
    // Time individual runs for statistics
    for (int run = 0; run < num_runs; ++run) {
        cudaEventRecord(start);
        
        implied_vol_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            d_strikes, d_exps, d_market_prices, spot, rate,
            d_output_vols, num_strikes, num_exps
        );
        
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        gpu_times.push_back(ms);
    }

    // Calculate GPU statistics
    std::sort(gpu_times.begin(), gpu_times.end());
    float gpu_sum = std::accumulate(gpu_times.begin(), gpu_times.end(), 0.0f);
    float gpu_mean = gpu_sum / gpu_times.size();
    float gpu_median = gpu_times[gpu_times.size() / 2];
    float gpu_min = gpu_times[0];
    float gpu_max = gpu_times[gpu_times.size() - 1];
    
    // Calculate standard deviation
    float sq_sum = 0;
    for (float t : gpu_times) {
        sq_sum += (t - gpu_mean) * (t - gpu_mean);
    }
    float gpu_stdev = std::sqrt(sq_sum / gpu_times.size());

    // Copy results back
    cudaMemcpy(h_output_vols, d_output_vols, 
                         sizeof(float) * num_strikes * num_exps, 
                         cudaMemcpyDeviceToHost);

    // ============ CPU TIMING ============
    std::cout << "Running CPU benchmark...\n";
    std::vector<float> cpu_times;
    
    for (int run = 0; run < std::min(10, num_runs); ++run) { // Fewer CPU runs as it's slower
        auto cpu_start = std::chrono::high_resolution_clock::now();
        
        cpu_implied_vol(h_strikes, h_exps, h_market_prices, 
                       spot, rate, h_cpu_output_vols, 
                       num_strikes, num_exps);
        
        auto cpu_end = std::chrono::high_resolution_clock::now();
        auto cpu_duration = std::chrono::duration_cast<std::chrono::microseconds>
                           (cpu_end - cpu_start);
        cpu_times.push_back(cpu_duration.count() / 1000.0f);
    }
    
    float cpu_sum = std::accumulate(cpu_times.begin(), cpu_times.end(), 0.0f);
    float cpu_mean = cpu_sum / cpu_times.size();

    // ============ PRINT DETAILED RESULTS ============
    std::cout << "\n========== PERFORMANCE COMPARISON ==========\n";
    std::cout << "Problem size: " << num_strikes << " strikes × " 
              << num_exps << " expirations = " 
              << (num_strikes * num_exps) << " total calculations\n\n";
    
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "GPU Timing Statistics (1000 runs):\n";
    std::cout << "  Mean:   " << gpu_mean << " ms\n";
    std::cout << "  Median: " << gpu_median << " ms\n";
    std::cout << "  StdDev: " << gpu_stdev << " ms (" 
              << (gpu_stdev/gpu_mean*100) << "%)\n";
    std::cout << "  Min:    " << gpu_min << " ms\n";
    std::cout << "  Max:    " << gpu_max << " ms\n\n";
    
    std::cout << "CPU Timing:\n";
    std::cout << "  Mean:   " << cpu_mean << " ms\n\n";
    
    std::cout << "Performance Metrics:\n";
    std::cout << "  Speedup: " << std::setprecision(1) << cpu_mean / gpu_mean << "x\n";
    
    // Calculate throughput
    int total_calcs = num_strikes * num_exps;
    float gpu_calcs_per_sec = (total_calcs / gpu_mean) * 1000;
    float cpu_calcs_per_sec = (total_calcs / cpu_mean) * 1000;
    
    std::cout << "  GPU Throughput: " << std::scientific << std::setprecision(3)
              << gpu_calcs_per_sec << " IVs/second\n";
    std::cout << "  CPU Throughput: " << std::scientific 
              << cpu_calcs_per_sec << " IVs/second\n";

    // Verify results match
    float max_diff = 0.0f;
    int diff_count = 0;
    for (int i = 0; i < num_strikes * num_exps; ++i) {
        float diff = fabsf(h_output_vols[i] - h_cpu_output_vols[i]);
        max_diff = fmaxf(max_diff, diff);
        if (diff > 1e-5f) diff_count++;
    }
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "\nAccuracy Check:\n";
    std::cout << "  Max difference: " << max_diff << "\n";
    std::cout << "  Values differing > 1e-5: " << diff_count << "/" 
              << (num_strikes * num_exps) << "\n";

    // Cleanup
    cudaFree(d_strikes);
    cudaFree(d_exps);
    cudaFree(d_market_prices);
    cudaFree(d_output_vols);
    
    delete[] h_strikes;
    delete[] h_exps;
    delete[] h_market_prices;
    delete[] h_output_vols;
    delete[] h_cpu_output_vols;

    return 0;
}
