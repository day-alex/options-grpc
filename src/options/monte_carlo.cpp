#include "monte_carlo.h"
#include <cmath>
#include <random>
#include <vector>
#include <algorithm>

MCPriceResult monte_carlo_price(
    double S, double K, double T, double r, double sigma,
    int num_paths, int num_steps, int num_vis_paths)
{
    // Cap visualization paths
    num_vis_paths = std::min(num_vis_paths, 200);
    num_vis_paths = std::min(num_vis_paths, num_paths);

    double dt = T / num_steps;
    double drift = (r - 0.5 * sigma * sigma) * dt;
    double vol_sqrt_dt = sigma * std::sqrt(dt);
    double discount = std::exp(-r * T);

    std::mt19937_64 rng(42);
    std::normal_distribution<double> norm(0.0, 1.0);

    std::vector<double> call_payoffs(num_paths);
    std::vector<double> put_payoffs(num_paths);

    // Pre-allocate path data: num_vis_paths rows × (num_steps+1) columns
    std::vector<double> path_data(num_vis_paths * (num_steps + 1));

    for (int i = 0; i < num_paths; ++i) {
        double St = S;
        bool record = (i < num_vis_paths);

        if (record) {
            path_data[i * (num_steps + 1)] = St; // step 0 = initial spot
        }

        for (int j = 0; j < num_steps; ++j) {
            double Z = norm(rng);
            St *= std::exp(drift + vol_sqrt_dt * Z);
            if (record) {
                path_data[i * (num_steps + 1) + j + 1] = St;
            }
        }
        call_payoffs[i] = std::max(St - K, 0.0);
        put_payoffs[i] = std::max(K - St, 0.0);
    }

    // Compute means
    double call_sum = 0.0, put_sum = 0.0;
    for (int i = 0; i < num_paths; ++i) {
        call_sum += call_payoffs[i];
        put_sum += put_payoffs[i];
    }
    double call_mean = call_sum / num_paths;
    double put_mean = put_sum / num_paths;

    // Compute standard errors
    double call_var_sum = 0.0, put_var_sum = 0.0;
    for (int i = 0; i < num_paths; ++i) {
        double cd = call_payoffs[i] - call_mean;
        double pd = put_payoffs[i] - put_mean;
        call_var_sum += cd * cd;
        put_var_sum += pd * pd;
    }
    double call_std = std::sqrt(call_var_sum / (num_paths - 1));
    double put_std = std::sqrt(put_var_sum / (num_paths - 1));
    double sqrt_n = std::sqrt(static_cast<double>(num_paths));

    MCPriceResult result;
    result.call_price = discount * call_mean;
    result.put_price = discount * put_mean;
    result.call_std_error = discount * call_std / sqrt_n;
    result.put_std_error = discount * put_std / sqrt_n;
    result.num_paths = num_paths;
    result.path_data = std::move(path_data);
    result.num_vis_paths = num_vis_paths;
    result.num_steps = num_steps;
    return result;
}
