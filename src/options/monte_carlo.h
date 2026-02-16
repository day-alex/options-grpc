#ifndef MONTE_CARLO_H
#define MONTE_CARLO_H

#include <vector>

struct MCPriceResult {
    double call_price;
    double put_price;
    double call_std_error;
    double put_std_error;
    int num_paths;
    std::vector<double> path_data;
    int num_vis_paths;
    int num_steps;
};

MCPriceResult monte_carlo_price(
    double S, double K, double T, double r, double sigma,
    int num_paths, int num_steps, int num_vis_paths = 50
);

#endif
