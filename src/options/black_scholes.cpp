#include "black_scholes.h"
#include <cmath>
#include <utility>

double normal_cdf(double x) {
    return 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)));
}

std::pair<double, double> calculate_black_scholes(double S, double K, double T, double R, double V) {
    double d1 = (std::log(S / K) + (R + 0.5 * V * V) * T) / (V * std::sqrt(T));
    double d2 = d1 - V * std::sqrt(T);

    double call = S * normal_cdf(d1) - K * std::exp(-R * T) * normal_cdf(d2);
    double put = K * std::exp(-R * T) * normal_cdf(-d2) - S * normal_cdf(-d1);

    return { call, put };
}
