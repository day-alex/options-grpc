#ifndef BLACK_SCHOLES_H
#define BLACK_SCHOLES_H

#include <utility>

double normal_cdf(double x);

std::pair<double, double> calculate_black_scholes(
    double S,
    double K,
    double T,
    double R,
    double V
);

#endif
