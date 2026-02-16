#include "options.grpc.pb.h"
#include "options/black_scholes.h"
#include "options/monte_carlo.h"
#include <grpcpp/grpcpp.h>
#include <cmath>
#include <iostream>

class OptionsImpl : public Options::Service {
public:
    grpc::Status BlackScholes(grpc::ServerContext* context,
                                const OptionInputs* request,
                                OptionPrices* response) override {
        std::cout << "-- Received gRPC request: "
              << "S=" << request->s() << ", "
              << "K=" << request->k() << ", "
              << "T=" << request->t() << ", "
              << "R=" << request->r() << ", "
              << "V=" << request->v() << std::endl;

        std::pair<double, double> prices = calculate_black_scholes(
            request->s(),
            request->k(),
            request->t(),
            request->r(),
            request->v()
        );

        response->set_c(prices.first);
        response->set_p(prices.second);

        return grpc::Status::OK;
    }

    grpc::Status MonteCarlo(grpc::ServerContext* context,
                            const MonteCarloInputs* request,
                            ::MonteCarloResult* response) override {
        int num_paths = request->num_paths() > 0 ? request->num_paths() : 100000;
        int num_steps = request->num_steps() > 0 ? request->num_steps() : 252;
        int num_vis_paths = request->num_vis_paths() > 0 ? request->num_vis_paths() : 50;

        std::cout << "-- MonteCarlo request: "
                  << "S=" << request->s() << ", "
                  << "K=" << request->k() << ", "
                  << "T=" << request->t() << ", "
                  << "R=" << request->r() << ", "
                  << "V=" << request->v() << ", "
                  << "paths=" << num_paths << ", "
                  << "steps=" << num_steps << ", "
                  << "vis_paths=" << num_vis_paths << std::endl;

        MCPriceResult mc = monte_carlo_price(
            request->s(), request->k(), request->t(),
            request->r(), request->v(),
            num_paths, num_steps, num_vis_paths
        );

        response->set_call_price(mc.call_price);
        response->set_put_price(mc.put_price);
        response->set_call_std_error(mc.call_std_error);
        response->set_put_std_error(mc.put_std_error);
        response->set_num_paths(mc.num_paths);
        response->set_num_vis_paths(mc.num_vis_paths);
        response->set_num_steps(mc.num_steps);
        for (double v : mc.path_data) {
            response->add_path_data(v);
        }

        return grpc::Status::OK;
    }
};


int main() {
    OptionsImpl service;
    grpc::ServerBuilder builder;
    builder.AddListeningPort("0.0.0.0:50051", grpc::InsecureServerCredentials());
    builder.RegisterService(&service);
    
    std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
    std::cout << "Server listening on port 50051" << std::endl;
    server->Wait();
    return 0;
}
