#include "options.grpc.pb.h"
#include "options/black_scholes.h"
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
