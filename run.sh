#!/bin/bash

# Stop on error
set -e

echo "🔨 Building the project..."
cmake -S . -B build
cmake --build build -j$(nproc)

echo "🚀 Running the server..."
./build/options_grpc_server
