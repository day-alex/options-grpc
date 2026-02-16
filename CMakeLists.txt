cmake_minimum_required(VERSION 3.14)
project(OptionsGRPCServer)

set(CMAKE_CXX_STANDARD 17)

# If gRPC was installed to ~/.local
set(CMAKE_PREFIX_PATH "$ENV{HOME}/.local")

# Include directories for headers
include_directories(
  ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_SOURCE_DIR}/src/options
  ${CMAKE_SOURCE_DIR}/generated
)

# Source files
set(GENERATED_SRCS
  ${CMAKE_SOURCE_DIR}/generated/options.pb.cc
  ${CMAKE_SOURCE_DIR}/generated/options.grpc.pb.cc
)

set(SRC_FILES
  ${CMAKE_SOURCE_DIR}/src/server.cpp
  ${CMAKE_SOURCE_DIR}/src/options/black_scholes.cpp
  ${CMAKE_SOURCE_DIR}/src/options/monte_carlo.cpp
)

# Find gRPC and Protobuf using installed CMake config files
find_package(Protobuf REQUIRED)
find_package(gRPC REQUIRED)

# Create the executable
add_executable(options_grpc_server ${SRC_FILES} ${GENERATED_SRCS})

# Link against shared libraries
target_link_libraries(options_grpc_server
  PRIVATE
    gRPC::grpc++
    protobuf::libprotobuf
)
