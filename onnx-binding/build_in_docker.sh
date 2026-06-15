#!/bin/bash
# Build script to compile inside the ROCm container

set -e

echo "Installing Rust..."
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"

echo "Installing build dependencies..."
apt-get update && apt-get install -y build-essential pkg-config libssl-dev

echo "Building with ROCm support..."
cd /workspace
cargo build --release --features rocm --example test_gpu

echo "Running test..."
./target/release/examples/test_gpu
