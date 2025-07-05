#!/bin/bash

set -e  # Exit immediately on error

echo "🧹 Cleaning previous build..."
rm -rf build
mkdir -p build
cd build

echo "🛠  Running CMake..."
cmake ..

echo "🔨 Building project..."
make -j$(nproc)

echo "✅ Build complete."

cd ..
