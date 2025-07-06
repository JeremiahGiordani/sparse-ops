#!/bin/bash

set -e  # Exit immediately on error

export CPPFLAGS="-I/opt/homebrew/opt/libomp/include"
export LDFLAGS="-L/opt/homebrew/opt/libomp/lib"

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
