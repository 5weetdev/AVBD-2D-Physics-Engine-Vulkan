#!/usr/bin/env bash
# Build avbd2d on Linux or macOS (Linux tested; macOS untested).
#
# Usage:  ./build.sh [release|debug] [clean] [run]
#   release   optimised build in build/release (default)
#   debug     debug build in build/debug
#   clean     delete the build folder first
#   run       launch the demo after building
set -euo pipefail
cd "$(dirname "$0")"

config=Release
dir=build/release
clean=0; run=0
for a in "$@"; do
    case "$a" in
        release) config=Release; dir=build/release ;;
        debug)   config=Debug;   dir=build/debug ;;
        clean)   clean=1 ;;
        run)     run=1 ;;
        *) echo "[build] unknown argument: $a"
           echo "usage: ./build.sh [release|debug] [clean] [run]"
           exit 2 ;;
    esac
done

for t in cmake ninja python3; do
    command -v "$t" >/dev/null || { echo "[build] ERROR: $t is not on PATH."; exit 1; }
done
if [ -z "${VULKAN_SDK:-}" ] && ! command -v slangc >/dev/null; then
    echo "[build] ERROR: slangc not found. Install the Vulkan SDK and source its setup-env.sh."
    exit 1
fi
if [ ! -f external/SDL/CMakeLists.txt ] || [ ! -f external/imgui/imgui.cpp ]; then
    echo "[build] Fetching submodules..."
    git submodule update --init --recursive
fi

if [ "$clean" = 1 ] && [ -d "$dir" ]; then
    echo "[build] Removing $dir"
    rm -rf "$dir"
fi

if command -v nproc >/dev/null; then jobs=$(nproc); else jobs=$(sysctl -n hw.ncpu); fi

echo "[build] Configuring $config in $dir"
cmake -B "$dir" -S . -G Ninja -DCMAKE_BUILD_TYPE="$config"
echo "[build] Building"
cmake --build "$dir" -j "$jobs"

if [ "$run" = 1 ]; then
    echo "[build] Launching the demo"
    "$dir/avbd2d_demo"
fi
echo "[build] Done: $PWD/$dir"
