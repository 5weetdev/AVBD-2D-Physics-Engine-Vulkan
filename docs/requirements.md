# Requirements

**What you get:** everything to install, then the commands that build and run the demo yourself.
Nothing is prebuilt: you compile the library and the demo from source.

A GPU with a current driver that supports Vulkan 1.3. The build is tested on one machine (RTX 4070 SUPER, Windows 10 x64).

## 1. Install

| Tool | Why | Install |
| :--- | :--- | :--- |
| Git | Clone the repo and its submodules | `winget install Git.Git` |
| Visual Studio 2022 Build Tools | MSVC compiler | `winget install Microsoft.VisualStudio.2022.BuildTools --override "--add Microsoft.VisualStudio.Workload.VCTools --includeRecommended --passive"` |
| CMake 3.20+ | Build system | `winget install Kitware.CMake` |
| Ninja | Generator | `winget install Ninja-build.Ninja` |
| Python 3 | Embeds the compiled shaders | `winget install Python.Python.3.12` |
| Vulkan SDK | Vulkan loader and `slangc` | Download from <https://vulkan.lunarg.com/sdk/home#windows>. The installer sets `VULKAN_SDK`; if CMake still cannot find it, pass `-DVulkan_ROOT=<SDK path>` |

Open a new terminal afterwards so `PATH` picks everything up. Check:

```bat
git --version && cmake --version && ninja --version && python --version && echo %VULKAN_SDK%
```

## 2. Get the source

```bat
git clone --recurse-submodules https://github.com/5weetdev/AVBD-2D-Physics-Engine-Vulkan.git
cd AVBD-2D-Physics-Engine-Vulkan
```

Already cloned? Run `git submodule update --init`. SDL2 and Dear ImGui (demo only) live in `external/`.

## 3. Build

Open **x64 Native Tools Command Prompt for VS 2022** (it has the compiler on `PATH`), then:

```bat
cmake -B build -S . -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

This produces, in `build\`:

| File | What |
| :--- | :--- |
| `avbd2d_demo.exe` + `SDL2.dll` | The demo |
| `avbd2d_solver.dll` + `.lib` | The C library |

## 4. Run the demo

```bat
build\avbd2d_demo.exe
```

## Targets

| Target | Output in `build\` | Purpose |
| :--- | :--- | :--- |
| `avbd2d_solver` | `avbd2d_solver.dll`, `.lib` | The C library. What games link against. |
| `avbd2d_demo` | `avbd2d_demo.exe`, `SDL2.dll` | SDL2 + ImGui demo. |

Build one with `cmake --build build --target <name>`.

## Install

```bat
cmake --install build --prefix <install dir>
```

Installs `avbd2d_solver.dll` to `bin`, `avbd2d_solver.lib` to `lib` and `avbd2d.h` to `include\avbd2d`.

## Building from a plain shell

CMake needs the MSVC compiler on `PATH`. Outside the Native Tools prompt, chain `vcvars64.bat`
(path depends on your edition: `Community`, `Professional`, `Enterprise` or `BuildTools`):

```bat
cmd /c "call "C:\Program Files\Microsoft Visual Studio2\Community\VC\Auxiliary\Buildcvars64.bat" && cmake -B build -S . -G Ninja && cmake --build build"
```

## Problems

See [troubleshooting](troubleshooting.md).
