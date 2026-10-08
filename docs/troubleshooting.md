# Troubleshooting

**What you get:** the symptom, the usual cause and the fix for the errors avbd2d and its build can report.

Result codes are the `Avbd2dResult` values in `avbd2d.h`. Each call returns one; `avbd2d_result_string()` gives text for a code.

## Runtime errors

| Symptom | Cause | Fix |
| :--- | :--- | :--- |
| `avbd2d_create` returns `AVBD2D_ERR_DEVICE` | No usable Vulkan 1.3 device, a required feature is missing, or the subgroup width is neither 32 nor 64 and cannot be pinned to 32 | Update the GPU driver. Use an NVIDIA RTX GPU (the only tested target). If you pass your own device, enable the full feature list from `limitations.md`. Fall back to Box2D |
| `AVBD2D_ERR_DEVICE_LOST` from any call | The GPU device failed or was reset by the driver. The world is dead; only `avbd2d_destroy` works on it | Destroy the world and create a new one. Check for driver crashes in the system event log. Repro with `AVBD_VK_VALIDATION=1` |
| `AVBD2D_ERR_OUT_OF_MEMORY` | Host or device memory ran out, or the memory budget refused a required allocation | Close other GPU-heavy apps. Reduce the body count. Raise or clear the budget with `avbd2d_set_memory_budget` (0 = unlimited) |
| `AVBD2D_ERR_CAPACITY` from `avbd2d_step` | A hard ceiling or the memory budget clamped the step. The step still ran | Reduce bodies or pairs, or raise the budget. Check `avbd2d_get_stats` (`ok` = 0) |
| `AVBD2D_ERR_INVALID_ARG` on a body, shape or joint call | The id was destroyed or never issued. Slots are reused with a new generation | Call `avbd2d_body_is_valid` before use. Drop ids after you destroy their object |
| `AVBD2D_ERR_NULL_HANDLE` | The world pointer is null, never created, or already destroyed | Keep the world handle alive until `avbd2d_destroy` returns |
| A setter seems to do nothing yet | Setters apply at the start of the next step. Getters read the last finished step | Read back after the next `avbd2d_step` |
| A resting body does not react to an event | Per-body `enableSleep` is ignored; sleep is velocity-gated | Call `avbd2d_body_set_awake` or `avbd2d_wake_all`, or set `sleepFrames = 0` |
| A bounce does not happen | Restitution has no effect in v1 | See the restitution row in `limitations.md` |
| Two machines give slightly different results | No determinism guarantee | Expected. Sync state, not inputs |

## Build and setup errors

| Symptom | Cause | Fix |
| :--- | :--- | :--- |
| CMake: `The Vulkan SDK was not found` | `find_package(Vulkan)` failed | Install the Vulkan SDK, set `VULKAN_SDK` to its root, or pass `-DVulkan_ROOT=<SDK path>` |
| CMake: `slangc was not found next to the Vulkan SDK` | The SDK's `Bin` folder has no `slangc.exe` | Install the SDK version the build was tested with (1.4.357.0), or set `VULKAN_SDK` to a path that contains `Bin\slangc.exe` |
| Demo fails to start, `SDL2.dll` missing | The DLL is not next to `avbd2d_demo.exe` | Run the demo from the build folder, where the build copies `SDL2.dll` |
| Start-up takes several seconds | About 50 pipelines are compiled each launch (cache not persisted) | Expected in v1. Create the world once during loading |

## Validation layer

Turn the validation layer on with `AVBD_VK_VALIDATION=1` (set it in the environment before the program starts).

| Symptom | Cause | Fix |
| :--- | :--- | :--- |
| stderr: `validation requested but VK_LAYER_KHRONOS_validation not found` | The layer is not installed | Install the Vulkan SDK (it ships the layer), then run again |
| Validation messages on a clean run | A real API misuse | Fix the reported call. Report it with the message text and GPU driver version |
