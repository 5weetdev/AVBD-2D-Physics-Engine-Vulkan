# Contributing to avbd2d

Thanks for helping. This page covers the minimum you need before opening a pull request.

## Build

Follow [docs/requirements.md](docs/requirements.md). Use a fresh `build\` directory when you change CMake files.

## AI-assisted contributions

AI-assisted pull requests are accepted. Say so in the description, and make sure you have read and
tested the change yourself.

## Rules for changes

- Keep the solver's numerics unchanged unless the change is a deliberate fix.
- Do not change the C ABI (`include/avbd2d/avbd2d.h`) without a version bump.
- Keep `src/` independent of `demo/`.

## Reporting GPU or driver bugs

Open an issue with:

- GPU model, driver version and operating system
- Vulkan SDK version
- The exact scene and steps that failed
- The output of `build\avbd2d_demo.exe --smoke` with `AVBD_VK_VALIDATION=1` set

## Pull request checklist

- [ ] Builds from a clean `build\` directory with no new warnings
- [ ] `avbd2d_demo.exe --smoke` exits 0
- [ ] Validation layer reports no messages (`AVBD_VK_VALIDATION=1`)
- [ ] Docs updated for any changed behaviour or command
