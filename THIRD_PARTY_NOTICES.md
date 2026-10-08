# Third-party notices

avbd2d is released under the MIT License (see [LICENSE](LICENSE)). It includes or is derived from the third-party works below.

## Box2D (ported code)

The contact-manifold code in the 2D solver's collider is a port of Box2D.

- Copyright (c) Erin Catto
- License: MIT

```
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

Source: https://github.com/erincatto/box2d

## SDL2 (demo only)

Vendored as a git submodule at `external/SDL`. Used by `avbd2d_demo` only; not part of `avbd2d_solver`.

- Copyright (C) 1997-2025 Sam Lantinga
- License: zlib. Full text: [external/SDL/LICENSE.txt](external/SDL/LICENSE.txt)

Source: https://github.com/libsdl-org/SDL

## Dear ImGui (demo only)

Vendored as a git submodule at `external/imgui`. Used by `avbd2d_demo` only.

- Copyright (c) 2014-2025 Omar Cornut
- License: MIT. Full text: [external/imgui/LICENSE.txt](external/imgui/LICENSE.txt)

Source: https://github.com/ocornut/imgui

## AVBD demo code (Chris Giles)

Parts of the Vulkan layer and the constraint-graph builder are derived from Chris Giles's AVBD
demo code. The files in `src/gpu/` that carry his header keep it, as its terms require:

- Copyright (c) 2025-2026 Chris Giles

```
Permission to use, copy, modify, distribute and sell this software
and its documentation for any purpose is hereby granted without fee,
provided that the above copyright notice appear in all copies.
Chris Giles makes no representations about the suitability
of this software for any purpose.
It is provided "as is" without express or implied warranty.
```

Source: https://github.com/savant117/avbd-demo3d

## AVBD method

The solver implements Augmented Vertex Block Descent, the method of Chris Giles, Elie Diaz and Cem Yuksel:

> Chris Giles, Elie Diaz, and Cem Yuksel. "Augmented Vertex Block Descent." ACM Transactions on Graphics (SIGGRAPH 2025).
