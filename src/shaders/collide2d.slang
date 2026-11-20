// 2D shape collision, split by family: collide2d_box.slang (box / circle SAT and clipping, the
// legacy path) and manifold2d.slang (the Box2D v3 manifold port). Kernels import this umbrella.

module collide2d;

__exported import collide2d_box;
__exported import manifold2d;
