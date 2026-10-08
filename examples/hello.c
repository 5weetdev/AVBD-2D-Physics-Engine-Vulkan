/*
 * hello.c -- a box falls onto a static ground box, through the avbd2d C ABI.
 *
 * Uses only the public header include/avbd2d/avbd2d.h. See docs/getting-started.md for how to
 * build and run it.
 */
#include <stdio.h>
#include <stdlib.h>

#include "avbd2d.h"

/* Every call returns Avbd2dResult; stop on anything but AVBD2D_OK. */
#define CHECK(call)                                                             \
    do {                                                                        \
        Avbd2dResult check_r_ = (call);                                         \
        if (check_r_ != AVBD2D_OK) {                                            \
            fprintf(stderr, "%s failed: %s\n", #call, avbd2d_result_string(check_r_)); \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

int main(void)
{
    /* 1. World. Start from the default def, then override the fields you care about. */
    Avbd2dWorldDef worldDef = avbd2d_default_world_def();
    worldDef.gravity = (Avbd2dVec2){0.0f, -10.0f};
    worldDef.timeStep = 1.0f / 60.0f;

    Avbd2dWorld *world = NULL;
    CHECK(avbd2d_create_world(&worldDef, &world));

    /* 2. Ground: a static body (the default body type) with one box shape. The box is
     *    40 m x 2 m centred on the origin, so its top face is at y = 1. */
    Avbd2dBodyDef groundDef = avbd2d_default_body_def();
    groundDef.position = (Avbd2dVec2){0.0f, 0.0f};

    Avbd2dBodyId ground;
    CHECK(avbd2d_create_body(world, &groundDef, &ground));

    Avbd2dShapeDef shapeDef = avbd2d_default_shape_def();
    Avbd2dPolygon groundBox = avbd2d_make_box(20.0f, 1.0f);

    Avbd2dShapeId groundShape;
    CHECK(avbd2d_create_polygon_shape(ground, &shapeDef, &groundBox, &groundShape));

    /* 3. A dynamic 1 m box, starting 10 m above the origin. */
    Avbd2dBodyDef boxDef = avbd2d_default_body_def();
    boxDef.type = AVBD2D_DYNAMIC_BODY;
    boxDef.position = (Avbd2dVec2){0.0f, 10.0f};

    Avbd2dBodyId box;
    CHECK(avbd2d_create_body(world, &boxDef, &box));

    Avbd2dPolygon boxShape = avbd2d_make_box(0.5f, 0.5f);

    Avbd2dShapeId boxShapeId;
    CHECK(avbd2d_create_polygon_shape(box, &shapeDef, &boxShape, &boxShapeId));

    /* 4. Step 120 frames (two seconds) and read the box's origin every 20 frames. */
    for (int frame = 1; frame <= 120; ++frame) {
        CHECK(avbd2d_step(world));

        if (frame % 20 == 0) {
            Avbd2dVec2 p;
            CHECK(avbd2d_body_get_position(box, &p));
            printf("frame %3d  y = %.3f\n", frame, p.y);
        }
    }

    /* 5. Destroying the world frees its bodies, shapes and joints. */
    CHECK(avbd2d_destroy(world));
    return 0;
}
