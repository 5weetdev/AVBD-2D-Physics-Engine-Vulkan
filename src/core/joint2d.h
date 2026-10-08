#pragma once

// The 2D joint model, shared by authoring (scene2d.h) and the device (JointAxis2 in
// types2d.slang mirrors JointAxis2 here, 12 words). A joint is a frame on each body (anchor plus,
// on body A, an axis angle; the reference angle is the relative angle at rest) and three axes:
//   axis 0: translation along frame-A x      (a distance joint: along the anchor-to-anchor line)
//   axis 1: translation along frame-A y
//   axis 2: relative angle  angB - angA - reference
// Every axis has up to three rows: sub 0 the position row (lock = hard, spring = soft), sub 1 the
// motor row, sub 2 the limit row. Presets (revolute, prismatic, ...) are only axis settings.

#include <cstdint>

namespace avbd2d
{

enum JointPos2 : uint32_t
{
    JOINT_AXIS_FREE = 0,
    JOINT_AXIS_LOCK = 1,   // hard row with a multiplier, target `rest`
    JOINT_AXIS_SPRING = 2, // soft row around `rest`: hertz > 0 derives the stiffness, else `stiff` ramps
};

enum JointMotor2 : uint32_t
{
    JOINT_MOTOR_OFF = 0,
    JOINT_MOTOR_VELOCITY = 1, // drive the axis at `mtarget` (m/s or rad/s), force bounded by `maxF`
    JOINT_MOTOR_TARGET = 2,   // hold the axis at `mtarget`, force bounded by `maxF` (Box2D motor joint)
};

enum JointFlags2 : uint32_t
{
    JOINT_FLAG_GRAB = 1,     // the mouse grab slot (axis 0 only)
    JOINT_FLAG_DISTANCE = 2, // axis 0 measures the anchor distance, not a frame-A coordinate
};

struct JointAxis2
{
    uint32_t pos = 0, motor = 0, limit = 0, flags = 0;
    float lo = 0.0f, hi = 0.0f; // limit range
    float rest = 0.0f;          // lock/spring target
    float mtarget = 0.0f;       // motor speed or target
    float maxF = 0.0f;          // motor force (N) or torque (N m)
    float hertz = 0.0f;         // spring frequency; 0 = use `stiff`
    float stiff = 0.0f;         // ramped spring stiffness cap (solver units: angular is per arm^2)
    float damp = 0.0f;          // damping ratio, accepted and ignored (TODO in the solver)
};
static_assert(sizeof(JointAxis2) == 48, "JointAxis2 drifted from types2d.slang");

} // namespace avbd2d
