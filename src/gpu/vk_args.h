#pragma once

// Host mirrors of the "*ArgsGpu" structs the shared primitives and constraint-graph kernels read, one
// header per shader family under args/. Each struct is a byte-for-byte transcription of its Slang
// counterpart, with VkDeviceAddress standing in for pointers and explicit _padN fields for the padding
// Slang inserts under scalar layout.

#include "args/primitives_args.h"
#include "args/constraints_args.h"
#include "args/gate_args.h"
