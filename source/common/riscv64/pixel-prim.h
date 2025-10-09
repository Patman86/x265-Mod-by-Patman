#ifndef PIXEL_PRIM_RVV_H__
#define PIXEL_PRIM_RVV_H__

#include "common.h"
#include "slicetype.h" // LOWRES_COST_MASK
#include "primitives.h"
#include "x265.h"

namespace X265_NS {

void setupPixelPrimitives_rvv(EncoderPrimitives &p);

} // namespace X265_NS

#endif