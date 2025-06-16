#ifndef PIXEL_PRIM_NEON_H__
#define PIXEL_PRIM_NEON_H__

#include "common.h"
#include "slicetype.h"      // LOWRES_COST_MASK
#include "primitives.h"
#include "x265.h"



namespace X265_NS
{



void setupPixelPrimitives_neon(EncoderPrimitives &p);

#if defined(HAVE_NEON_DOTPROD)
void setupPixelPrimitives_neon_dotprod(EncoderPrimitives &p);
#endif

#if defined(HAVE_SVE) && HAVE_SVE_BRIDGE
void setupPixelPrimitives_sve(EncoderPrimitives &p);
#endif
}


#endif

