/*****************************************************************************
 * Copyright (C) 2024 MulticoreWare, Inc
 *
 * Authors: Hari Limaye <hari.limaye@arm.com>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02111, USA.
 *
 * This program is also available under a commercial proprietary license.
 * For more information, contact us at license @ x265.com.
 *****************************************************************************/

#ifndef X265_COMMON_AARCH64_FILTER_NEON_DOTPROD_H
#define X265_COMMON_AARCH64_FILTER_NEON_DOTPROD_H

#if defined(HAVE_NEON_DOTPROD)

#include "primitives.h"

namespace X265_NS {
void setupFilterPrimitives_neon_dotprod(EncoderPrimitives &p);
}

#endif // defined(HAVE_NEON_DOTPROD)

#endif // X265_COMMON_AARCH64_FILTER_NEON_DOTPROD_H
