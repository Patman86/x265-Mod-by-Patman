/*****************************************************************************
 * Copyright (C) 2025 MulticoreWare, Inc
 *
 * Authors: Gerda Zsejke More <gerdazsejke.more@arm.com>
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

#ifndef X265_FILTER_PRIM_SVE_H
#define X265_FILTER_PRIM_SVE_H

#if defined(HAVE_SVE)

#include "primitives.h"

namespace X265_NS {
void setupFilterPrimitives_sve(EncoderPrimitives &p);
}

#endif // defined(HAVE_SVE)

#endif // X265_FILTER_PRIM_SVE_H
