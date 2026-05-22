/*****************************************************************************
 * Copyright (C) 2026 MulticoreWare, Inc
 *
 * Authors: Alex Davicenko <alex.davicenko@arm.com>
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

#ifndef X265_COMMON_AARCH64_NEON_SVE2_BRIDGE_H
#define X265_COMMON_AARCH64_NEON_SVE2_BRIDGE_H

#include <arm_neon.h>

#if defined(HAVE_SVE2) && HAVE_SVE_BRIDGE
#include <arm_sve.h>
#include <arm_neon_sve_bridge.h>

/* We can access instructions that are exclusive to the SVE2 instruction
 * sets from a predominantly Neon context by making use of the Neon-SVE2 bridge
 * intrinsics to reinterpret Neon vectors as SVE2 vectors - with the high part of
 * the SVE2 vector (if it's longer than 128 bits) being "don't care".
 *
 * While sub-optimal on machines that have SVE2 vector length > 128-bit - as the
 * remainder of the vector is unused - this approach is still beneficial when
 * compared to a Neon-only implementation. */

template<uint64_t Rotation>
static inline int16x8_t x265_caddq_s16(const int16x8_t x, const int16x8_t y)
{
    return svget_neonq_s16(svcadd_s16(svset_neonq_s16(svundef_s16(), x),
                                      svset_neonq_s16(svundef_s16(), y), Rotation));
}

#endif // defined(HAVE_SVE2) && HAVE_SVE_BRIDGE

#endif // X265_COMMON_AARCH64_NEON_SVE2_BRIDGE_H
