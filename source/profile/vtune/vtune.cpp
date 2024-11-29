/*****************************************************************************
 * Copyright (C) 2013-2020 MulticoreWare, Inc
 *
 * Authors: Steve Borho <steve@borho.org>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02111, USA.
 *
 * This program is also available under a commercial proprietary license.
 * For more information, contact us at license @ x265.com.
 *****************************************************************************/

#include "vtune.h"
#include <stdio.h>

namespace {

#define CPU_EVENT(x) #x,
const char *stringNames[] =
{
#include "../cpuEvents.h"
};
#undef CPU_EVENT

}

namespace X265_NS {

__itt_domain* domain;
__itt_string_handle* taskHandle[NUM_VTUNE_TASKS];

void vtuneInit()
{
    domain = __itt_domain_create("x265");
    size_t length = sizeof(stringNames) / sizeof(const char *);
    for (size_t i = 0; i < length; i++)
        taskHandle[i] = __itt_string_handle_create(stringNames[i]);
}

void vtuneSetThreadName(const char *name, int id)
{
    char threadname[128];
    snprintf(threadname, sizeof(threadname), "%s %d", name, id);
    __itt_thread_set_name(threadname);
}

}
