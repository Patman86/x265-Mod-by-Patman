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

#ifndef X265_YUV_H
#define X265_YUV_H

#include "input.h"
#include "threading.h"
#include <fstream>

#define QUEUE_SIZE 5

namespace X265_NS {
// private x265 namespace

class YUVInput : public InputFile, public Thread
{
protected:

    int width;

    int height;

    int colorSpace; //< source Color Space Parameter

    uint32_t depth;

    uint32_t framesize;

    bool alphaAvailable;

    bool threadActive;

    ThreadSafeInteger readCount;

    ThreadSafeInteger writeCount;
    char* buf[QUEUE_SIZE];
    FILE *ifs;
    int guessFrameCount();
    void threadMain();

    bool populateFrameQueue();

public:

    YUVInput(InputFileInfo& info, bool alpha, int format);

    virtual ~YUVInput();
    void release();
    bool isEof() const                            { return ifs && feof(ifs); }
    bool isFail()                                 { return !(ifs && !ferror(ifs) && threadActive); }
    void startReader();
    void stopReader() {}

    bool readPicture(x265_picture&);

    const char *getName() const                   { return "yuv"; }

    int getWidth() const                          { return width; }

    int getHeight() const                         { return height; }
};
}

#endif // ifndef X265_YUV_H
