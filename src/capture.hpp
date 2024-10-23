#pragma once

extern "C"
{
#include <libavcodec/avcodec.h>
}

void video_encode_example(const char *filename, AVCodecID codec_id);