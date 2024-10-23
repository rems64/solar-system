#include "capture.hpp"

extern "C"
{
#include <libavutil/opt.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}

void video_encode_example(const char *filename, AVCodecID codec_id)
{
    AVCodec *codec;
    AVCodecContext *c = NULL;
    int i, ret, x, y, got_output;
    FILE *f;
    AVFrame *frame;
    AVPacket pkt;
    uint8_t endcode[] = {0, 0, 1, 0xb7};

    printf("Encode video file %s\n", filename);

    /* find the mpeg1 video encoder */
    codec = avcodec_find_encoder(codec_id);
    if (!codec)
    {
        fprintf(stderr, "Codec not found\n");
        exit(1);
    }

    c = avcodec_alloc_context3(codec);
    if (!c)
    {
        fprintf(stderr, "Could not allocate video codec context\n");
        exit(1);
    }

    /* put sample parameters */
    c->bit_rate = 400000;
    /* resolution must be a multiple of two */
    c->width = 352;
    c->height = 288;
    /* frames per second */
    c->time_base = AVRational{1, 25};
    /* emit one intra frame every ten frames
     * check frame pict_type before passing frame
     * to encoder, if frame->pict_type is AV_PICTURE_TYPE_I
     * then gop_size is ignored and the output of encoder
     * will always be I frame irrespective to gop_size
     */
    c->gop_size = 10;
    c->max_b_frames = 1;
    c->pix_fmt = AV_PIX_FMT_YUV420P /* AV_PIX_FMT_RGB48 */;

    if (codec_id == AV_CODEC_ID_H264)
        av_opt_set(c->priv_data, "preset", "slow", 0);

    /* open it */
    if (avcodec_open2(c, codec, NULL) < 0)
    {
        fprintf(stderr, "Could not open codec\n");
        exit(1);
    }

    f = fopen(filename, "wb");
    if (!f)
    {
        fprintf(stderr, "Could not open %s\n", filename);
        exit(1);
    }

    frame = av_frame_alloc();
    if (!frame)
    {
        fprintf(stderr, "Could not allocate video frame\n");
        exit(1);
    }
    frame->format = c->pix_fmt;
    frame->width = c->width;
    frame->height = c->height;

    /* the image can be allocated by any means and av_image_alloc() is
     * just the most convenient way if av_malloc() is to be used */
    ret = av_image_alloc(frame->data, frame->linesize, c->width, c->height,
                         c->pix_fmt, 32);
    if (ret < 0)
    {
        fprintf(stderr, "Could not allocate raw picture buffer\n");
        exit(1);
    }

    uint8_t *image_data = new uint8_t[4 * c->width * c->height];
    /* encode 10 second of video */
    int frame_count = 250;
    for (i = 0; i < frame_count; i++)
    {
        av_init_packet(&pkt);
        pkt.data = NULL; // packet data will be allocated by the encoder
        pkt.size = 0;
        pkt.pts = i;

        fflush(stdout);
        /* prepare a dummy image */
        /* Y */
        for (y = 0; y < c->height; y++)
        {
            for (x = 0; x < c->width; x++)
            {
                uint8_t val = (x % 40 >= (40 * ((float)i / (float)frame_count))) ? 255 : 0;
                // val = 255;
                uint8_t r = val;
                uint8_t g = val;
                uint8_t b = val;
                image_data[4 * (y * c->width + x) + 0] = r;
                image_data[4 * (y * c->width + x) + 1] = g;
                image_data[4 * (y * c->width + x) + 2] = b;
                image_data[4 * (y * c->width + x) + 3] = 255;
                // frame->data[0][y * frame->linesize[0] + x] = (uint8_t)yp;
                // frame->data[1][y * frame->linesize[1] + x] = (uint8_t)(u + 0.436) / 0.872;
                // frame->data[2][y * frame->linesize[2] + x] = (uint8_t)(v + 0.615) / 1.230;
            }
        }
        SwsContext *ctx = sws_getContext(c->width, c->height,
                                         AV_PIX_FMT_RGBA,
                                         c->width, c->height,
                                         AV_PIX_FMT_YUV420P,
                                         0, 0, 0, 0);
        uint8_t *inData[1] = {image_data};

        int inLinesize[1] = {4 * c->width};
        sws_scale(ctx, inData, inLinesize, 0, c->height,
                  frame->data, frame->linesize);

        frame->pts = i;

        /* encode the image */
        ret = avcodec_encode_video2(c, &pkt, frame, &got_output);
        if (ret < 0)
        {
            fprintf(stderr, "Error encoding frame\n");
            exit(1);
        }

        if (got_output)
        {
            printf("Write frame %3d (size=%5d)\n", i, pkt.size);
            fwrite(pkt.data, 1, pkt.size, f);
            av_free_packet(&pkt);
        }
    }

    /* get the delayed frames */
    for (got_output = 1; got_output; i++)
    {
        fflush(stdout);

        ret = avcodec_encode_video2(c, &pkt, NULL, &got_output);
        if (ret < 0)
        {
            fprintf(stderr, "Error encoding frame\n");
            exit(1);
        }

        if (got_output)
        {
            printf("Write frame %3d (size=%5d)\n", i, pkt.size);
            fwrite(pkt.data, 1, pkt.size, f);
            av_free_packet(&pkt);
        }
    }

    /* add sequence end code to have a real mpeg file */
    fwrite(endcode, 1, sizeof(endcode), f);
    fclose(f);

    avcodec_close(c);
    av_free(c);
    av_freep(&frame->data[0]);
    av_frame_free(&frame);
    printf("\n");
}