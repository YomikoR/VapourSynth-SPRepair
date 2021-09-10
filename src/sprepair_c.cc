#include "vapoursynth/VapourSynth.h"
#include "vapoursynth/VSHelper.h"

#include <algorithm>
#include <vector>

template <class T>
static inline T sort9_repair_c(const T &src0,
    const T &src1, const T &src2, const T &src3,
    const T &src4, const T &src5, const T &src6,
    const T &src7, const T &src8, const T &srca, int mode)
{
    if (mode == 0)
    {
        return srca;
    }
    std::vector<T> vals = {src0, src1, src2, src3, src4, src5, src6, src7, src8};
    std::sort(vals.begin(), vals.end());
    if (mode >= 1 && mode <= 4)
    {
        T vmin = vals[mode - 1];
        T vmax = vals[9 - mode];
        return VSMIN(VSMAX(srca, vmin), vmax);
    }
    if (mode >= 11 && mode <= 14)
    {
        T vmin = VSMIN(src4, vals[mode - 11]);
        T vmax = VSMAX(src4, vals[19 - mode]);
        return VSMIN(VSMAX(srca, vmin), vmax);
    }
    return 0;
}


struct spRepairData
{
    VSNodeRef *node;
    VSNodeRef *repairnode;
    VSNodeRef *bil_nodes[8]; // 1, 2, 3, 4, repairnode=5, 6, 7, 8, 9
    const VSVideoInfo *vi;
    int mode[3];
};


static void VS_CC spRepairInit(VSMap *in, VSMap *out, void **instanceData, VSNode *node, VSCore *core, const VSAPI *vsapi)
{
    spRepairData *d = reinterpret_cast<spRepairData *>(*instanceData);
    vsapi->setVideoInfo(d->vi, 1, node);
}


static const VSFrameRef *VS_CC spRepairGetFrame(int n, int activationReason, void **instanceData, void **frameData, VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi)
{
    spRepairData *d = reinterpret_cast<spRepairData *>(*instanceData);

    if (activationReason == arInitial)
    {
        vsapi->requestFrameFilter(n, d->node, frameCtx);
        vsapi->requestFrameFilter(n, d->repairnode, frameCtx);
        for (int i = 0; i < 8; ++i)
        {
            vsapi->requestFrameFilter(n, d->bil_nodes[i], frameCtx);
        }
    }
    else if (activationReason == arAllFramesReady)
    {
        const VSFrameRef *src_frame = vsapi->getFrameFilter(n, d->node, frameCtx);
        const VSFrameRef *rep4_frame = vsapi->getFrameFilter(n, d->repairnode, frameCtx);
        const VSFrameRef *rep0_frame = vsapi->getFrameFilter(n, d->bil_nodes[0], frameCtx);
        const VSFrameRef *rep1_frame = vsapi->getFrameFilter(n, d->bil_nodes[1], frameCtx);
        const VSFrameRef *rep2_frame = vsapi->getFrameFilter(n, d->bil_nodes[2], frameCtx);
        const VSFrameRef *rep3_frame = vsapi->getFrameFilter(n, d->bil_nodes[3], frameCtx);
        const VSFrameRef *rep5_frame = vsapi->getFrameFilter(n, d->bil_nodes[4], frameCtx);
        const VSFrameRef *rep6_frame = vsapi->getFrameFilter(n, d->bil_nodes[5], frameCtx);
        const VSFrameRef *rep7_frame = vsapi->getFrameFilter(n, d->bil_nodes[6], frameCtx);
        const VSFrameRef *rep8_frame = vsapi->getFrameFilter(n, d->bil_nodes[7], frameCtx);

        int planes[3] = {0, 1, 2};
        const VSFrameRef *cp_planes[3] = {
            d->mode[0] > 0 ? nullptr : src_frame,
            d->mode[1] > 0 ? nullptr : src_frame,
            d->mode[2] > 0 ? nullptr : src_frame
        };
        VSFrameRef *dst_frame = vsapi->newVideoFrame2(vsapi->getFrameFormat(src_frame), vsapi->getFrameWidth(src_frame, 0), vsapi->getFrameHeight(src_frame, 0), cp_planes, planes, src_frame, core);

        if (d->vi->format->sampleType == stInteger && d->vi->format->bytesPerSample == 1)
        {
            // uint8_t
            for (int plane = 0; plane < d->vi->format->numPlanes; ++plane)
            {
                const uint8_t *srca = reinterpret_cast<const uint8_t *>(vsapi->getReadPtr(src_frame, plane));
                const uint8_t *src0 = reinterpret_cast<const uint8_t *>(vsapi->getReadPtr(rep0_frame, plane));
                const uint8_t *src1 = reinterpret_cast<const uint8_t *>(vsapi->getReadPtr(rep1_frame, plane));
                const uint8_t *src2 = reinterpret_cast<const uint8_t *>(vsapi->getReadPtr(rep2_frame, plane));
                const uint8_t *src3 = reinterpret_cast<const uint8_t *>(vsapi->getReadPtr(rep3_frame, plane));
                const uint8_t *src4 = reinterpret_cast<const uint8_t *>(vsapi->getReadPtr(rep4_frame, plane));
                const uint8_t *src5 = reinterpret_cast<const uint8_t *>(vsapi->getReadPtr(rep5_frame, plane));
                const uint8_t *src6 = reinterpret_cast<const uint8_t *>(vsapi->getReadPtr(rep6_frame, plane));
                const uint8_t *src7 = reinterpret_cast<const uint8_t *>(vsapi->getReadPtr(rep7_frame, plane));
                const uint8_t *src8 = reinterpret_cast<const uint8_t *>(vsapi->getReadPtr(rep8_frame, plane));
                uint8_t * VS_RESTRICT dst = reinterpret_cast<uint8_t *>(vsapi->getWritePtr(dst_frame, plane));

                int stride = vsapi->getStride(src_frame, plane) / sizeof(uint8_t);
                int pheight = vsapi->getFrameHeight(src_frame, plane);
                int pwidth = vsapi->getFrameWidth(src_frame, plane);

                int index = 0;
                for (int h = 0; h < pheight; ++h)
                {
                    for (int w = 0; w < pwidth; ++w)
                    {
                        dst[index] = sort9_repair_c<uint8_t>(src0[index], src1[index], src2[index], src3[index], src4[index], src5[index], src6[index], src7[index], src8[index], srca[index], d->mode[plane]);
                        index += 1;
                    }
                    index += (stride - pwidth);
                }
            }
        }
        else if (d->vi->format->sampleType == stInteger && d->vi->format->bytesPerSample == 2)
        {
            // uint16_t
            for (int plane = 0; plane < d->vi->format->numPlanes; ++plane)
            {
                const uint16_t *srca = reinterpret_cast<const uint16_t *>(vsapi->getReadPtr(src_frame, plane));
                const uint16_t *src0 = reinterpret_cast<const uint16_t *>(vsapi->getReadPtr(rep0_frame, plane));
                const uint16_t *src1 = reinterpret_cast<const uint16_t *>(vsapi->getReadPtr(rep1_frame, plane));
                const uint16_t *src2 = reinterpret_cast<const uint16_t *>(vsapi->getReadPtr(rep2_frame, plane));
                const uint16_t *src3 = reinterpret_cast<const uint16_t *>(vsapi->getReadPtr(rep3_frame, plane));
                const uint16_t *src4 = reinterpret_cast<const uint16_t *>(vsapi->getReadPtr(rep4_frame, plane));
                const uint16_t *src5 = reinterpret_cast<const uint16_t *>(vsapi->getReadPtr(rep5_frame, plane));
                const uint16_t *src6 = reinterpret_cast<const uint16_t *>(vsapi->getReadPtr(rep6_frame, plane));
                const uint16_t *src7 = reinterpret_cast<const uint16_t *>(vsapi->getReadPtr(rep7_frame, plane));
                const uint16_t *src8 = reinterpret_cast<const uint16_t *>(vsapi->getReadPtr(rep8_frame, plane));
                uint16_t * VS_RESTRICT dst = reinterpret_cast<uint16_t *>(vsapi->getWritePtr(dst_frame, plane));

                int stride = vsapi->getStride(src_frame, plane) / sizeof(uint16_t);
                int pheight = vsapi->getFrameHeight(src_frame, plane);
                int pwidth = vsapi->getFrameWidth(src_frame, plane);

                int index = 0;
                for (int h = 0; h < pheight; ++h)
                {
                    for (int w = 0; w < pwidth; ++w)
                    {
                        dst[index] = sort9_repair_c<uint16_t>(src0[index], src1[index], src2[index], src3[index], src4[index], src5[index], src6[index], src7[index], src8[index], srca[index], d->mode[plane]);
                        index += 1;
                    }
                    index += (stride - pwidth);
                }
            }
        }
        else if (d->vi->format->sampleType == stFloat && d->vi->format->bytesPerSample == 4)
        {
            // float
            for (int plane = 0; plane < d->vi->format->numPlanes; ++plane)
            {
                const float *srca = reinterpret_cast<const float *>(vsapi->getReadPtr(src_frame, plane));
                const float *src0 = reinterpret_cast<const float *>(vsapi->getReadPtr(rep0_frame, plane));
                const float *src1 = reinterpret_cast<const float *>(vsapi->getReadPtr(rep1_frame, plane));
                const float *src2 = reinterpret_cast<const float *>(vsapi->getReadPtr(rep2_frame, plane));
                const float *src3 = reinterpret_cast<const float *>(vsapi->getReadPtr(rep3_frame, plane));
                const float *src4 = reinterpret_cast<const float *>(vsapi->getReadPtr(rep4_frame, plane));
                const float *src5 = reinterpret_cast<const float *>(vsapi->getReadPtr(rep5_frame, plane));
                const float *src6 = reinterpret_cast<const float *>(vsapi->getReadPtr(rep6_frame, plane));
                const float *src7 = reinterpret_cast<const float *>(vsapi->getReadPtr(rep7_frame, plane));
                const float *src8 = reinterpret_cast<const float *>(vsapi->getReadPtr(rep8_frame, plane));
                float * VS_RESTRICT dst = reinterpret_cast<float *>(vsapi->getWritePtr(dst_frame, plane));

                int stride = vsapi->getStride(src_frame, plane) / sizeof(float);
                int pheight = vsapi->getFrameHeight(src_frame, plane);
                int pwidth = vsapi->getFrameWidth(src_frame, plane);

                int index = 0;
                for (int h = 0; h < pheight; ++h)
                {
                    for (int w = 0; w < pwidth; ++w)
                    {
                        dst[index] = sort9_repair_c<uint16_t>(src0[index], src1[index], src2[index], src3[index], src4[index], src5[index], src6[index], src7[index], src8[index], srca[index], d->mode[plane]);
                        index += 1;
                    }
                    index += (stride - pwidth);
                }
            }
        }
        else
        {
            vsapi->setFilterError("spRepair: Input format is not supported.", frameCtx);
            vsapi->freeFrame(dst_frame);
            dst_frame = nullptr;
        }
        vsapi->freeFrame(src_frame);
        vsapi->freeFrame(rep0_frame);
        vsapi->freeFrame(rep1_frame);
        vsapi->freeFrame(rep2_frame);
        vsapi->freeFrame(rep3_frame);
        vsapi->freeFrame(rep4_frame);
        vsapi->freeFrame(rep5_frame);
        vsapi->freeFrame(rep6_frame);
        vsapi->freeFrame(rep7_frame);
        vsapi->freeFrame(rep8_frame);
        return dst_frame;
    }
    return nullptr;
}


static void VS_CC spRepairFree(void *instanceData, VSCore *core, const VSAPI *vsapi)
{
    spRepairData *d = reinterpret_cast<spRepairData *>(instanceData);
    vsapi->freeNode(d->node);
    vsapi->freeNode(d->repairnode);
    for (int i = 0; i < 8; ++i)
    {
        vsapi->freeNode(d->bil_nodes[i]);
    }
    delete d;
}


inline VSNodeRef *invokeBilinear(VSPlugin *resize_plugin, VSNodeRef *clip, double src_left, double src_top, const VSAPI *vsapi)
{
    VSMap *args = vsapi->createMap();
    vsapi->propSetNode(args, "clip", clip, paAppend);
    vsapi->propSetFloat(args, "src_left", src_left, paAppend);
    vsapi->propSetFloat(args, "src_top", src_top, paAppend);
    VSMap *res = vsapi->invoke(resize_plugin, "Bilinear", args);
    vsapi->freeMap(args);
    VSNodeRef *res_node = vsapi->propGetNode(res, "clip", 0, nullptr);
    vsapi->freeMap(res);
    return res_node;
}


void VS_CC spRepairCreate(const VSMap *in, VSMap *out, void *userData, VSCore *core, const VSAPI *vsapi)
{
    spRepairData d;

    d.node = vsapi->propGetNode(in, "clip", 0, nullptr);
    d.vi = vsapi->getVideoInfo(d.node);

    if (!isConstantFormat(d.vi))
    {
        vsapi->freeNode(d.node);
        vsapi->setError(out, "spRepair: Only constant format input is supported.");
        return;
    }

    d.repairnode = vsapi->propGetNode(in, "repairclip", 0, nullptr);

    if (!isSameFormat(d.vi, vsapi->getVideoInfo(d.repairnode)))
    {
        vsapi->freeNode(d.node);
        vsapi->freeNode(d.repairnode);
        vsapi->setError(out, "spRepair: Input clips must have the same format.");
        return;
    }

    int num_pl = d.vi->format->numPlanes;
    int m = vsapi->propNumElements(in, "mode");
    if (num_pl < m)
    {
        vsapi->freeNode(d.node);
        vsapi->freeNode(d.repairnode);
        vsapi->setError(out, "spRepair: Number of modes specified must be equal or fewer than the number of input planes.");
        return;
    }

    bool all_modes_are_zero = true;
    for (int i = 0; i < 3; ++i)
    {
        if (i < m)
        {
            d.mode[i] = int64ToIntS(vsapi->propGetInt(in, "mode", i, nullptr));
            if (d.mode[i] < 0 || d.mode[i] > 24)
            {
                vsapi->freeNode(d.node);
                vsapi->freeNode(d.repairnode);
                vsapi->setError(out, "spRepair: Invalid mode specified, only 0-24 supported.");
                return;
            }
            else if ((d.mode[i] >= 5 && d.mode[i] <= 10) || (d.mode[i] >= 15))
            {
                vsapi->freeNode(d.node);
                vsapi->freeNode(d.repairnode);
                vsapi->setError(out, "spRepair: This mode is not yet implemented.");
                return;
            }
            if (d.mode[i] != 0)
            {
                all_modes_are_zero = false;
            }
        }
        else
        {
            d.mode[i] = d.mode[i - 1];
        }
    }

    if (all_modes_are_zero)
    {
        // Return without processing
        vsapi->propSetNode(out, "clip", d.node, paReplace);
        vsapi->freeNode(d.node);
        vsapi->freeNode(d.repairnode);
        return;
    }

    int err;
    double pixel = vsapi->propGetFloat(in, "pixel", 0, &err);
    if (err)
    {
        pixel = 1.0;
    }
    if (pixel < 0.0)
    {
        pixel = -pixel;
    }
    if (pixel > 20.0)
    {
        vsapi->freeNode(d.node);
        vsapi->freeNode(d.repairnode);
        vsapi->setError(out, "spRepair: The pixel value is considered to be too large.");
        return;
    }

    // Invoke core.resize.Bilinear to generate ...
    VSPlugin *resize_plugin = vsapi->getPluginById("com.vapoursynth.resize", core);
    // 1, top-left
    d.bil_nodes[0] = invokeBilinear(resize_plugin, d.repairnode, pixel, pixel, vsapi);
    // 2, top
    d.bil_nodes[1] = invokeBilinear(resize_plugin, d.repairnode, 0, pixel, vsapi);
    // 3, top-right
    d.bil_nodes[2] = invokeBilinear(resize_plugin, d.repairnode, -pixel, pixel, vsapi);
    // 4, left
    d.bil_nodes[3] = invokeBilinear(resize_plugin, d.repairnode, pixel, 0, vsapi);
    // 6, right
    d.bil_nodes[4] = invokeBilinear(resize_plugin, d.repairnode, -pixel, 0, vsapi);
    // 7, bottom-left
    d.bil_nodes[5] = invokeBilinear(resize_plugin, d.repairnode, pixel, -pixel, vsapi);
    // 8, bottom
    d.bil_nodes[6] = invokeBilinear(resize_plugin, d.repairnode, 0, -pixel, vsapi);
    // 9, bottom-right
    d.bil_nodes[7] = invokeBilinear(resize_plugin, d.repairnode, -pixel, -pixel, vsapi);

    spRepairData *data = new spRepairData(d);

    vsapi->createFilter(in, out, "spRepair", spRepairInit, spRepairGetFrame, spRepairFree, fmParallel, 0, data, core);
}

VS_EXTERNAL_API(void) VapourSynthPluginInit(VSConfigPlugin configFunc, VSRegisterFunction registerFunc, VSPlugin *plugin)
{
    configFunc("yomiko.collection.sprepair", "sprep", "Sub-pixel repair", VAPOURSYNTH_API_VERSION, 1, plugin);

    registerFunc("spRepair",
        "clip:clip;"
        "repairclip:clip;"
        "mode:int[];"
        "pixel:float",
        spRepairCreate, nullptr, plugin);
}
