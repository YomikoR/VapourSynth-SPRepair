#include "vapoursynth/VapourSynth4.h"
#include "vapoursynth/VSHelper4.h"

#include <immintrin.h>
#include <algorithm>
#include <utility>
#include <memory>
#include <vector>

/* 
 * Ref for the sorting network used: Vinod K Valsalam and Risto Miikkulainen,
 * Using Symmetry and Evolutionary Search to Minimize Sorting Networks,
 * Journal of Machine Learning Research 14 (2013) 303-331.
 */

#if defined(_MSC_VER)
  #define ALWAYS_INLINE __forceinline
#elif defined(__GNUC__)
  #define ALWAYS_INLINE __attribute__((always_inline))
#else
  #define ALWAYS_INLINE
#endif

template <int i, int j>
inline void ALWAYS_INLINE swap_u8(__m256i *val)
{
    __m256i tmp = _mm256_min_epu8(val[i], val[j]);
    val[j] = _mm256_max_epu8(val[i], val[j]);
    val[i] = tmp;
}


template <int i, int j>
inline void ALWAYS_INLINE swap_u16(__m256i *val)
{
    __m256i tmp = _mm256_min_epu16(val[i], val[j]);
    val[j] = _mm256_max_epu16(val[i], val[j]);
    val[i] = tmp;
}


template <int i, int j>
inline void ALWAYS_INLINE swap_f(__m256 *val)
{
    __m256 tmp = _mm256_min_ps(val[i], val[j]);
    val[j] = _mm256_max_ps(val[i], val[j]);
    val[i] = tmp;
}


inline __m256i ALWAYS_INLINE loadu(const uint8_t *src)
{
    return _mm256_loadu_si256(reinterpret_cast<const __m256i *>(src));
}


inline __m256i ALWAYS_INLINE loadu(const uint16_t *src)
{
    return _mm256_loadu_si256(reinterpret_cast<const __m256i *>(src));
}


inline __m256 ALWAYS_INLINE loadu(const float *src)
{
    return _mm256_loadu_ps(src);
}


inline void ALWAYS_INLINE storeu(uint8_t *dst, __m256i value)
{
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(dst), value);
}


inline void ALWAYS_INLINE storeu(uint16_t *dst, __m256i value)
{
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(dst), value);
}


inline void ALWAYS_INLINE storeu(float *dst, __m256 value)
{
    _mm256_storeu_ps(dst, value);
}


#define DO_LOADU \
    loadu(src0), loadu(src1), loadu(src2), \
    loadu(src3), loadu(src4), loadu(src5), \
    loadu(src6), loadu(src7), loadu(src8),


template <int order>
std::pair<__m256i, __m256i> sort9_minmax_u8(const uint8_t *src0,
    const uint8_t *src1, const uint8_t *src2, const uint8_t *src3,
    const uint8_t *src4, const uint8_t *src5, const uint8_t *src6,
    const uint8_t *src7, const uint8_t *src8)
{
    __m256i val[9] = {
DO_LOADU
    };

#define SWAP swap_u8

    SWAP<2, 6>(val); SWAP<0, 5>(val); SWAP<1, 4>(val);
    SWAP<7, 8>(val); SWAP<0, 7>(val); SWAP<1, 2>(val);
    SWAP<3, 5>(val); SWAP<4, 6>(val); SWAP<5, 8>(val);

    SWAP<1, 3>(val); SWAP<6, 8>(val); SWAP<0, 1>(val);
    SWAP<4, 5>(val); SWAP<2, 7>(val); SWAP<3, 7>(val);
    SWAP<3, 4>(val); SWAP<5, 6>(val); SWAP<1, 2>(val);

    SWAP<1, 3>(val); SWAP<6, 7>(val); SWAP<4, 5>(val);

    SWAP<2, 4>(val); SWAP<5, 6>(val);
    SWAP<2, 3>(val); SWAP<4, 5>(val);

#undef SWAP

    return std::pair<__m256i, __m256i>(val[order], val[8 - order]);
}


template<int order>
std::pair<__m256i, __m256i> sort9_minmax_u16(const uint16_t *src0,
    const uint16_t *src1, const uint16_t *src2, const uint16_t *src3,
    const uint16_t *src4, const uint16_t *src5, const uint16_t *src6,
    const uint16_t *src7, const uint16_t *src8)
{
    __m256i val[9] = {
DO_LOADU
    };

#define SWAP swap_u16

    SWAP<2, 6>(val); SWAP<0, 5>(val); SWAP<1, 4>(val);
    SWAP<7, 8>(val); SWAP<0, 7>(val); SWAP<1, 2>(val);
    SWAP<3, 5>(val); SWAP<4, 6>(val); SWAP<5, 8>(val);

    SWAP<1, 3>(val); SWAP<6, 8>(val); SWAP<0, 1>(val);
    SWAP<4, 5>(val); SWAP<2, 7>(val); SWAP<3, 7>(val);
    SWAP<3, 4>(val); SWAP<5, 6>(val); SWAP<1, 2>(val);

    SWAP<1, 3>(val); SWAP<6, 7>(val); SWAP<4, 5>(val);

    SWAP<2, 4>(val); SWAP<5, 6>(val);
    SWAP<2, 3>(val); SWAP<4, 5>(val);

#undef SWAP

    return std::pair<__m256i, __m256i>(val[order], val[8 - order]);
}


template <int order>
std::pair<__m256, __m256> sort9_minmax_f(const float *src0,
    const float *src1, const float *src2, const float *src3,
    const float *src4, const float *src5, const float *src6,
    const float *src7, const float *src8)
{
    __m256 val[9] = {
DO_LOADU
    };

#define SWAP swap_f

    SWAP<2, 6>(val); SWAP<0, 5>(val); SWAP<1, 4>(val);
    SWAP<7, 8>(val); SWAP<0, 7>(val); SWAP<1, 2>(val);
    SWAP<3, 5>(val); SWAP<4, 6>(val); SWAP<5, 8>(val);

    SWAP<1, 3>(val); SWAP<6, 8>(val); SWAP<0, 1>(val);
    SWAP<4, 5>(val); SWAP<2, 7>(val); SWAP<3, 7>(val);
    SWAP<3, 4>(val); SWAP<5, 6>(val); SWAP<1, 2>(val);

    SWAP<1, 3>(val); SWAP<6, 7>(val); SWAP<4, 5>(val);

    SWAP<2, 4>(val); SWAP<5, 6>(val);
    SWAP<2, 3>(val); SWAP<4, 5>(val);

#undef SWAP

    return std::pair<__m256, __m256>(val[order], val[8 - order]);
}


inline __m256i ALWAYS_INLINE set_minmax_u8(__m256i vmin, __m256i vmax, __m256i va)
{
    va = _mm256_min_epu8(va, vmax);
    va = _mm256_max_epu8(va, vmin);
    return va;
}


inline __m256i ALWAYS_INLINE set_minmax_u8(__m256i vmin, __m256i vmax, __m256i va, __m256i v4)
{
    vmin = _mm256_min_epu8(vmin, v4);
    vmax = _mm256_max_epu8(vmax, v4);
    va = _mm256_min_epu8(va, vmax);
    va = _mm256_max_epu8(va, vmin);
    return va;
}


inline __m256i ALWAYS_INLINE set_minmax_u16(__m256i vmin, __m256i vmax, __m256i va)
{
    va = _mm256_min_epu16(va, vmax);
    va = _mm256_max_epu16(va, vmin);
    return va;
}


inline __m256i ALWAYS_INLINE set_minmax_u16(__m256i vmin, __m256i vmax, __m256i va, __m256i v4)
{
    vmin = _mm256_min_epu16(vmin, v4);
    vmax = _mm256_max_epu16(vmax, v4);
    va = _mm256_min_epu16(va, vmax);
    va = _mm256_max_epu16(va, vmin);
    return va;
}


inline __m256 ALWAYS_INLINE set_minmax_f(__m256 vmin, __m256 vmax, __m256 va)
{
    va = _mm256_min_ps(va, vmax);
    va = _mm256_max_ps(va, vmin);
    return va;
}


inline __m256 ALWAYS_INLINE set_minmax_f(__m256 vmin, __m256 vmax, __m256 va, __m256 v4)
{
    vmin = _mm256_min_ps(vmin, v4);
    vmax = _mm256_max_ps(vmax, v4);
    va = _mm256_min_ps(va, vmax);
    va = _mm256_max_ps(va, vmin);
    return va;
}


static void sort9_repair_u8(const uint8_t *src0,
    const uint8_t *src1, const uint8_t *src2, const uint8_t *src3,
    const uint8_t *src4, const uint8_t *src5, const uint8_t *src6,
    const uint8_t *src7, const uint8_t *src8, const uint8_t *srca,
    uint8_t *dst, int mode)
{
    if (mode == 0)
    {
        // pass
    }
    else if (mode == 1)
    {
        std::pair<__m256i, __m256i> minmax = sort9_minmax_u8<0>(src0, src1, src2, src3, src4, src5, src6, src7, src8);
        __m256i va = set_minmax_u8(minmax.first, minmax.second, loadu(srca));
        storeu(dst, va);
    }
    else if (mode == 2)
    {
        std::pair<__m256i, __m256i> minmax = sort9_minmax_u8<1>(src0, src1, src2, src3, src4, src5, src6, src7, src8);
        __m256i va = set_minmax_u8(minmax.first, minmax.second, loadu(srca));
        storeu(dst, va);
    }
    else if (mode == 3)
    {
        std::pair<__m256i, __m256i> minmax = sort9_minmax_u8<2>(src0, src1, src2, src3, src4, src5, src6, src7, src8);
        __m256i va = set_minmax_u8(minmax.first, minmax.second, loadu(srca));
        storeu(dst, va);
    }
    else if (mode == 4)
    {
        std::pair<__m256i, __m256i> minmax = sort9_minmax_u8<3>(src0, src1, src2, src3, src4, src5, src6, src7, src8);
        __m256i va = set_minmax_u8(minmax.first, minmax.second, loadu(srca));
        storeu(dst, va);
    }
    else if (mode == 11)
    {
        std::pair<__m256i, __m256i> minmax = sort9_minmax_u8<0>(src0, src1, src2, src3, src4, src5, src6, src7, src8);
        __m256i va = set_minmax_u8(minmax.first, minmax.second, loadu(srca), loadu(src4));
        storeu(dst, va);
    }
    else if (mode == 12)
    {
        std::pair<__m256i, __m256i> minmax = sort9_minmax_u8<1>(src0, src1, src2, src3, src4, src5, src6, src7, src8);
        __m256i va = set_minmax_u8(minmax.first, minmax.second, loadu(srca), loadu(src4));
        storeu(dst, va);
    }
    else if (mode == 13)
    {
        std::pair<__m256i, __m256i> minmax = sort9_minmax_u8<2>(src0, src1, src2, src3, src4, src5, src6, src7, src8);
        __m256i va = set_minmax_u8(minmax.first, minmax.second, loadu(srca), loadu(src4));
        storeu(dst, va);
    }
    else if (mode == 14)
    {
        std::pair<__m256i, __m256i> minmax = sort9_minmax_u8<3>(src0, src1, src2, src3, src4, src5, src6, src7, src8);
        __m256i va = set_minmax_u8(minmax.first, minmax.second, loadu(srca), loadu(src4));
        storeu(dst, va);
    }
}


static void sort9_repair_u16(const uint16_t *src0,
    const uint16_t *src1, const uint16_t *src2, const uint16_t *src3,
    const uint16_t *src4, const uint16_t *src5, const uint16_t *src6,
    const uint16_t *src7, const uint16_t *src8, const uint16_t *srca,
    uint16_t *dst, int mode)
{
    if (mode == 0)
    {
        // pass
    }
    else if (mode == 1)
    {
        std::pair<__m256i, __m256i> minmax = sort9_minmax_u16<0>(src0, src1, src2, src3, src4, src5, src6, src7, src8);
        __m256i va = set_minmax_u16(minmax.first, minmax.second, loadu(srca));
        storeu(dst, va);
    }
    else if (mode == 2)
    {
        std::pair<__m256i, __m256i> minmax = sort9_minmax_u16<1>(src0, src1, src2, src3, src4, src5, src6, src7, src8);
        __m256i va = set_minmax_u16(minmax.first, minmax.second, loadu(srca));
        storeu(dst, va);
    }
    else if (mode == 3)
    {
        std::pair<__m256i, __m256i> minmax = sort9_minmax_u16<2>(src0, src1, src2, src3, src4, src5, src6, src7, src8);
        __m256i va = set_minmax_u16(minmax.first, minmax.second, loadu(srca));
        storeu(dst, va);
    }
    else if (mode == 4)
    {
        std::pair<__m256i, __m256i> minmax = sort9_minmax_u16<3>(src0, src1, src2, src3, src4, src5, src6, src7, src8);
        __m256i va = set_minmax_u16(minmax.first, minmax.second, loadu(srca));
        storeu(dst, va);
    }
    else if (mode == 11)
    {
        std::pair<__m256i, __m256i> minmax = sort9_minmax_u16<0>(src0, src1, src2, src3, src4, src5, src6, src7, src8);
        __m256i va = set_minmax_u16(minmax.first, minmax.second, loadu(srca), loadu(src4));
        storeu(dst, va);
    }
    else if (mode == 12)
    {
        std::pair<__m256i, __m256i> minmax = sort9_minmax_u16<1>(src0, src1, src2, src3, src4, src5, src6, src7, src8);
        __m256i va = set_minmax_u16(minmax.first, minmax.second, loadu(srca), loadu(src4));
        storeu(dst, va);
    }
    else if (mode == 13)
    {
        std::pair<__m256i, __m256i> minmax = sort9_minmax_u16<2>(src0, src1, src2, src3, src4, src5, src6, src7, src8);
        __m256i va = set_minmax_u16(minmax.first, minmax.second, loadu(srca), loadu(src4));
        storeu(dst, va);
    }
    else if (mode == 14)
    {
        std::pair<__m256i, __m256i> minmax = sort9_minmax_u16<3>(src0, src1, src2, src3, src4, src5, src6, src7, src8);
        __m256i va = set_minmax_u16(minmax.first, minmax.second, loadu(srca), loadu(src4));
        storeu(dst, va);
    }
}


static void sort9_repair_f(const float *src0,
    const float *src1, const float *src2, const float *src3,
    const float *src4, const float *src5, const float *src6,
    const float *src7, const float *src8, const float *srca,
    float *dst, int mode)
{
    if (mode == 0)
    {
        // pass
    }
    else if (mode == 1)
    {
        std::pair<__m256, __m256> minmax = sort9_minmax_f<0>(src0, src1, src2, src3, src4, src5, src6, src7, src8);
        __m256 va = set_minmax_f(minmax.first, minmax.second, loadu(srca));
        storeu(dst, va);
    }
    else if (mode == 2)
    {
        std::pair<__m256, __m256> minmax = sort9_minmax_f<1>(src0, src1, src2, src3, src4, src5, src6, src7, src8);
        __m256 va = set_minmax_f(minmax.first, minmax.second, loadu(srca));
        storeu(dst, va);
    }
    else if (mode == 3)
    {
        std::pair<__m256, __m256> minmax = sort9_minmax_f<2>(src0, src1, src2, src3, src4, src5, src6, src7, src8);
        __m256 va = set_minmax_f(minmax.first, minmax.second, loadu(srca));
        storeu(dst, va);
    }
    else if (mode == 4)
    {
        std::pair<__m256, __m256> minmax = sort9_minmax_f<3>(src0, src1, src2, src3, src4, src5, src6, src7, src8);
        __m256 va = set_minmax_f(minmax.first, minmax.second, loadu(srca));
        storeu(dst, va);
    }
    else if (mode == 11)
    {
        std::pair<__m256, __m256> minmax = sort9_minmax_f<0>(src0, src1, src2, src3, src4, src5, src6, src7, src8);
        __m256 va = set_minmax_f(minmax.first, minmax.second, loadu(srca), loadu(src4));
        storeu(dst, va);
    }
    else if (mode == 12)
    {
        std::pair<__m256, __m256> minmax = sort9_minmax_f<1>(src0, src1, src2, src3, src4, src5, src6, src7, src8);
        __m256 va = set_minmax_f(minmax.first, minmax.second, loadu(srca), loadu(src4));
        storeu(dst, va);
    }
    else if (mode == 13)
    {
        std::pair<__m256, __m256> minmax = sort9_minmax_f<2>(src0, src1, src2, src3, src4, src5, src6, src7, src8);
        __m256 va = set_minmax_f(minmax.first, minmax.second, loadu(srca), loadu(src4));
        storeu(dst, va);
    }
    else if (mode == 14)
    {
        std::pair<__m256, __m256> minmax = sort9_minmax_f<3>(src0, src1, src2, src3, src4, src5, src6, src7, src8);
        __m256 va = set_minmax_f(minmax.first, minmax.second, loadu(srca), loadu(src4));
        storeu(dst, va);
    }
}


struct spRepairData
{
    VSNode *node;
    VSNode *repairnode;
    VSNode *bil_nodes[8]; // 1, 2, 3, 4, repairnode=5, 6, 7, 8, 9
    VSVideoInfo vi;
    int mode[3];
};


static const VSFrame *VS_CC spRepairGetFrame(int n, int activationReason, void *instanceData, void **frameData, VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi)
{
    spRepairData *d = reinterpret_cast<spRepairData *>(instanceData);

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
        const VSFrame *src_frame = vsapi->getFrameFilter(n, d->node, frameCtx);
        const VSFrame *rep4_frame = vsapi->getFrameFilter(n, d->repairnode, frameCtx);
        const VSFrame *rep0_frame = vsapi->getFrameFilter(n, d->bil_nodes[0], frameCtx);
        const VSFrame *rep1_frame = vsapi->getFrameFilter(n, d->bil_nodes[1], frameCtx);
        const VSFrame *rep2_frame = vsapi->getFrameFilter(n, d->bil_nodes[2], frameCtx);
        const VSFrame *rep3_frame = vsapi->getFrameFilter(n, d->bil_nodes[3], frameCtx);
        const VSFrame *rep5_frame = vsapi->getFrameFilter(n, d->bil_nodes[4], frameCtx);
        const VSFrame *rep6_frame = vsapi->getFrameFilter(n, d->bil_nodes[5], frameCtx);
        const VSFrame *rep7_frame = vsapi->getFrameFilter(n, d->bil_nodes[6], frameCtx);
        const VSFrame *rep8_frame = vsapi->getFrameFilter(n, d->bil_nodes[7], frameCtx);

        int planes[3] = {0, 1, 2};
        const VSFrame *cp_planes[3] = {
            d->mode[0] > 0 ? nullptr : src_frame,
            d->mode[1] > 0 ? nullptr : src_frame,
            d->mode[2] > 0 ? nullptr : src_frame
        };
        VSFrame *dst_frame = vsapi->newVideoFrame2(vsapi->getVideoFrameFormat(src_frame), vsapi->getFrameWidth(src_frame, 0), vsapi->getFrameHeight(src_frame, 0), cp_planes, planes, src_frame, core);

        if (d->vi.format.sampleType == stInteger && d->vi.format.bytesPerSample == 1)
        {
            // uint8_t
            for (int plane = 0; plane < d->vi.format.numPlanes; ++plane)
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

                int index = 0;
                for (int h = 0; h < pheight; ++h)
                {
                    for (int w = 0; w < stride / 32; ++w)
                    {
                        sort9_repair_u8(src0 + index, src1 + index, src2 + index, src3 + index, src4 + index, src5 + index, src6 + index, src7 + index, src8 + index, srca + index, dst + index, d->mode[plane]);
                        index += 32;
                    }
                }
            }
        }
        else if (d->vi.format.sampleType == stInteger && d->vi.format.bytesPerSample == 2)
        {
            // uint16_t
            for (int plane = 0; plane < d->vi.format.numPlanes; ++plane)
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

                int index = 0;
                for (int h = 0; h < pheight; ++h)
                {
                    for (int w = 0; w < stride / 16; ++w)
                    {
                        sort9_repair_u16(src0 + index, src1 + index, src2 + index, src3 + index, src4 + index, src5 + index, src6 + index, src7 + index, src8 + index, srca + index, dst + index, d->mode[plane]);
                        index += 16;
                    }
                }
            }
        }
        else if (d->vi.format.sampleType == stFloat && d->vi.format.bytesPerSample == 4)
        {
            // float
            for (int plane = 0; plane < d->vi.format.numPlanes; ++plane)
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

                int index = 0;
                for (int h = 0; h < pheight; ++h)
                {
                    for (int w = 0; w < stride / 8; ++w)
                    {
                        sort9_repair_f(src0 + index, src1 + index, src2 + index, src3 + index, src4 + index, src5 + index, src6 + index, src7 + index, src8 + index, srca + index, dst + index, d->mode[plane]);
                        index += 8;
                    }
                }
            }
        }
        else
        {
            vsapi->setFilterError("spRepair: Input format is not supported->", frameCtx);
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


inline VSNode *invokeBilinear(VSPlugin *resize_plugin, VSNode *clip, double src_left, double src_top, const VSAPI *vsapi)
{
    VSMap *args = vsapi->createMap();
    vsapi->mapSetNode(args, "clip", clip, maAppend);
    vsapi->mapSetFloat(args, "src_left", src_left, maAppend);
    vsapi->mapSetFloat(args, "src_top", src_top, maAppend);
    VSMap *res = vsapi->invoke(resize_plugin, "Bilinear", args);
    vsapi->freeMap(args);
    VSNode *res_node = vsapi->mapGetNode(res, "clip", 0, nullptr);
    vsapi->freeMap(res);
    return res_node;
}


void VS_CC spRepairCreate(const VSMap *in, VSMap *out, void *userData, VSCore *core, const VSAPI *vsapi)
{
    std::unique_ptr<spRepairData> d(new spRepairData());

    d->node = vsapi->mapGetNode(in, "clip", 0, nullptr);
    const VSVideoInfo *vi = vsapi->getVideoInfo(d->node);
    d->vi = *vi;

    d->repairnode = vsapi->mapGetNode(in, "repairclip", 0, nullptr);

    if (!vsh::isSameVideoFormat(&vi->format, &vsapi->getVideoInfo(d->repairnode)->format))
    {
        vsapi->freeNode(d->node);
        vsapi->freeNode(d->repairnode);
        vsapi->mapSetError(out, "spRepair: Input clips must have the same format.");
        return;
    }

    int num_pl = d->vi.format.numPlanes;
    int m = vsapi->mapNumElements(in, "mode");
    if (num_pl < m)
    {
        vsapi->freeNode(d->node);
        vsapi->freeNode(d->repairnode);
        vsapi->mapSetError(out, "spRepair: Number of modes specified must be equal or fewer than the number of input planes.");
        return;
    }

    bool all_modes_are_zero = true;
    for (int i = 0; i < 3; ++i)
    {
        if (i < m)
        {
            d->mode[i] = vsh::int64ToIntS(vsapi->mapGetInt(in, "mode", i, nullptr));
            if (d->mode[i] < 0 || d->mode[i] > 24)
            {
                vsapi->freeNode(d->node);
                vsapi->freeNode(d->repairnode);
                vsapi->mapSetError(out, "spRepair: Invalid mode specified, only 0-24 supported->");
                return;
            }
            else if ((d->mode[i] >= 5 && d->mode[i] <= 10) || (d->mode[i] >= 15))
            {
                vsapi->freeNode(d->node);
                vsapi->freeNode(d->repairnode);
                vsapi->mapSetError(out, "spRepair: This mode is not yet implemented->");
                return;
            }
            if (d->mode[i] != 0)
            {
                all_modes_are_zero = false;
            }
        }
        else
        {
            d->mode[i] = d->mode[i - 1];
        }
    }

    if (all_modes_are_zero)
    {
        // Return without processing
        vsapi->mapSetNode(out, "clip", d->node, maReplace);
        vsapi->freeNode(d->node);
        vsapi->freeNode(d->repairnode);
        return;
    }

    int err;
    double pixel = vsapi->mapGetFloat(in, "pixel", 0, &err);
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
        vsapi->freeNode(d->node);
        vsapi->freeNode(d->repairnode);
        vsapi->mapSetError(out, "spRepair: The pixel value is considered to be too large.");
        return;
    }

    // Invoke core.resize.Bilinear to generate ...
    VSPlugin *resize_plugin = vsapi->getPluginByID("com.vapoursynth.resize", core);
    // 1, top-left
    d->bil_nodes[0] = invokeBilinear(resize_plugin, d->repairnode, pixel, pixel, vsapi);
    // 2, top
    d->bil_nodes[1] = invokeBilinear(resize_plugin, d->repairnode, 0, pixel, vsapi);
    // 3, top-right
    d->bil_nodes[2] = invokeBilinear(resize_plugin, d->repairnode, -pixel, pixel, vsapi);
    // 4, left
    d->bil_nodes[3] = invokeBilinear(resize_plugin, d->repairnode, pixel, 0, vsapi);
    // 6, right
    d->bil_nodes[4] = invokeBilinear(resize_plugin, d->repairnode, -pixel, 0, vsapi);
    // 7, bottom-left
    d->bil_nodes[5] = invokeBilinear(resize_plugin, d->repairnode, pixel, -pixel, vsapi);
    // 8, bottom
    d->bil_nodes[6] = invokeBilinear(resize_plugin, d->repairnode, 0, -pixel, vsapi);
    // 9, bottom-right
    d->bil_nodes[7] = invokeBilinear(resize_plugin, d->repairnode, -pixel, -pixel, vsapi);

    std::vector<VSFilterDependency> dep_req =
    {
        {d->node, rpStrictSpatial},
        {d->repairnode, rpStrictSpatial}
    };
    for (int n = 0; n < 8; ++n)
    {
        dep_req.push_back({d->bil_nodes[n], rpStrictSpatial});
    }

    vsapi->createVideoFilter(out, "spRepair", &d->vi, spRepairGetFrame, spRepairFree, fmParallel, dep_req.data(), dep_req.size(), d.get(), core);\
    d.release();
}


VS_EXTERNAL_API(void) VapourSynthPluginInit2(VSPlugin* plugin, const VSPLUGINAPI* vspapi)
{
    vspapi->configPlugin("yomiko.collection.sprepair", "sprep", "Sub-pixel repair", VS_MAKE_VERSION(1, 0), VAPOURSYNTH_API_VERSION, 0, plugin);

    vspapi->registerFunction("spRepair",
        "clip:vnode;"
        "repairclip:vnode;"
        "mode:int[];"
        "pixel:float;",
        "clip:vnode;",
        spRepairCreate, nullptr, plugin
    );
}
