/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include "miopen/tensor_view_utils.hpp"

template <typename DTYPE>
DTYPE bilinear_interpolate(const DTYPE* __restrict__ input,
                           const long roi_batch_index,
                           const long c,
                           long height,
                           long width,
                           DTYPE y,
                           DTYPE x,
                           tensor_view_t<4> input_tv)
{
    long y_low;
    long x_low;
    long y_high, x_high;
    DTYPE ly, lx, hy, hx;

    DTYPE v1, v2, v3, v4;
    DTYPE w1, w2, w3, w4;
    DTYPE val;

    if(y < -1.0f || y > height || x < -1.0f || x > width)
    {
        return static_cast<DTYPE>(0.0f);
    }

    if(y <= 0)
    {
        y = 0;
    }
    if(x <= 0)
    {
        x = 0;
    }

    y_low = (long)y;
    x_low = (long)x;

    if(y_low >= height - 1)
    {
        y_high = y_low = height - 1;
        y              = (DTYPE)y_low;
    }
    else
    {
        y_high = y_low + 1;
    }

    if(x_low >= width - 1)
    {
        x_high = x_low = width - 1;
        x              = (DTYPE)x_low;
    }
    else
    {
        x_high = x_low + 1;
    }

    if((y_low < 0) || (y_high > height - 1) || (x_low < 0) || (x_high > width - 1) ||
       (roi_batch_index < 0) || (roi_batch_index > input_tv.size[0] - 1))
    {
        return static_cast<DTYPE>(0.0f);
    }

    ly = y - y_low;
    lx = x - x_low;
    hy = 1.0f - ly;
    hx = 1.0f - lx;

    // v1 = GET_4D_VAL_AT(input, roi_batch_index, c, y_low, x_low);
    // v2 = GET_4D_VAL_AT(input, roi_batch_index, c, y_low, x_high);
    // v3 = GET_4D_VAL_AT(input, roi_batch_index, c, y_high, x_low);
    // v4 = GET_4D_VAL_AT(input, roi_batch_index, c, y_high, x_high);

    v1 = input[input_tv.get_tensor_view_idx({roi_batch_index, c, y_low, x_low})];
    v2 = input[input_tv.get_tensor_view_idx({roi_batch_index, c, y_low, x_high})];
    v3 = input[input_tv.get_tensor_view_idx({roi_batch_index, c, y_high, x_low})];
    v4 = input[input_tv.get_tensor_view_idx({roi_batch_index, c, y_high, x_high})];

    w1 = hy * hx;
    w2 = hy * lx;
    w3 = ly * hx;
    w4 = ly * lx;

    val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);

    return val;
}

template <typename Tgpu, typename Tref>
void mloRoIAlignForward(const Tgpu* __restrict__ input,
                        const Tgpu* __restrict__ rois,
                        Tref* __restrict__ output,
                        int output_h,
                        int output_w,
                        float spatial_scale,
                        int sampling_ratio,
                        bool aligned,
                        int roi_batch_base_idx,
                        tensor_view_t<4> input_tv,
                        tensor_view_t<2> rois_tv,
                        tensor_view_t<4> output_tv)
{
    for(auto gid = 0; gid < rois_tv.size[0] * input_tv.size[1] * output_h * output_w; gid++)
    {

        long N = input_tv.size[0];
        long C = input_tv.size[1];
        long H = input_tv.size[2];
        long W = input_tv.size[3];
        long K = rois_tv.size[0];

        if(gid > K * C * output_h * output_w - 1)
        {
            return;
        }

        long kch = gid / output_w;
        long kc  = kch / output_h;

        long pw = gid % output_w;
        long ph = kch % output_h;
        long c  = kc % C;
        long k  = kc / C;

        // long roi_batch_index = (long)(GET_2D_VAL_AT(rois, k, 0));
        Tref roi_batch_index = rois[rois_tv.get_tensor_view_idx({k, 0})];
        roi_batch_index -= roi_batch_base_idx;

        if(roi_batch_index < 0 || roi_batch_index >= N)
        {
            // SET_4D_VAL_AT(output, k, c, ph, pw, 0);
            output[output_tv.get_tensor_view_idx({k, c, ph, pw})] = 0;
            return;
        }

        Tref roi_offset = aligned ? 0.5f : 0;
        // DTYPE roi_start_w = GET_2D_VAL_AT(rois, k, 1) * spatial_scale - roi_offset;
        // DTYPE roi_start_h = GET_2D_VAL_AT(rois, k, 2) * spatial_scale - roi_offset;
        // DTYPE roi_end_w   = GET_2D_VAL_AT(rois, k, 3) * spatial_scale - roi_offset;
        // DTYPE roi_end_h   = GET_2D_VAL_AT(rois, k, 4) * spatial_scale - roi_offset;

        // tensor_layout_t<2> rois_layout = tensor_layout_t<2>(rois_tv, k * C + 1);
        Tref roi_start_w = rois[rois_tv.get_tensor_view_idx({k, 1})] * spatial_scale - roi_offset;
        Tref roi_start_h = rois[rois_tv.get_tensor_view_idx({k, 2})] * spatial_scale - roi_offset;
        Tref roi_end_w   = rois[rois_tv.get_tensor_view_idx({k, 3})] * spatial_scale - roi_offset;
        Tref roi_end_h   = rois[rois_tv.get_tensor_view_idx({k, 4})] * spatial_scale - roi_offset;

        Tref roi_width  = roi_end_w - roi_start_w;
        Tref roi_height = roi_end_h - roi_start_h;

        Tref bin_size_h;
        Tref bin_size_w;

        long roi_bin_grid_h;
        long roi_bin_grid_w;

        Tref count;

        Tref output_val = 0.0f;

        long iy, ix;

        if(!aligned)
        {
            roi_width  = fmax((Tref)roi_width, (Tref)1.0f);
            roi_height = fmax((Tref)roi_height, (Tref)1.0f);
        }
        bin_size_h = roi_height / output_h;
        bin_size_w = roi_width / output_w;

        roi_bin_grid_h = (sampling_ratio > 0) ? sampling_ratio : ceil(roi_height / output_h);
        roi_bin_grid_w = (sampling_ratio > 0) ? sampling_ratio : ceil(roi_width / output_w);

        roi_bin_grid_h = (isnan(roi_height) || (roi_bin_grid_h > H)) ? H : roi_bin_grid_h;
        roi_bin_grid_w = (isnan(roi_width) || (roi_bin_grid_w > W)) ? W : roi_bin_grid_w;
        bin_size_h     = (isnan(roi_height) || (bin_size_h > H)) ? H : bin_size_h;
        bin_size_w     = (isnan(roi_width) || (bin_size_w > W)) ? W : bin_size_w;

        count = roi_bin_grid_h * roi_bin_grid_w;

        for(iy = 0; iy < roi_bin_grid_h; ++iy)
        {
            const Tgpu y = static_cast<Tgpu>(roi_start_h + ph * bin_size_h +
                                             (iy + 0.5f) * bin_size_h / roi_bin_grid_h);
            for(ix = 0; ix < roi_bin_grid_w; ++ix)
            {
                const Tgpu x = static_cast<Tgpu>(roi_start_w + pw * bin_size_w +
                                                 (ix + 0.5f) * bin_size_w / roi_bin_grid_w);
                Tref val     = static_cast<Tref>(
                    bilinear_interpolate<Tgpu>(input, roi_batch_index, c, H, W, y, x, input_tv));
                output_val += val;
            }
        }
        output_val /= count;
        // SET_4D_VAL_AT(output, k, c, ph, pw, output_val);
        output[output_tv.get_tensor_view_idx({k, c, ph, pw})] = output_val;
    }
}
