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

#include "image_adjust_brightness_driver.hpp"
#include "image_adjust_hue_driver.hpp"
#include "image_adjust_saturation_driver.hpp"
#include "image_normalize_driver.hpp"
#include "registry_driver_maker.hpp"

static Driver* makeDriver(const std::string& base_arg)
{
    // AdjustHue
    if(base_arg == "image_adjust_hue")
        return new ImageAdjustHueDriver<float, float>();
    if(base_arg == "image_adjust_hue_fp16")
        return new ImageAdjustHueDriver<float16, float>();
    if(base_arg == "image_adjust_hue_bfp16")
        return new ImageAdjustHueDriver<bfloat16, float>();

    // AdjustBrightness
    if(base_arg == "image_adjust_brightness")
        return new ImageAdjustBrightnessDriver<float, float>();
    if(base_arg == "image_adjust_brightness_fp16")
        return new ImageAdjustBrightnessDriver<float16, float>();
    if(base_arg == "image_adjust_brightness_bfp16")
        return new ImageAdjustBrightnessDriver<bfloat16, float>();

    // Normalize
    if(base_arg == "image_normalize")
        return new ImageNormalizeDriver<float, float>();
    if(base_arg == "image_normalize_fp16")
        return new ImageNormalizeDriver<float16, float>();
    if(base_arg == "image_normalize_bfp16")
        return new ImageNormalizeDriver<bfloat16, float>();

    // AdjustSaturation
    if(base_arg == "image_adjust_saturation")
        return new ImageAdjustSaturationDriver<float, float>();
    if(base_arg == "image_adjust_saturation_fp16")
        return new ImageAdjustSaturationDriver<float16, float>();
    if(base_arg == "image_adjust_saturation_bfp16")
        return new ImageAdjustSaturationDriver<bfloat16, float>();

    return nullptr;
}

REGISTER_DRIVER_MAKER(makeDriver);
