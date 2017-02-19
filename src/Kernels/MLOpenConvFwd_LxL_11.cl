/*
 * Copyright (c) 2017 AMD Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and/or associated documentation files (the
 * "Materials"), to deal in the Materials without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Materials, and to
 * permit persons to whom the Materials are furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Materials.
 *
 * THE MATERIALS ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * MATERIALS OR THE USE OR OTHER DEALINGS IN THE MATERIALS.
 */


#define _FLOAT					float
#define _FLOAT2					float2
#define _FLOAT4					float4
#define _FLOAT8					float8

#ifndef FLT_MAX
#define FLT_MAX         3.402823466e+38F        /* max value */
#endif


// filter size for all filters with small n of input maps (first layer)
// split a long filter by stride
#define MLO_N_FILTER_SPLITS0 ((MLO_FILTER_SIZE0 + MLO_FILTER_STRIDE0 - 1)/ MLO_FILTER_STRIDE0)
#define MLO_WEI_LCL_WIDTH (MLO_N_FILTER_SPLITS0*MLO_FILTER_STRIDE0)
#define MLO_WEI_SZ (MLO_FILTER_SIZE1*MLO_WEI_LCL_WIDTH)
#define MLO_WEI_LCL_SZ (MLO_WEI_SZ * MLO_OUT_STACKS * MLO_N_LCL_OUT_MAPS* MLO_N_INPUTS)


#define MLO_IN_LCL_HEIGHT ((MLO_OUT_PIX_TILE1-1)*MLO_FILTER_STRIDE1 + MLO_FILTER_SIZE1)
// there is an assumption that the scanline fits into LDS
#define MLO_N_IN_HORIZ_PIX_READS (MLO_IN_WIDTH)
#define MLO_N_IN_HORIZ_READS ((MLO_N_IN_HORIZ_PIX_READS + MLO_READ_UNIT - 1) / MLO_READ_UNIT)
#define MLO_IN_N_PIXS_OFF  (MLO_N_IN_HORIZ_PIX_READS - (MLO_N_IN_HORIZ_PIX_READS  / MLO_READ_UNIT)*MLO_READ_UNIT)

#define MLO_IN_LCL_WIDTH (MLO_N_IN_HORIZ_READS * MLO_READ_UNIT + 2 * MLO_FILTER_PAD0)
#define MLO_IN_LCL_SZ (MLO_IN_LCL_WIDTH*MLO_IN_LCL_HEIGHT)
// LDS IN SIZE
#define MLO_TOTAL_IN_LCL_SZ (MLO_N_LCL_BATCHS*MLO_IN_LCL_SZ* MLO_N_LCL_IN_MAPS)

#if (MLO_WEI_LCL_SZ + MLO_TOTAL_IN_LCL_SZ) > (MLO_N_FILTER_SPLITS0 *MLO_OUT_WIDTH*MLO_OUT_STACKS)
#define MLO_LCL_MEM_SZ (MLO_WEI_LCL_SZ + MLO_TOTAL_IN_LCL_SZ)
#else
#define MLO_LCL_MEM_SZ (MLO_N_FILTER_SPLITS0 *MLO_OUT_WIDTH*MLO_OUT_STACKS)
#endif

// number of loops to flush put full output map
#define MLO_N_OUT_BLKS ((MLO_OUT_HEIGHT + (MLO_OUT_PIX_TILE1*MLO_N_OUT_FOLDS1) -1) / (MLO_OUT_PIX_TILE1*MLO_N_OUT_FOLDS1))

#define MLO_HW_WAVE_ID_SETTING 0

#if MLO_HW_WAVE_ID_SETTING
extern __attribute__((const)) uint __hsail_get_dynwave_id(void);
static inline int getWaveId()
{
	int wave_id = 0;

	wave_id = __hsail_get_dynwave_id();
	wave_id = wave_id & MLO_N_PHYS_WAVES_MASK;
	return(wave_id);
}
#else
static inline int getWaveId()
{
	int wave_id = 0;

	wave_id = (get_local_id(0) >> MLO_LG2_PHYS_WAVE_SZ);

	return(wave_id);
}
#endif

static inline int gePhysLocalId()
{
	int lcl_wave_id = get_local_id(0) - ((get_local_id(0) >> MLO_LG2_PHYS_WAVE_SZ) << MLO_LG2_PHYS_WAVE_SZ);
	return(lcl_wave_id);
}

static inline int iDiv(int v, int d)
{
	int r = (int)((float)v / d + 0.00001f);
	return(r);
}

static inline int iMod(int v, int u, int d)
{
	int r = v - mul24((int)u, (int)d);
	return(r);
}

static inline void ReduceKernel(__local _FLOAT * lcl_blob, _FLOAT *weights_accum, int lcl_id, int scan_lcl, int sum_stride, int unit_len, bool debug)
{
	for (int j = (sum_stride >> 1); j > 0; j >>= 1)
	{
		barrier(CLK_LOCAL_MEM_FENCE);
		if (scan_lcl < j)
		{
			for (int i = 0; i < unit_len; ++i)
			{

				weights_accum[i] += lcl_blob[(lcl_id + j) * unit_len + i];

				lcl_blob[lcl_id * unit_len + i] = weights_accum[i];
			}

		}
	}
}

static inline void  Kahan_summation(_FLOAT *sum, _FLOAT * c, _FLOAT v)
{
	_FLOAT y = v - *c;    //So far, so good: c is zero.
	_FLOAT t = *sum + y;         //Alas, sum is big, y small, so low-order digits of y are lost.
	*c = (t - *sum) - y;   //(t - sum) recovers the high-order part of y; subtracting y recovers -(low part of y)
	*sum = t;             //Algebraically, c should always be zero. Beware eagerly optimising compilers!
}

static inline void  Kahan_summation_tricked(_FLOAT *sum, _FLOAT * c, _FLOAT v, _FLOAT mod)
{
	_FLOAT y = v - *c;    //So far, so good: c is zero.
	_FLOAT t = *sum + y;         //Alas, sum is big, y small, so low-order digits of y are lost.
	*c = (t - *sum) * mod - y;   //(t - sum) recovers the high-order part of y; subtracting y recovers -(low part of y)
	*sum = t;             //Algebraically, c should always be zero. Beware eagerly optimising compilers!
}


static inline void Kahan_summation2(_FLOAT *sum, _FLOAT *c, _FLOAT *v, int n)
{
	for (int i = 0; i < n; ++i)
	{
		_FLOAT y = v[i] - c[i];    //So far, so good: c is zero.
		_FLOAT t = sum[i] + y;         //Alas, sum is big, y small, so low-order digits of y are lost.
		c[i] = (t - sum[i]) - y;   //(t - sum) recovers the high-order part of y; subtracting y recovers -(low part of y)
		sum[i] = t;             //Algebraically, c should always be zero. Beware eagerly optimising compilers!
	}
}

/*********************************************************************************************************
// wrw algorithm for large filters
// idea:
// move all filters for 3 ( or less) input maps into LDS
// genrate full output map moving vertically.
// 4 output maps per group
// generate MLO_OUT_PIX_TILE1 * MLO_N_OUT_FOLDS1 *MLO_N_FILTER_SPLITS0 at a time per wk-item
// MLO_N_FILTER_SPLITS0: keeps prev, current and next partial sums, each sums up 4 taps (last == 0). 


**********************************************************************************************************/

__attribute__((reqd_work_group_size(MLO_GRP_SZ0, MLO_GRP_SZ1, MLO_GRP_SZ2)))
__kernel void MLOpenCvFwd(
	const __global _FLOAT * bot,
	const __global _FLOAT * weights,
#if MLO_CONV_BIAS == 1
	const __global _FLOAT * bias,
#endif
	__global _FLOAT *top,
	_FLOAT padding_val
)
{

	__local _FLOAT lcl_mem[MLO_LCL_MEM_SZ];
	__local _FLOAT * bot_mem = lcl_mem;
	__local _FLOAT * wei_mem = lcl_mem + MLO_TOTAL_IN_LCL_SZ;

	int wave_id = getWaveId();
	int lcl_id = get_local_id(0);
	int lcl_wv_id = gePhysLocalId();




	int k_idx = get_group_id(1) * (MLO_OUT_STACKS * MLO_N_LCL_OUT_MAPS); // input map index based

	int c_idx = 0;

	int ib_idx = get_group_id(2)*MLO_N_LCL_BATCHS; // batch idx

	int ib = ib_idx;


	int gbl_in_off = /*c_idx * MLO_IN_CHANNEL_STRIDE + */ib * MLO_IN_BATCH_STRIDE;
	int gbl_wei_off = k_idx * MLO_WEI_BATCH_STRIDE;


#define MLO_ACCUM_SZ (MLO_OUT_PIX_TILE1 * MLO_N_FILTER_SPLITS0 * MLO_N_OUT_FOLDS1)

	__private _FLOAT pvt_accum[MLO_ACCUM_SZ];


	// zero out LDS
	for (int i = lcl_id; i < (MLO_LCL_MEM_SZ); i += MLO_GRP_SZ)
	{
		lcl_mem[i] = 0;
	}

// processing arrangement
// assume full width for now
// output stack
#define MLO_PROCESSING_WIDTH  (MLO_OUT_WIDTH + MLO_N_FILTER_SPLITS0 - 1)
	int k = iDiv(lcl_id, MLO_PROCESSING_WIDTH);
// my pixel position in 
	int k_pix = iMod(lcl_id, k, MLO_PROCESSING_WIDTH);


	// over all batches

	for (int b = 0;
		b < MLO_N_BATCH_LOOPS;
		b += MLO_N_LCL_BATCHS,
		gbl_in_off += MLO_N_LCL_BATCHS*MLO_IN_BATCH_STRIDE
		)
	{

		barrier(CLK_LOCAL_MEM_FENCE);

		// read all weights assuming they are fit into LDS
		for (int w = lcl_id; w < MLO_OUT_STACKS * MLO_N_LCL_OUT_MAPS*MLO_N_INPUTS*MLO_FILTER_SIZE1*MLO_FILTER_SIZE0; w += MLO_GRP_SZ)
		{
			int k = iDiv(w, MLO_N_INPUTS*MLO_FILTER_SIZE1*MLO_FILTER_SIZE0);
			int t0 = iMod(w, k, MLO_N_INPUTS*MLO_FILTER_SIZE1*MLO_FILTER_SIZE0);
			int c = iDiv(t0, MLO_FILTER_SIZE1*MLO_FILTER_SIZE0);
			int t1 = iMod(t0, c, MLO_FILTER_SIZE1*MLO_FILTER_SIZE0);
			int j = iDiv(t1, MLO_FILTER_SIZE0);
			int i = iMod(t1, j, MLO_FILTER_SIZE0);
			wei_mem[(k*MLO_N_INPUTS + c)* MLO_WEI_SZ + j*MLO_WEI_LCL_WIDTH + i] = weights[gbl_wei_off + j*MLO_FILTER_SIZE0 + i];

		}


		int out_y = 0;
		int in_y0 = 0;

		// prefetch MLO_FILTER_STRIDE1 - MLO_FILTER_PAD1 input scans
		__private _FLOAT in_rd_data[MLO_READ_UNIT];

		int gbl_in_scan_off0 = gbl_in_off;

		// generate pixels all MLO_OUT_STACKS * MLO_N_LCL_OUT_MAPS output maps
		for (int ob = 0; ob < MLO_N_OUT_BLKS; ++ob, in_y0 += (MLO_OUT_PIX_TILE1 *MLO_FILTER_STRIDE1*MLO_N_OUT_FOLDS1), gbl_in_scan_off0 += (MLO_OUT_PIX_TILE1 *MLO_FILTER_STRIDE1*MLO_N_OUT_FOLDS1) * MLO_IN_CHANNEL_STRIDE, out_y += MLO_OUT_PIX_TILE1 *MLO_N_OUT_FOLDS1)
		{


			for (int i = 0; i < MLO_ACCUM_SZ; ++i)
			{
				pvt_accum[i] = 0;
			}

			int n_prefetch_reads = (ob == 0) ? MLO_IN_LCL_HEIGHT - MLO_OUT_PIX_TILE1 *MLO_FILTER_STRIDE1 - MLO_FILTER_PAD1 : MLO_IN_LCL_HEIGHT - MLO_OUT_PIX_TILE1 *MLO_FILTER_STRIDE1;
			int prefetch_lcl_scan = (ob == 0) ? MLO_FILTER_PAD1 : 0;

			// all input maps
			for (int c = 0, gbl_in_scan_off = gbl_in_scan_off0, in_y = in_y0; c < MLO_N_INPUTS; ++c, ++in_y, gbl_in_scan_off += MLO_IN_CHANNEL_STRIDE)
			{

				barrier(CLK_LOCAL_MEM_FENCE);

				// prefetch 
				for (int p4 = lcl_id, c_scan = 0;  p4 < MLO_N_IN_HORIZ_READS * n_prefetch_reads && in_y + c_scan < MLO_IN_HEIGHT;
					p4 += MLO_GRP_SZ)
				{

					c_scan = iDiv(p4, MLO_N_IN_HORIZ_READS);
					int c_pix4 = iMod(p4, c_scan, MLO_N_IN_HORIZ_READS);

					// still problems with unaligned LDS access
#if MLO_IN_N_PIXS_OFF > 0
					if (c_pix4 == MLO_N_IN_HORIZ_READS - 1)
					{
						int i = 0;
						for (; i < MLO_IN_N_PIXS_OFF; ++i)
						{
							in_rd_data[i] = bot[gbl_in_scan_off + c_scan * MLO_IN_STRIDE + c_pix4*MLO_READ_UNIT + i];
						}
						for (; i < MLO_READ_UNIT; ++i)
						{
							in_rd_data[i] = 0;
						}

					}
					else
#endif
					{
						//					*(MLO_READ_TYPE*)in_rd_data = *(__global MLO_READ_TYPE*)&bot[gbl_in_scan_off + c_scan * MLO_IN_STRIDE + c_pix4*MLO_READ_UNIT];

						for (int i = 0; i < MLO_READ_UNIT; ++i)
						{
							in_rd_data[i] = bot[gbl_in_scan_off + c_scan * MLO_IN_STRIDE + c_pix4*MLO_READ_UNIT + i];
						}
					}
					for (int i = 0; i < MLO_READ_UNIT; ++i)
					{
						bot_mem[(c_scan + MLO_FILTER_PAD1)*MLO_IN_LCL_WIDTH + MLO_FILTER_PAD0 + c_pix4*MLO_READ_UNIT + i] = in_rd_data[i];
					}


				}


				// folds
				int lcl_scan = MLO_IN_LCL_HEIGHT - MLO_OUT_PIX_TILE1 *MLO_FILTER_STRIDE1;
				for (int of = 0; of < MLO_N_OUT_FOLDS1; ++of, in_y += (MLO_OUT_PIX_TILE1 *MLO_FILTER_STRIDE1), gbl_in_scan_off += ((MLO_OUT_PIX_TILE1 *MLO_FILTER_STRIDE1) * MLO_IN_STRIDE))
				{

					barrier(CLK_LOCAL_MEM_FENCE);

					// fetch next input
					for (int p4 = lcl_id, c_scan = 0; p4 < MLO_N_IN_HORIZ_READS * (MLO_OUT_PIX_TILE1 *MLO_FILTER_STRIDE1);
						p4 += MLO_GRP_SZ)
					{
						c_scan = iDiv(p4, MLO_N_IN_HORIZ_READS);

						int c_pix4 = iMod(p4, c_scan, MLO_N_IN_HORIZ_READS);

						for (int i = 0; i < MLO_READ_UNIT; ++i)
						{
							in_rd_data[i] = 0;
						}

						if (in_y + c_scan < MLO_IN_HEIGHT)
						{


#if MLO_IN_N_PIXS_OFF > 0
							if (c_pix4 == MLO_N_IN_HORIZ_READS - 1)
							{

								int i = 0;
								for (; i < MLO_IN_N_PIXS_OFF; ++i)
								{
									in_rd_data[i] = bot[gbl_in_scan_off + c_scan * MLO_IN_STRIDE + c_pix4*MLO_READ_UNIT + i];
								}
								for (; i < MLO_READ_UNIT; ++i)
								{
									in_rd_data[i] = 0;
								}

							}
							else
#endif
							{

								//								*(MLO_READ_TYPE*)in_rd_data = *(__global MLO_READ_TYPE*)&bot[gbl_in_scan_off + c_scan * MLO_IN_STRIDE + c_pix4*MLO_READ_UNIT];
								for (int i = 0; i < MLO_READ_UNIT; ++i)
								{
									in_rd_data[i] = bot[gbl_in_scan_off + c_scan * MLO_IN_STRIDE + c_pix4*MLO_READ_UNIT + i];
								}
							}

						} // if (in_y + c_scan < MLO_IN_HEIGHT)

						for (int i = 0; i < MLO_READ_UNIT; ++i)
						{
							bot_mem[(c_scan + lcl_scan)*MLO_IN_LCL_WIDTH + MLO_FILTER_PAD0 + c_pix4*MLO_READ_UNIT + i] = in_rd_data[i];
#if 0
							if (lcl_id == 0 && p4 == 0)
							{
								printf("K:g:%d %d %d %d %d %f\n",
									ob,
									MLO_IN_LCL_WIDTH,
									(MLO_IN_LCL_HEIGHT - MLO_FILTER_SIZE1 + MLO_FILTER_STRIDE1),
									MLO_FILTER_SIZE1 - MLO_FILTER_STRIDE1,
									(c_scan + MLO_FILTER_SIZE1 - MLO_FILTER_STRIDE1)*MLO_IN_LCL_WIDTH + MLO_FILTER_PAD0 + c_pix4*MLO_READ_UNIT + i,
									lcl_bot[(c_scan + MLO_FILTER_SIZE1 - MLO_FILTER_STRIDE1)*MLO_IN_LCL_WIDTH + MLO_FILTER_PAD0 + c_pix4*MLO_READ_UNIT + i]
								);
							}
#endif
						}


					}

					// convolution
					// along vertical filter
					for (int m = 0; m < MLO_FILTER_SIZE1; ++m)
					{
						// along vertical tile
						for (int j = 0; j < MLO_OUT_PIX_TILE1; ++j)
						{
							// select all vertical scans that matches the vertical filter tap 
							int in_j = j*MLO_FILTER_STRIDE1 + m;
							// k_pix is my current pixel position
							int in_x = k_pix * MLO_FILTER_STRIDE0;
							for (int t = 0; t < MLO_FILTER_STRIDE0; ++t)
							{
								_FLOAT val = bot_mem[in_j*MLO_IN_LCL_WIDTH + in_x + t];
								// 0 next, 1 current, 2 previous
								for (int s = 0; s < MLO_N_FILTER_SPLITS0; ++s)
								{
									_FLOAT wei = wei_mem[(k*MLO_N_INPUTS + c)* MLO_WEI_SZ + m*MLO_WEI_LCL_WIDTH + s*MLO_FILTER_STRIDE0 + t];
									pvt_accum[(of*MLO_OUT_PIX_TILE1 + j) * MLO_N_FILTER_SPLITS0 + s]
										+= val*wei;
								}

							}

						}
					}

					barrier(CLK_LOCAL_MEM_FENCE);

					// move input data to free space for the next fold
					// watch for barrier.
										// barrier can be skipped for sure if (MLO_IN_LCL_HEIGHT - MLO_OUT_PIX_TILE1 *MLO_FILTER_STRIDE1) < MLO_IN_LCL_HEIGHT / 2 && MLO_N_IN_HORIZ_READS <= MLO_WAVE_SZ

					for (int p4 = lcl_id, c_scan = 0;  p4 < MLO_N_IN_HORIZ_READS * (MLO_IN_LCL_HEIGHT - MLO_OUT_PIX_TILE1 *MLO_FILTER_STRIDE1);
						p4 += MLO_GRP_SZ)
					{

						c_scan = iDiv(p4, MLO_N_IN_HORIZ_READS);
						int c_pix4 = iMod(p4, c_scan, MLO_N_IN_HORIZ_READS);
						for (int i = 0; i < MLO_READ_UNIT; ++i)
						{
							bot_mem[c_scan*MLO_IN_LCL_WIDTH + MLO_FILTER_PAD0 + c_pix4*MLO_READ_UNIT + i]
								= bot_mem[(c_scan + MLO_OUT_PIX_TILE1 *MLO_FILTER_STRIDE1) *MLO_IN_LCL_WIDTH + MLO_FILTER_PAD0 + c_pix4*MLO_READ_UNIT + i];
						}

					}
				} // of
			} // c

// final summation
// 0 next, 1 current, 2 previous
			for (int of = 0; of < MLO_N_OUT_FOLDS1; ++of)
			{
				for (int j = 0; j < MLO_OUT_PIX_TILE1; ++j)
				{
					barrier(CLK_LOCAL_MEM_FENCE);

					for (int s = 0; s < MLO_N_FILTER_SPLITS0; ++s)
					{
						if (s <= k_pix &&  k_pix < (MLO_OUT_WIDTH + s))
						{
							lcl_mem[k*MLO_OUT_WIDTH + k_pix - s + s*MLO_OUT_WIDTH*MLO_OUT_STACKS]
								= pvt_accum[(of*MLO_OUT_PIX_TILE1 + j) * MLO_N_FILTER_SPLITS0 + s];
						}
					}

					barrier(CLK_LOCAL_MEM_FENCE);
					if (k_pix < MLO_OUT_WIDTH && (out_y + of*MLO_OUT_PIX_TILE1 + j) < MLO_OUT_HEIGHT)
					{
						_FLOAT final_sum = 0;
						for (int s = 0; s < MLO_N_FILTER_SPLITS0; ++s)
						{
							final_sum += k*MLO_OUT_WIDTH + k_pix + s*MLO_OUT_WIDTH*MLO_OUT_STACKS;
						}

						// write out 
						// inputs are outputs
						int out_off = (ib + b) * MLO_OUT_BATCH_STRIDE + (k_idx + k) * MLO_OUT_CHANNEL_STRIDE + (out_y + of*MLO_OUT_PIX_TILE1 + j) *MLO_OUT_STRIDE + k_pix;
						top[out_off] = final_sum;
					}
				}
			}

		} // ob

	}
}
