/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Dynamic Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.  

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.
  

  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/

#include "utils.h"

__global__ void find_min_kernel(const float* const d_in, float* d_out)
{
    // sdata is allocated in the kernel call: 3rd arg to <<<b, t, shmem>>>
    extern __shared__ float sdata[];

    int myId = threadIdx.x + blockDim.x * blockIdx.x;
    int tid  = threadIdx.x;

    // load shared mem from global mem
    sdata[tid] = d_in[myId];
    __syncthreads();            // make sure entire block is loaded!

    // do reduction in shared mem
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] = min(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();        // make sure all adds at one stage are done!
    }

    // only thread 0 writes result for this block back to global mem
    if (tid == 0)
    {
        d_out[blockIdx.x] = sdata[0];
    }
}

__global__ void find_max_kernel(const float* const d_in, float* d_out)
{
    // sdata is allocated in the kernel call: 3rd arg to <<<b, t, shmem>>>
    extern __shared__ float sdata[];

    int myId = threadIdx.x + blockDim.x * blockIdx.x;
    int tid  = threadIdx.x;

    // load shared mem from global mem
    sdata[tid] = d_in[myId];
    __syncthreads();            // make sure entire block is loaded!

    // do reduction in shared mem
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] = max(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();        // make sure all adds at one stage are done!
    }

    // only thread 0 writes result for this block back to global mem
    if (tid == 0)
    {
        d_out[blockIdx.x] = sdata[0];
    }
}

void find_min_max(const size_t numPixels, const float* const d_logLuminance, float &min_logLum, float &max_logLum)
{
  const int maxThreadsPerBlock = 256;
  int threads = maxThreadsPerBlock;
  int blocks = numPixels / maxThreadsPerBlock;
  int threads2 = blocks;
  int blocks2 = 1;

  float* d_out = 0;
  float* d_min = 0;
  float* d_max = 0;
  checkCudaErrors(cudaMalloc(&d_out, blocks * sizeof(float)));
  checkCudaErrors(cudaMalloc(&d_min, sizeof(float)));
  checkCudaErrors(cudaMalloc(&d_max, sizeof(float)));


  find_min_kernel<<<blocks, threads, threads * sizeof(float)>>>(d_logLuminance, d_out);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  find_min_kernel<<<blocks2, threads2, threads2 * sizeof(float)>>>(d_out, d_min);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  find_max_kernel<<<blocks, threads, threads * sizeof(float)>>>(d_logLuminance, d_out);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  find_max_kernel<<<blocks2, threads2, threads2 * sizeof(float)>>>(d_out, d_max);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  checkCudaErrors(cudaMemcpy(&min_logLum, d_min, sizeof(float), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(&max_logLum, d_max, sizeof(float), cudaMemcpyDeviceToHost));

  checkCudaErrors(cudaFree(d_out));
  checkCudaErrors(cudaFree(d_min));
  checkCudaErrors(cudaFree(d_max));
}



__global__ void log_histogram_kernel(const size_t numPixels, 
                                     const float* const d_in, 
                                     unsigned int* d_bins, 
                                     const float min_logLum,
                                     const float range, 
                                     const size_t numBins)
{
  int myId = threadIdx.x + blockDim.x * blockIdx.x;
  if (myId >= numPixels)
  {
    return;
  }

  unsigned int myBin = min(numBins-1, size_t(numBins * (d_in[myId] - min_logLum) / range));
  atomicAdd(&(d_bins[myBin]), 1);
}

__global__ void hillis_steele_scan_inc_kernel(const size_t numBins, int step, unsigned int* d_bins_tmp1, unsigned int* const d_bins_tmp2)
{
  int myId = threadIdx.x + blockDim.x * blockIdx.x;
  if (myId + step >= numBins)
  {
    return;
  }
  d_bins_tmp2[myId + step] = d_bins_tmp1[myId] + d_bins_tmp1[myId + step];
}

void hillis_steele_scan_exc(const size_t numBins, unsigned int* d_bins, unsigned int* const d_cdf)
{
  const int maxThreadsPerBlock = 256;
  int threads = maxThreadsPerBlock;
  int blocks = numBins / maxThreadsPerBlock;

  unsigned int* d_bins_tmp1 = 0;
  unsigned int* d_bins_tmp2 = 0;
  checkCudaErrors(cudaMalloc(&d_bins_tmp1, numBins * sizeof(unsigned int)));
  checkCudaErrors(cudaMalloc(&d_bins_tmp2, numBins * sizeof(unsigned int)));
  checkCudaErrors(cudaMemcpy(d_bins_tmp1, d_bins, numBins * sizeof(unsigned int), cudaMemcpyDeviceToDevice));
  checkCudaErrors(cudaMemcpy(d_bins_tmp2, d_bins, numBins * sizeof(unsigned int), cudaMemcpyDeviceToDevice));


  int step = 1;
  while (step < numBins)
  {
    hillis_steele_scan_inc_kernel<<<blocks, threads>>>(numBins, step, d_bins_tmp1, d_bins_tmp2);
    checkCudaErrors(cudaMemcpy(d_bins_tmp1, d_bins_tmp2, numBins * sizeof(unsigned int), cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaGetLastError());
    step *= 2;
  }
  
  checkCudaErrors(cudaMemset(d_bins_tmp1, 0, numBins * sizeof(unsigned int)));
  checkCudaErrors(cudaMemcpy(d_bins_tmp1+1, d_bins_tmp2, (numBins-1) * sizeof(unsigned int), cudaMemcpyDeviceToDevice));
  checkCudaErrors(cudaMemcpy(d_cdf, d_bins_tmp1, numBins * sizeof(unsigned int), cudaMemcpyDeviceToDevice));
}


// void blelloch_scan_exc(const size_t numBins, unsigned int* d_bins, unsigned int* const d_cdf)
// {
//   const int maxThreadsPerBlock = 256;
//   int threads = maxThreadsPerBlock;
//   int blocks = numPixels / maxThreadsPerBlock;

//   unsigned int* d_bins_tmp1 = 0, d_bins_tmp2 = 0;
//   checkCudaErrors(cudaMalloc(&d_bins_tmp1, numBins * sizeof(unsigned int)));
//   checkCudaErrors(cudaMalloc(&d_bins_tmp2, numBins * sizeof(unsigned int)));
//   checkCudaErrors(cudaMemcpy(d_bins_tmp1, d_bins, numBins * sizeof(unsigned int), cudaDeviceToDevice));
//   checkCudaErrors(cudaMemcpy(d_bins_tmp2, d_bins, numBins * sizeof(unsigned int), cudaDeviceToDevice));
// }

void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
  //TODO
  /*Here are the steps you need to implement
    1) find the minimum and maximum value in the input logLuminance channel
       store in min_logLum and max_logLum
    2) subtract them to find the range
    3) generate a histogram of all the values in the logLuminance channel using
       the formula: bin = (lum[i] - lumMin) / lumRange * numBins
    4) Perform an exclusive scan (prefix sum) on the histogram to get
       the cumulative distribution of luminance values (this should go in the
       incoming d_cdf pointer which already has been allocated for you)       */

  size_t numPixels = numRows * numCols;
  find_min_max(numPixels, d_logLuminance, min_logLum, max_logLum);
  float range = max_logLum - min_logLum;

  const int maxThreadsPerBlock = 256;
  int threads = maxThreadsPerBlock;
  int blocks = numPixels / maxThreadsPerBlock;

  unsigned int* d_bins = 0;
  checkCudaErrors(cudaMalloc(&d_bins, numBins * sizeof(unsigned int)));
  checkCudaErrors(cudaMemset(d_bins, 0, numBins * sizeof(unsigned int)));

  log_histogram_kernel<<<blocks, threads>>>(numPixels, d_logLuminance, d_bins, min_logLum, range, numBins);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());


  hillis_steele_scan_exc(numBins, d_bins, d_cdf);
//  checkCudaErrors(cudaMemcpy(d_cdf, d_bins, numBins * sizeof(unsigned int), cudaMemcpyDeviceToDevice));
  checkCudaErrors(cudaFree(d_bins));
}
