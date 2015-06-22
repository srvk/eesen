/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#ifndef CUPRINTF_H
#define CUPRINTF_H

/*
 *      This is the header file supporting cuPrintf.cu and defining both
 *      the host and device-side interfaces. See that file for some more
 *      explanation and sample use code. See also below for details of the
 *      host-side interfaces.
 *
 *  Quick sample code:
 *
        #include "cuPrintf.cu"
        
        __global__ void testKernel(int val)
        {
                cuPrintf("Value is: %d\n", val);
        }
        int main()
        {
                cudaPrintfInit();
                testKernel<<< 2, 3 >>>(10);
                cudaPrintfDisplay(stdout, true);
                cudaPrintfEnd();
        return 0;
        }
 */

///////////////////////////////////////////////////////////////////////////////
// DEVICE SIDE
// External function definitions for device-side code

// Abuse of templates to simulate varargs
__device__ int cuPrintf(const char *fmt);
template <typename T1> __device__ int cuPrintf(const char *fmt, T1 arg1);
template <typename T1, typename T2> __device__ int cuPrintf(const char *fmt, T1 arg1, T2 arg2);
template <typename T1, typename T2, typename T3> __device__ int cuPrintf(const char *fmt, T1 arg1, T2 arg2, T3 arg3);
template <typename T1, typename T2, typename T3, typename T4> __device__ int cuPrintf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4);
template <typename T1, typename T2, typename T3, typename T4, typename T5> __device__ int cuPrintf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5);
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6> __device__ int cuPrintf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6);
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7> __device__ int cuPrintf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7);
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8> __device__ int cuPrintf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8);
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9> __device__ int cuPrintf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9);
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename T10> __device__ int cuPrintf(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, T10 arg10);


//
//      cuPrintfRestrict
//
//      Called to restrict output to a given thread/block. Pass
//      the constant CUPRINTF_UNRESTRICTED to unrestrict output
//      for thread/block IDs. Note you can therefore allow
//      "all printfs from block 3" or "printfs from thread 2
//      on all blocks", or "printfs only from block 1, thread 5".
//
//      Arguments:
//              threadid - Thread ID to allow printfs from
//              blockid - Block ID to allow printfs from
//
//      NOTE: Restrictions last between invocations of
//      kernels unless cudaPrintfInit() is called again.
//
#define CUPRINTF_UNRESTRICTED   -1
__device__ void cuPrintfRestrict(int threadid, int blockid);



///////////////////////////////////////////////////////////////////////////////
// HOST SIDE
// External function definitions for host-side code

//
//      cudaPrintfInit
//
//      Call this once to initialise the printf system. If the output
//      file or buffer size needs to be changed, call cudaPrintfEnd()
//      before re-calling cudaPrintfInit().
//
//      The default size for the buffer is 1 megabyte. For CUDA
//      architecture 1.1 and above, the buffer is filled linearly and
//      is completely used;     however for architecture 1.0, the buffer
//      is divided into as many segments are there are threads, even
//      if some threads do not call cuPrintf().
//
//      Arguments:
//              bufferLen - Length, in bytes, of total space to reserve
//                          (in device global memory) for output.
//
//      Returns:
//              cudaSuccess if all is well.
//
extern "C" cudaError_t cudaPrintfInit(size_t bufferLen=1048576);   // 1-meg - that's enough for 4096 printfs by all threads put together

//
//      cudaPrintfEnd
//
//      Cleans up all memories allocated by cudaPrintfInit().
//      Call this at exit, or before calling cudaPrintfInit() again.
//
extern "C" void cudaPrintfEnd();

//
//      cudaPrintfDisplay
//
//      Dumps the contents of the output buffer to the specified
//      file pointer. If the output pointer is not specified,
//      the default "stdout" is used.
//
//      Arguments:
//              outputFP     - A file pointer to an output stream.
//              showThreadID - If "true", output strings are prefixed
//                             by "[blockid, threadid] " at output.
//
//      Returns:
//              cudaSuccess if all is well.
//
extern "C" cudaError_t cudaPrintfDisplay(void *outputFP=NULL, bool showThreadID=false);

#endif  // CUPRINTF_H
