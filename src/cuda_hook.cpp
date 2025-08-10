/*
 * Modified on Sat May 18 2024
 *
 * Author: KontonGu
 * Copyright (c) 2024 TUM - CAPS Cloud
 * Licensed under the Apache License, Version 2.0 (the "License")
 */

#include "cuda_hook.h"

#include <stdio.h>
#include <dlfcn.h>
#include <signal.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <cstring>

#include <fstream>
#include <atomic>
#include <algorithm>

void* libcuda_driver_handle = dlopen("libcuda.so", RTLD_LAZY);

// get original dlsym pointer with different libc6 versions under differnt ubuntu system
// More information of libc6: https://pkgs.org/download/libc6
oriDlsymFn* get_dlsym_ptr(){
    const char* libc_version = gnu_get_libc_version();
    LOG_INFO("The libc Version in the system is:{}", libc_version);
    if(strcmp(libc_version,"2.31") == 0){
        return (oriDlsymFn*) __libc_dlsym(libdl_handle, "dlsym");  // ubuntu 20.04
    }else if(strcmp(libc_version, "2.35") == 0 || strcmp(libc_version, "2.39") == 0){
        return (oriDlsymFn*) dlvsym(libc_handle, "dlsym", "GLIBC_2.34");  // ubuntu 22.04 (2.35), ubuntu 24.04 (2.39)
    }else{ 
        return (oriDlsymFn*) __libc_dlsym(libdl_handle, "dlsym");  // lower than ubuntu 20.04
    }
}

// interception management 
Interception cuda_intercept;

// signal 
std::atomic<bool> running(true);

// Intialization for interception
void intercept_initialize();
// Initialize once
std::once_flag init_once;

// Containe processing functions before and after the hooked Driver API
struct hookInfo {
    int debug_mode = 0;
    void *hook_pre[NUM_HOOK_SYMBOLS];
    void *hook_post[NUM_HOOK_SYMBOLS];
};
static struct hookInfo hook_info;


// Return the intercepted function's pointer when called from cuGetProcAddress_v2
#define TRY_INTERCEPT(symbol, intercept_func, text)                                      \
    if(stringCompare(symbol, text)){                                                     \
        *pfn = (void *) (&intercept_func);                                               \
        return CUDA_SUCCESS;                                                             \
    }                                                                                    

// The Handling process of an interception
#define CU_HOOK_MAKE_CUDA_INTERCEPT(symbol, hook_index, params, ...)                                         \
CUresult CUDAAPI symbol params{                                                                              \
    std::call_once(init_once, intercept_initialize);                                                         \
    LOG_DEBUG("cudaAPI = {}, is called.", SYMBOL_TO_STRING(symbol));                                         \
    using symbol##handler = CUresult CUDAAPI (params);                                                       \
    auto real_func = (symbol##handler *) dlsym(libcuda_driver_handle, SYMBOL_TO_STRING(symbol));             \
    if(real_func == nullptr){                                                                                \
        LOG_ERROR("Interception method is not found: {}, error: {}", SYMBOL_TO_STRING(symbol), dlerror());   \
        return CUDA_ERROR_UNKNOWN;                                                                           \
    }                                                                                                        \
    CUresult res = CUDA_SUCCESS;                                                                             \
    if (hook_info.hook_pre[hook_index]){                                                                     \
        res = ((CUresult CUDAAPI (*) params) hook_info.hook_pre[hook_index])(__VA_ARGS__);                   \
        if(res != CUDA_SUCCESS){                                                                             \
            return res;                                                                                      \
        }                                                                                                    \
    }                                                                                                        \
    res = real_func(__VA_ARGS__);                                                                            \
    if(res != CUDA_SUCCESS) {                                                                                \
        LOG_DEBUG("Orignial/real CUDA driver api call failed {}.", SYMBOL_TO_STRING(symbol));                \
    }                                                                                                        \
    if(hook_info.hook_post[hook_index]){                                                                     \
        res = ((CUresult CUDAAPI (*) params) hook_info.hook_post[hook_index])(__VA_ARGS__);                  \
    }                                                                                                        \
    return res;                                                                                              \
}      

// LOG_INFO("Intercepted func: {}, result: {}", SYMBOL_TO_STRING(symbol), res);                             

// // memory allocation and free APIs
CU_HOOK_MAKE_CUDA_INTERCEPT(cuMemAlloc, CU_HOOK_MEM_ALLOC, 
                            (CUdeviceptr * dptr, size_t bytesize), 
                            dptr, bytesize)
CU_HOOK_MAKE_CUDA_INTERCEPT(cuMemAllocManaged, CU_HOOK_MEM_ALLOC_MANAGED,  
                            (CUdeviceptr * dptr, size_t bytesize, unsigned int flags), 
                            dptr, bytesize, flags)
CU_HOOK_MAKE_CUDA_INTERCEPT(cuMemAllocPitch, CU_HOOK_MEM_ALLOC_PITCH, 
                            (CUdeviceptr* dptr, size_t* pPitch, size_t WidthInBytes, size_t Height, unsigned int  ElementSizeBytes), 
                            dptr, pPitch, WidthInBytes, Height, ElementSizeBytes)
CU_HOOK_MAKE_CUDA_INTERCEPT(cuMemAllocAsync, CU_HOOK_MEM_ALLOC_ASYNC,  
                            (CUdeviceptr* dptr, size_t bytesize, CUstream hStream), 
                            dptr, bytesize, hStream)
CU_HOOK_MAKE_CUDA_INTERCEPT(cuMemFree, CU_HOOK_MEM_FREE, 
                            (CUdeviceptr dptr), 
                            dptr)
CU_HOOK_MAKE_CUDA_INTERCEPT(cuArrayCreate, CU_HOOK_ARRAY_CREATE, 
                            (CUarray* pHandle, const CUDA_ARRAY_DESCRIPTOR* pAllocateArray),
                            pHandle, pAllocateArray)
CU_HOOK_MAKE_CUDA_INTERCEPT(cuArray3DCreate, CU_HOOK_ARRAY3D_CREATE, 
                            (CUarray* pHandle, const CUDA_ARRAY3D_DESCRIPTOR* pAllocateArray),
                            pHandle, pAllocateArray)
CU_HOOK_MAKE_CUDA_INTERCEPT(cuMipmappedArrayCreate, CU_HOOK_MIPMAPPED_ARRAY_CREATE,
                            (CUmipmappedArray* pHandle, const CUDA_ARRAY3D_DESCRIPTOR* pMipmappedArrayDesc, unsigned int  numMipmapLevels),
                            pHandle, pMipmappedArrayDesc, numMipmapLevels)
CU_HOOK_MAKE_CUDA_INTERCEPT(cuArrayDestroy, CU_HOOK_ARRAY_DESTROY,
                            (CUarray hArray),
                            hArray)
CU_HOOK_MAKE_CUDA_INTERCEPT(cuMipmappedArrayDestroy, CU_HOOK_MIPMAPPED_ARRAY_DESTROY,
                             (CUmipmappedArray hMipmappedArray),
                             hMipmappedArray)


// memory copy API
CU_HOOK_MAKE_CUDA_INTERCEPT(cuMemcpyAtoH, CU_HOOK_MEMCPY_ATOH, 
                            (void *dstHost, CUarray srcArray, size_t srcOffset, size_t ByteCount), 
                            dstHost, srcArray, srcOffset, ByteCount)
CU_HOOK_MAKE_CUDA_INTERCEPT(cuMemcpyDtoH, CU_HOOK_MEMCPY_DTOH,  
                            (void *dstHost, CUdeviceptr srcDevice, size_t ByteCount), 
                           dstHost, srcDevice, ByteCount)
CU_HOOK_MAKE_CUDA_INTERCEPT(cuMemcpyDtoHAsync, CU_HOOK_MEMCPY_DTOH_ASYNC,
                            (void* dstHost, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream),
                            dstHost, srcDevice, ByteCount, hStream)
CU_HOOK_MAKE_CUDA_INTERCEPT(cuMemcpyHtoA, CU_HOOK_MEMCPY_HTOA, 
                            (CUarray dstArray, size_t dstOffset, const void *srcHost,size_t ByteCount), 
                            dstArray, dstOffset, srcHost, ByteCount)

CU_HOOK_MAKE_CUDA_INTERCEPT(cuMemcpyHtoD, CU_HOOK_MEMCPY_HTOD,
                            (CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount),
                            dstDevice, srcHost, ByteCount)
CU_HOOK_MAKE_CUDA_INTERCEPT(cuMemcpyHtoDAsync, CU_HOOK_MEMCPY_HTOD_ASYNC,
                            (CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount, CUstream hStream),
                            dstDevice, srcHost, ByteCount, hStream)



//kernel launch API
CU_HOOK_MAKE_CUDA_INTERCEPT(cuLaunchKernel, CU_HOOK_LAUNCH_KERNEL, (CUfunction f, unsigned int gridDimX, unsigned int gridDimY,
                            unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY,
                            unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream,
                            void **kernelParams, void **extra),
                            f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ,
                            sharedMemBytes, hStream, kernelParams, extra)
CU_HOOK_MAKE_CUDA_INTERCEPT(cuLaunchCooperativeKernel, CU_HOOK_LAUNCH_COOPERATIVE_KERNEL, 
                            (CUfunction f, unsigned int  gridDimX, unsigned int  gridDimY, unsigned int  gridDimZ, 
                            unsigned int  blockDimX, unsigned int  blockDimY, unsigned int  blockDimZ, 
                            unsigned int  sharedMemBytes, CUstream hStream, void** kernelParams), f, gridDimX, gridDimY, 
                            gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams)


// // context synchronization API
CU_HOOK_MAKE_CUDA_INTERCEPT(cuCtxSynchronize, CU_HOOK_CTX_SYNC,
                            (void))

CU_HOOK_MAKE_CUDA_INTERCEPT(cuCtxCreate, CU_HOOK_CTX_CREATE, 
                            (CUcontext* pctx, unsigned int  flags, CUdevice dev),
                            pctx, flags, dev);


// CUresult cuCtxCreate ( CUcontext* pctx, unsigned int  flags, CUdevice dev )


// CU_HOOK_MAKE_CUDA_INTERCEPT(cuGetProcAddress, CU_HOOK_GET_PROC_ADDRESS, 
//                             (const char *symbol, void **pfn, int driverVersion, cuuint64_t flags, CUdriverProcAddressQueryResult *symbolStatus), 
//                             symbol, pfn, driverVersion, flags, symbolStatus)



CUresult getProcAddressBySymbol(const char *symbol, void **pfn, int driverVersion, cuuint64_t flags,
                                    CUdriverProcAddressQueryResult *symbolStatus){
    
    if(libcuda_driver_handle == nullptr){                                                                
        LOG_ERROR("Interception error (libcuda_driver_handle is null): {}", dlerror());                                                                                    
    }
    if(symbol == nullptr){
        LOG_ERROR("Symbol is empty.");
        return CUDA_ERROR_UNKNOWN;
    }
    // LOG_INFO("GetMethodBySymbol is called:  symbol_name = {}, cudaVersion = {}, flags = {}.", symbol, driverVersion, flags);

    // memory copy api
    TRY_INTERCEPT(symbol, cuMemcpyAtoH, "cuMemcpyAtoH");
    TRY_INTERCEPT(symbol, cuMemcpyDtoH, "cuMemcpyDtoH");
    TRY_INTERCEPT(symbol, cuMemcpyHtoA, "cuMemcpyHtoA");
    TRY_INTERCEPT(symbol, cuMemcpyHtoD, "cuMemcpyHtoD");
    TRY_INTERCEPT(symbol, cuMemcpyHtoDAsync, "cuMemcpyHtoDAsync");
    TRY_INTERCEPT(symbol, cuMemcpyDtoHAsync, "cuMemcpyDtoHAsync");
    
    
    // // memory allocation/free api
    TRY_INTERCEPT(symbol, cuMemAlloc, "cuMemAlloc");
    // TRY_INTERCEPT(symbol, cuMemAllocManaged, "cuMemAllocManaged");
    // TRY_INTERCEPT(symbol, cuMemAllocPitch, "cuMemAllocPitch");
    // TRY_INTERCEPT(symbol, cuMemAllocAsync, "cuMemAllocAsync");
    TRY_INTERCEPT(symbol, cuMemFree, "cuMemFree");
    // TRY_INTERCEPT(symbol, cuArrayCreate, "cuArrayCreate");
    // TRY_INTERCEPT(symbol, cuArray3DCreate, "cuArray3DCreate");
    // TRY_INTERCEPT(symbol, cuArrayDestroy, "cuArrayDestroy");
    // TRY_INTERCEPT(symbol, cuMipmappedArrayCreate, "cuMipmappedArrayCreate");
    // TRY_INTERCEPT(symbol, cuMipmappedArrayDestroy, "cuMipmappedArrayDestroy");

    // kernel api intercept
    TRY_INTERCEPT(symbol, cuLaunchKernel, "cuLaunchKernel");
    TRY_INTERCEPT(symbol, cuLaunchCooperativeKernel, "cuLaunchCooperativeKernel");

    // context syntax intercept
    TRY_INTERCEPT(symbol, cuCtxSynchronize, "cuCtxSynchronize");
    TRY_INTERCEPT(symbol, cuCtxCreate, "cuCtxCreate");

     if(strcmp(symbol, "cuGetProcAddress") == 0){
        *pfn = (void *) &getProcAddressBySymbol;
        return CUDA_SUCCESS;
    }

    CUresult res_tmp = cuGetProcAddress_v2(symbol, pfn, driverVersion, flags, symbolStatus);
    if(res_tmp == CUDA_SUCCESS){
        // LOG_DEBUG("cuda api {} is found, pfn= {}, symbolStatus = {}", symbol, *pfn, symbolStatus);
    }else{
        LOG_DEBUG("cuda api {} is not found.");
    }
    return res_tmp;

    /*  manual check */ 
    // // 1. try to find "symbol" + _v3
    // char symbolExtV3[strlen(symbol) + 3] = {};
    // strcat(symbolExtV3, symbol);
    // strcat(symbolExtV3, "_v3");
    // *pfn = dlsym(libcuda_driver_handle, symbolExtV3);
    // if (*pfn != nullptr)
    // {
    //     return CUDA_SUCCESS;
    // }

    // // 2. try to find "symbol" + _v2
    // char symbolExtV2[strlen(symbol) + 3] = {};
    // strcat(symbolExtV2, symbol);
    // strcat(symbolExtV2, "_v2");
    // *pfn = dlsym(libcuda_driver_handle, symbolExtV2);
    // if (*pfn != nullptr)
    // {
    //     return CUDA_SUCCESS;
    // }

    // *pfn = dlsym(libcuda_driver_handle, symbol);
    // if (*pfn != nullptr)
    // {
    //     return CUDA_SUCCESS;
    // }

    // if(strcmp(symbol, "")){
    //     LOG_ERROR("symbol not found: {}", symbol);
    // } 
    // return CUDA_ERROR_UNKNOWN;
}


CUresult cuMemAlloc_prehook(CUdeviceptr *dptr, size_t bytesize){
    // size_t remain, limit;
    LOG_DEBUG("cuMemAlloc_prehook Entered.");
    // std::tie(remain, limit) = cuda_intercept.get_gpu_mem_info();
    LOG_DEBUG("Requested memory size: {}", bytesize);
    std::unique_lock<std::mutex> m_lock(cuda_intercept.mem_mtx);
    if(bytesize > cuda_intercept.gpu_mem_remain){
        LOG_ERROR("Exceeds memory limit, out of memory.");
        cuda_intercept.grace_exit();
        m_lock.unlock();
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    m_lock.unlock();
    return CUDA_SUCCESS;
}


CUresult cuMemAlloc_posthook(CUdeviceptr * dptr, size_t bytesize){
    // Allocate first, Check later
    std::unique_lock<std::mutex> m_lock(cuda_intercept.mem_mtx);
    cuda_intercept.gpu_mem_usage += bytesize;
    cuda_intercept.gpu_mem_remain -= bytesize;
    cuda_intercept.mem_addr_size_record[*dptr] = bytesize;
    LOG_DEBUG("Mem allocated address: CUdeviceptr = {}", *dptr);
    cuda_intercept.mem_update_queue.push(std::make_pair(1, bytesize));
    cuda_intercept.mem_update_cond.notify_one();
    m_lock.unlock();
    return CUDA_SUCCESS;
}

CUresult cuMemAllocManaged_prehook(CUdeviceptr *dptr, size_t bytesize, unsigned int flags){
    //Also consider the unified memory
    CUresult res = cuMemAlloc_prehook(dptr, bytesize);
    return res;
}


CUresult cuMemAllocManaged_posthook(CUdeviceptr *dptr, size_t bytesize, unsigned int flags){
    cuMemAlloc_posthook(dptr, bytesize);
    int device_num;
    cudaError_t err = cudaGetDevice(&device_num);
    if(err != cudaSuccess){
        LOG_ERROR("cudaGetDevice error: {}", cudaGetErrorString(err));
        // exit(EXIT_FAILURE);
        cuda_intercept.grace_exit();
    }
    CUdevice cur_device;
    CUresult res = cuDeviceGet(&cur_device, device_num);
    if(res == CUDA_SUCCESS){
        cuda_intercept.devptrs_mngr.push_back(std::make_tuple(*dptr, bytesize, cur_device));
    }else{
        LOG_DEBUG("cuDeviceGet False.");
    }
    
    return res;
}

CUresult cuMemAllocPitch_prehook(CUdeviceptr *dptr, size_t *pPitch, size_t WidthInBytes,
                                 size_t Height, unsigned int ElementSizeBytes) {
  return cuMemAlloc_prehook(dptr, (*pPitch) * Height);
}
CUresult cuMemAllocPitch_posthook(CUdeviceptr *dptr, size_t *pPitch, size_t WidthInBytes,
                                  size_t Height, unsigned int ElementSizeBytes) {
  return cuMemAlloc_posthook(dptr, (*pPitch) * Height);
}

CUresult cuMemFree_prehook(CUdeviceptr ptr){
    std::unique_lock<std::mutex> m_lock(cuda_intercept.mem_mtx);
    if(cuda_intercept.mem_addr_size_record.find(ptr) != cuda_intercept.mem_addr_size_record.end()){
        size_t mem_size = cuda_intercept.mem_addr_size_record[ptr];
        cuda_intercept.gpu_mem_usage -= mem_size;
        cuda_intercept.mem_update_queue.push(std::make_pair(0, mem_size));
        cuda_intercept.mem_update_cond.notify_one();
        cuda_intercept.gpu_mem_remain += mem_size;
        // int is_succeed = cuda_intercept.update_gpu_mem_usage(mem_size, 0);
        cuda_intercept.mem_addr_size_record.erase(ptr);
        auto new_end = std::remove_if(cuda_intercept.devptrs_mngr.begin(), cuda_intercept.devptrs_mngr.end(),
        [ptr](std::tuple<CUdeviceptr, size_t, CUdevice>& x){ return (std::get<0>(x) == ptr);});
        if(new_end != cuda_intercept.devptrs_mngr.end()){
            cuda_intercept.devptrs_mngr.erase(new_end);
        }
        LOG_DEBUG("memory freed: {}", mem_size);
    }else{
        LOG_ERROR("Cannot find the memory address, cuMemFree failed. CUdeviceptr = {}", ptr);
    }
    m_lock.unlock();
    return CUDA_SUCCESS;
}


CUresult cuArrayDestroy_prehook(CUarray hArray) { 
    return cuMemFree_prehook((CUdeviceptr)hArray); 
}

CUresult cuMipmappedArrayDestroy_prehook(CUmipmappedArray hMipmappedArray) {
  return cuMemFree_prehook((CUdeviceptr)hMipmappedArray);
}

CUresult cuArrayCreate_prehook(CUarray *pHandle, const CUDA_ARRAY_DESCRIPTOR *pAllocateArray) {
  size_t totalMemoryNumber =
      pAllocateArray->Width * pAllocateArray->Height * pAllocateArray->NumChannels;
  size_t formatSize = CUarray_format_to_byte(pAllocateArray->Format);
  return cuMemAlloc_prehook((CUdeviceptr *)pHandle, totalMemoryNumber * formatSize);
}

CUresult cuArrayCreate_posthook(CUarray *pHandle, const CUDA_ARRAY_DESCRIPTOR *pAllocateArray) {
  size_t totalMemoryNumber =
      pAllocateArray->Width * pAllocateArray->Height * pAllocateArray->NumChannels;
  size_t formatSize = CUarray_format_to_byte(pAllocateArray->Format);
  return cuMemAlloc_posthook((CUdeviceptr *)pHandle, totalMemoryNumber * formatSize);
}

CUresult cuArray3DCreate_prehook(CUarray *pHandle, const CUDA_ARRAY3D_DESCRIPTOR *pAllocateArray) {
    size_t totalMemoryNumber = pAllocateArray->Width * pAllocateArray->Height *
                             pAllocateArray->Depth * pAllocateArray->NumChannels;
  size_t formatSize = CUarray_format_to_byte(pAllocateArray->Format);
  return cuMemAlloc_prehook((CUdeviceptr *)pHandle, totalMemoryNumber * formatSize);
}

CUresult cuArray3DCreate_posthook(CUarray *pHandle, const CUDA_ARRAY3D_DESCRIPTOR *pAllocateArray) {
  size_t totalMemoryNumber = pAllocateArray->Width * pAllocateArray->Height *
                             pAllocateArray->Depth * pAllocateArray->NumChannels;
  size_t formatSize = CUarray_format_to_byte(pAllocateArray->Format);
  return cuMemAlloc_posthook((CUdeviceptr *)pHandle, totalMemoryNumber * formatSize);
}


void Interception::cuda_execution_monitor(){
    LOG_INFO("cuda_execution_monitor thread is running.");
    cuCtxSetCurrent(cur_ctx);
    while(running.load()){
        std::unique_lock<std::mutex> m_lock(monitor_mtx);
        monitor_ready = true;
        monitor_ready_cond.notify_all();
        monitor_start_cond.wait(m_lock);
        LOG_DEBUG("[Monitoring] running for a period.");
        m_lock.unlock();
        int token_ms = std::max(token_time, 0.0);
        m_lock.lock();
        auto status = monitor_wait_cond.wait_for(m_lock, std::chrono::milliseconds(token_ms));
        m_lock.unlock();
        LOG_DEBUG("[Monitoring] monitor finish waiting.");
        if(status != std::cv_status::timeout){
            // LOG_DEBUG("[Monitoring] The monitoring is interrupted before the token expiration");
            LOG_DEBUG("[Monitoring] The monitoring is interrupted before the token expiration");
        }
        cuEventRecord(event_stop, 0);
        cuEventSynchronize(event_stop);
        float elapsed_time;
        cuEventElapsedTime(&elapsed_time, event_start, event_stop);
        LOG_DEBUG("[Monitoring] Period elapsed time: elapsed = {}ms", elapsed_time);
        overused_time = std::max(0.0, (float) elapsed_time - token_time);
        m_lock.lock();
        monitor_finished = true;
        monitor_ready = false;
        LOG_DEBUG("[Monitoring] Monitor finished calculating.");
        monitor_finished_cond.notify_all();
        m_lock.unlock();
    }
}

CUresult cuLaunchKernel_prehook(CUfunction f, unsigned int gridDimX, unsigned int gridDimY,
                            unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY,
                            unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream,
                            void **kernelParams, void **extra){
    LOG_DEBUG("Entered cuLanuchKernel_prehook");
    std::unique_lock<std::mutex> k_lock(cuda_intercept.kernel_mtx);
    // //  std::this_thread::sleep_for(std::chrono::milliseconds(40));
    if(cuda_intercept.ms_since_token_start() + 1.0 > cuda_intercept.token_time){
        // LOG_DEBUG("[launchkernel], to check if the monitor finished.");
        // std::unique_lock<std::mutex> m_lock(cuda_intercept.monitor_mtx);
        // if(!cuda_intercept.monitor_finished){
        //     LOG_DEBUG("[launchkernel], monitor does not finished, wait monitor to finish.");
        //     cuda_intercept.monitor_wait_cond.notify_all();
        //     cuda_intercept.monitor_finished_cond.wait(m_lock);
        //     LOG_DEBUG("[launchkernel], monitor finished elapsed time calculation.");
        // }
        // m_lock.unlock();
        
        cuda_intercept.token_time = cuda_intercept.get_token_from_gpu_client(0.0);
        //KONTON_TEST_START
        // cuda_intercept.token_time = 20.0;
        // usleep(20000);
        //KONTON_TEST_END
        // std::unique_lock<std::mutex> m_lock(cuda_intercept.monitor_mtx);
        // auto status = cuda_intercept.monitor_wait_cond.wait_for(m_lock, std::chrono::milliseconds(20));
        // m_lock.unlock();

        LOG_DEBUG("cuLaunchKernel_prehook, get new token from gpu client: {}", cuda_intercept.token_time);

        cuda_intercept.token_start = steady_clock::now();
        // cuEventRecord(cuda_intercept.event_start, 0);
        
        // m_lock.lock();
        // if(!cuda_intercept.monitor_ready){
        //     cuda_intercept.monitor_ready_cond.wait(m_lock);
        // }
        // cuda_intercept.monitor_start_cond.notify_all();
        // m_lock.unlock();
    }
    k_lock.unlock();
    LOG_DEBUG("[launchkernel], kernerl passed.");
    return CUDA_SUCCESS;
}





// Initialization when performing the first interception 
void intercept_initialize(){
    LOG_INFO("Start Initilization");
    if(libcuda_driver_handle == nullptr){                                                                
        LOG_ERROR("Interception error (libcuda_driver_handle is null): {}", dlerror());                                                                                    
    }
    // memory pre and post hook processing functions
    hook_info.hook_pre[CU_HOOK_MEM_ALLOC] = (void *) cuMemAlloc_prehook;
    hook_info.hook_post[CU_HOOK_MEM_ALLOC] = (void *) cuMemAlloc_posthook;
    // hook_info.hook_pre[CU_HOOK_MEM_ALLOC_MANAGED] = (void *) cuMemAllocManaged_prehook;
    // hook_info.hook_post[CU_HOOK_MEM_ALLOC_MANAGED] = (void *) cuMemAllocManaged_posthook;
    // hook_info.hook_pre[CU_HOOK_MEM_ALLOC_PITCH] = (void *) cuMemAllocPitch_prehook;
    // hook_info.hook_post[CU_HOOK_MEM_ALLOC_PITCH] = (void *) cuMemAllocPitch_posthook;
    // hook_info.hook_pre[CU_HOOK_ARRAY_DESTROY] = (void *) cuArrayDestroy_prehook;
    // hook_info.hook_pre[CU_HOOK_MIPMAPPED_ARRAY_DESTROY] = (void *) cuMipmappedArrayDestroy_prehook;
    // hook_info.hook_pre[CU_HOOK_ARRAY_CREATE] = (void *)cuArrayCreate_prehook;
    // hook_info.hook_post[CU_HOOK_ARRAY_CREATE] = (void *)cuArrayCreate_posthook;
    // hook_info.hook_pre[CU_HOOK_ARRAY3D_CREATE] = (void *)cuArray3DCreate_prehook;
    // hook_info.hook_post[CU_HOOK_ARRAY3D_CREATE] = (void *)cuArray3DCreate_posthook;
    hook_info.hook_pre[CU_HOOK_MEM_FREE] = (void *) cuMemFree_prehook;

    hook_info.hook_pre[CU_HOOK_LAUNCH_KERNEL] = (void *) cuLaunchKernel_prehook;
    
    cuda_intercept.config_connection();
    cuda_intercept.gc_sockfd = cuda_intercept.establish_connection();
    LOG_DEBUG("Connection established.");
    cuEventCreate(&cuda_intercept.event_start, CU_EVENT_DEFAULT);
    cuEventCreate(&cuda_intercept.event_stop, CU_EVENT_DEFAULT);
    cuCtxGetCurrent(&(cuda_intercept.cur_ctx));
    
    std::unique_lock<std::mutex> m_lock(cuda_intercept.mem_mtx);
    std::tie(cuda_intercept.gpu_mem_remain, cuda_intercept.gpu_mem_limit) = cuda_intercept.get_gpu_mem_info();
    m_lock.unlock();

    std::thread monitor_thread(&Interception::cuda_execution_monitor, &cuda_intercept);
    monitor_thread.detach();
    cuda_intercept.tids.push_back(monitor_thread.native_handle());

    std::thread mem_update_thread(&Interception::update_gpu_mem, &cuda_intercept);
    mem_update_thread.detach();
    cuda_intercept.tids.push_back(mem_update_thread.native_handle());

    cuda_intercept.token_time = cuda_intercept.get_token_from_gpu_client(0.0);
    cuda_intercept.monitor_finished = true;
    cuda_intercept.monitor_ready = true;

    signal(SIGINT, sigintHandler);  
    signal(SIGCONT, sigcontHandler);  
}




void Interception::config_connection(){
    std::ifstream ifs_ip(gpu_client_ip_file, std::ios::in);
    if(!ifs_ip.is_open()){
        LOG_ERROR("Failed to open the ip file");
        exit(-1);
    }
    std::string line;
    std::getline(ifs_ip, line);
    client_gpu_ip = line;
    ifs_ip.close();

    char *port = getenv("GPU_CLIENT_PORT");
    if(port != NULL) client_gpu_port = atoi(port);

    LOG_INFO("The IP and Port to the client GPU is: {}:{}", client_gpu_ip, client_gpu_port);
}


int Interception::establish_connection(){
    int sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if(sockfd < 0){
        LOG_ERROR("Error creating socket!");
        exit(-1);
    }
    struct sockaddr_in gc_addr;
    int addr_len = sizeof(gc_addr);
    std::memset(&gc_addr, 0, sizeof(gc_addr));
    gc_addr.sin_family = AF_INET;
    gc_addr.sin_port = htons(client_gpu_port);
    gc_addr.sin_addr.s_addr = inet_addr(client_gpu_ip.c_str());
    int res = multiple_try(
        [&]() -> int {return connect(sockfd, (sockaddr *)&gc_addr, (socklen_t) addr_len); },
        SOCKET_MAX_TRY, SOCKET_RETRY_INTV
    );

    if(res == -1){
        LOG_ERROR("Error connecting client gpu server!, error = {}", strerror(errno));
        close(sockfd);
        exit(-1);
    }
    return sockfd;
}


int Interception::socket_communicate(MsgBuffer &sbuf, MsgBuffer &rbuf, int sock_timeout){
    // gc_sockfd = establish_connection();
    timeval tv;
    tv = {sock_timeout, 0};
    setsockopt(gc_sockfd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    std::unique_lock<std::mutex> m_lock(comm_mtx);
    char rec_buf[MSGBUFFER_MAX_LEN];
    int res =multiple_try(
        [&]() -> int {
            if(send(gc_sockfd, sbuf.getData(), MSGBUFFER_MAX_LEN, 0) <= 0){
                LOG_DEBUG("socket message sending error: {}", strerror(errno));
                return -1;
            }
            if(recv(gc_sockfd, rec_buf, MSGBUFFER_MAX_LEN, 0) <= 0){
                LOG_DEBUG("socket message receving error: {}", strerror(errno));
                return -1;
            }
            return 0;
        }, SOCKET_MAX_TRY, 1);
    rbuf.copy2MsgBuffer(rec_buf, MSGBUFFER_MAX_LEN);
    m_lock.unlock();
    return res;
}


std::pair<size_t, size_t> Interception::get_gpu_mem_info(){
    MsgBuffer sbuf;
    MsgBuffer rbuf;
    ReqID req_id = new_request(sbuf, MEM_LIMIT);
    LOG_DEBUG("new request: {}", req_id);
    int res = socket_communicate(sbuf, rbuf, SOCKET_RETRY_INTV);
    if (res < 0){
        LOG_ERROR("Failed to get client GPU memory information.");
        exit(-1);
    }
    size_t offset = parse_meta_discard(rbuf);
    size_t total, used;
    parse_response_data(rbuf, offset, MEM_LIMIT, &used, &total);
    LOG_DEBUG("Received MEM_LIMIT, used = {}, limited = {}, remain = {}", used, total, total-used);
    return std::make_pair(total-used, total);
}


void Interception::update_gpu_mem(){
    LOG_INFO("update_gpu_mem thread is running.");
    MemByte remain, limit;
    std::tie(remain, limit) = get_gpu_mem_info();
    
    while(running){
        if(mem_update_queue.size() != 0){
            int is_allocate = mem_update_queue.front().first; 
            MemByte mem_size = mem_update_queue.front().second;
            mem_update_queue.pop();
            MsgBuffer sbuf, rbuf;
            ReqID req_id = new_request(sbuf, MEM_UPDATE, mem_size, is_allocate);
            int res = socket_communicate(sbuf, rbuf, SOCKET_RETRY_INTV);
            if (res < 0){
                LOG_ERROR("Failed to update client GPU memory usage.");
                grace_exit();
            }
            size_t offset = parse_meta_discard(rbuf);
            int is_succeed = 1;
            parse_response_data(rbuf, offset, MEM_UPDATE, &is_succeed, &remain);
            if(is_succeed != 1){
                LOG_ERROR("Cannot allocate/free memory from GPU client. Out of Memory.");
                grace_exit();
            }
        }else{
            std::unique_lock<std::mutex> m_lock(mem_mtx);
            gpu_mem_remain = remain;
            mem_update_cond.wait(m_lock);
            m_lock.unlock();
        }
    }
}

// int Interception::update_gpu_mem_usage(const MemByte &mem_size, const int &is_allocate){
//     MsgBuffer sbuf, rbuf;
//     ReqID req_id = new_request(sbuf, MEM_UPDATE, mem_size, is_allocate);
//     int res = socket_communicate(sbuf, rbuf, SOCKET_RETRY_INTV);
//     if (res < 0){
//         LOG_ERROR("Failed to update client GPU memory usage.");
//         exit(-1);
//     }
//     size_t offset = parse_meta_discard(rbuf);
//     int is_succeed = 1;
//     parse_response_data(rbuf, offset, MEM_LIMIT, &is_succeed);
//     return is_succeed;
// }

std::string t_client_name;
ReqID t_req_id;
RequestType t_req_t;
double Interception::get_token_from_gpu_client(double max_burst){
    MsgBuffer sbuf, rbuf;
    ReqID req_id = new_request(sbuf, REQ_TOKEN, overused_time, max_burst);
    int res = socket_communicate(sbuf, rbuf, 0);
    // size_t offset = parse_meta_discard(rbuf);
    //KONTON_TEST
    size_t offset = parse_meta(rbuf, t_client_name, t_req_id, t_req_t);
    // LOG_INFO("[KONTON_TEST] Get response: req_tytpe = {}, req_id =  {}, client_name = {}.", t_req_t, t_req_id, t_client_name.c_str());
    //KONTON_TEST_END
    double new_token = 0.0;
    parse_response_data(rbuf, offset, REQ_TOKEN, &new_token);
    // LOG_DEBUG("Get token time from gpu client: {}ms", new_token);
    return new_token;
}



// Process exit gracefully when the process meet the error;
void Interception::grace_exit(){
    running.store(false);
    exit_handle();
    usleep(50000);
    exit(-1);
}


void Interception::exit_handle(){
     for(auto devptrInfo:cuda_intercept.devptrs_mngr){
        cuMemPrefetchAsync(std::get<0>(devptrInfo), std::get<1>(devptrInfo), CU_DEVICE_CPU, 0);
    }
    // cudaDeviceSynchronize();
    cuCtxSynchronize();
    close(cuda_intercept.gc_sockfd);
    cuda_intercept.monitor_start_cond.notify_all();
    cuda_intercept.monitor_wait_cond.notify_all();
    cuda_intercept.mem_update_cond.notify_all();
    for(auto it=tids.begin(); it!=tids.end(); it++){
        pthread_cancel(*it);
    }
}



//sigint stop the program, so we advise the ptr to the host
void sigintHandler(int signum){
    LOG_DEBUG("Interrupt signal ( {} ) received. STOP the program.", signum);
    // for(auto devptrInfo:cuda_intercept.devptrs_mngr){
    //     cuMemPrefetchAsync(std::get<0>(devptrInfo), std::get<1>(devptrInfo), CU_DEVICE_CPU, 0);
    // }
    // cudaDeviceSynchronize();
    // running.store(false);
    // close(cuda_intercept.gc_sockfd);
    // cuda_intercept.monitor_start_cond.notify_all();
    // cuda_intercept.monitor_wait_cond.notify_all();
    // cuda_intercept.mem_update_cond.notify_all();
    // for(auto it=cuda_intercept.tids.begin(); it!=cuda_intercept.tids.end(); it++){
    //     pthread_cancel(*it);
    // }
    cuda_intercept.grace_exit();
    // usleep(500000);
}


void sigcontHandler(int signum){
    LOG_DEBUG( "Interrupt signal ( {} ) received. CONTINUE the program.\n", signum);
    for(auto devptrInfo:cuda_intercept.devptrs_mngr){
        cuMemPrefetchAsync(std::get<0>(devptrInfo), std::get<1>(devptrInfo), std::get<2>(devptrInfo), 0);
    }
    // cudaDeviceSynchronize();
    cuCtxSynchronize();
}


