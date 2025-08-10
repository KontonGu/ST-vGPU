/*
 * Modified on Sat May 18 2024
 *
 * Author: KontonGu
 * Copyright (c) 2024 TUM - CAPS Cloud
 * Licensed under the Apache License, Version 2.0 (the "License")
 */

#ifndef CUDAHOOK_H
#define CUDAHOOK_H

#pragma once

#include <string.h>
#include <dlfcn.h>
#include <list>
#include <cuda.h>
#include <cuda_runtime.h>
#include <unistd.h>

#include <chrono>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <map>
#include <gnu/libc-version.h>
#include <queue>
#include <vector>

#include "msg_process.h"
#include "hook_util.h"
#include "log_debug.h"
// #include "test/cuda.h"

using std::chrono::steady_clock;
using std::chrono::microseconds;
using std::chrono::duration_cast;


#define SYMBOL_TO_STRING(x) stringtify(x)
#define stringtify(x) #x

inline bool stringCompare(const char* st1, const char *st2){
    return strcmp(st1, st2) == 0;
}

extern "C"{
    void *__libc_dlsym(void *map, const char *name);
    void *__libc_dlopen_mode(const char *name, int mode);   
}

using oriDlsymFn = void *(void *, const char *);
// void *libdl_handle = __libc_dlopen_mode("libdl.so.2", RTLD_LAZY);
void *libdl_handle = dlopen("libdl.so.2", RTLD_LAZY);
void *libc_handle = dlopen("libc.so.6", RTLD_LAZY);

oriDlsymFn* get_dlsym_ptr();

CUresult getProcAddressBySymbol(const char *symbol, void **pfn, int driverVersion, cuuint64_t flags,
                                CUdriverProcAddressQueryResult *symbolStatus);


void *dlsym(void *handle, const char *symbol){
    // LOG_INFO("ALL symbol: {}", symbol);
    if(stringCompare(symbol, SYMBOL_TO_STRING(cuGetProcAddress))){
        LOG_DEBUG("dlsym symbol: {}", symbol);
        return (void *) &getProcAddressBySymbol;
    }
    // std::cout << "all other dlsym: " << symbol << std::endl;
    // static auto normalDsym = (oriDlsymFn*) __libc_dlsym(libdl_handle, "dlsym");
    // static auto normalDsym = (oriDlsymFn*) dlvsym(libc_handle, "__libc_dlsym", NULL);
    static auto normalDlsym = get_dlsym_ptr();
    return (*normalDlsym)(handle, symbol);
}

typedef enum HookSymbolsIndexEnum {
  CU_HOOK_GET_PROC_ADDRESS,
  CU_HOOK_MEM_ALLOC,
  CU_HOOK_MEM_ALLOC_MANAGED,
  CU_HOOK_MEM_ALLOC_PITCH,
  CU_HOOK_MEM_ALLOC_ASYNC,
  CU_HOOK_MEM_FREE,
  CU_HOOK_ARRAY_CREATE,
  CU_HOOK_ARRAY3D_CREATE,
  CU_HOOK_MIPMAPPED_ARRAY_CREATE,
  CU_HOOK_ARRAY_DESTROY,
  CU_HOOK_MIPMAPPED_ARRAY_DESTROY,
  CU_HOOK_CTX_GET_CURRENT,
  CU_HOOK_CTX_CREATE,
  CU_HOOK_CTX_SET_CURRENT,
  CU_HOOK_CTX_DESTROY,
  CU_HOOK_LAUNCH_KERNEL,
  CU_HOOK_LAUNCH_COOPERATIVE_KERNEL,
  CU_HOOK_DEVICE_TOTOAL_MEM,
  CU_HOOK_MEM_INFO,
  CU_HOOK_CTX_SYNC,
  CU_HOOK_MEMCPY_ATOH,
  CU_HOOK_MEMCPY_DTOH,
  CU_HOOK_MEMCPY_DTOH_ASYNC,
  CU_HOOK_MEMCPY_HTOA,
  CU_HOOK_MEMCPY_HTOD,
  CU_HOOK_MEMCPY_HTOD_ASYNC,
  NUM_HOOK_SYMBOLS,
} HookSymbolsIndex;



class Interception {

private:
    // std::string gpu_client_ip_file = "/fastpod/library/GPUClientIP.txt";
    // std::string gpu_client_ip_file = "/home/ubuntu/konton_ws/git_repo/FaST-Manager/test/GPUClientIP.txt";
    std::string gpu_client_ip_file = "/fastpod/library/GPUClientsIP.txt";

    // currently the gpu client port is obtained from ENV variable
    // std::string gpu_client_port_file = "/fastpod/library/GPUClientPort.txt";

    // configuration data for the connection to the client GPU
    std::string client_gpu_ip = "127.0.0.1";
    // the default gpu client port is 56001
    uint16_t client_gpu_port = 56001;

public:
    Interception(){
        token_start = steady_clock::now();
    }

    // connection to the client GPU
    void config_connection();
    int establish_connection();
    int socket_communicate(MsgBuffer &sbuf, MsgBuffer &rbuf, int sock_timeout);  

    // token management
    double get_token_from_gpu_client(double max_burst);

    // connection to the client GPU
    int gc_sockfd; // client gpu socket file descriptor; 
    std::mutex comm_mtx;  // communication mutex
    
    double ms_since_token_start(){
        return duration_cast<microseconds>(steady_clock::now() - token_start).count() / 1e3;
    }
    
    // time of token got from gpu_client
    double token_time;

    // time usage monitoring
    void cuda_execution_monitor();
    double overused_time;
    std::chrono::time_point<std::chrono::steady_clock> token_start;
    std::mutex monitor_mtx;
    std::condition_variable monitor_start_cond;
    std::condition_variable monitor_wait_cond;
    std::condition_variable monitor_finished_cond;
    std::condition_variable monitor_ready_cond;
    bool monitor_finished;
    bool monitor_ready;
    std::mutex kernel_mtx;

    // memory and device
    std::list<std::tuple<CUdeviceptr, size_t, CUdevice>> devptrs_mngr;
    std::map<CUdeviceptr, size_t> mem_addr_size_record;
    MemByte gpu_mem_usage = 0;
    std::mutex mem_mtx;
    MemByte gpu_mem_limit = 0;
    MemByte gpu_mem_remain = 0;
    std::queue<std::pair<int, MemByte>> mem_update_queue;
    std::condition_variable mem_update_cond;
    std::pair<size_t, size_t> get_gpu_mem_info();
    // int update_gpu_mem_usage(const MemByte &mem_size, const int &is_allocate);
    void update_gpu_mem();

    // event recording execution time
    // cudaEvent_t event_start;
    CUevent event_start;
    CUevent event_stop;
    CUcontext cur_ctx;

    //thread management
    std::vector<pthread_t> tids;

    //exit handling
    void grace_exit();
    void exit_handle();

};


#endif









