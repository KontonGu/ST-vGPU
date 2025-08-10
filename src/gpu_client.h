/*
 * Modified on Mon May 20 2024
 *
 * Author: KontonGu
 * Copyright (c) 2024 TUM - CAPS Cloud
 * Licensed under the Apache License, Version 2.0 (the "License")
 */

#ifndef GPU_CLIENT_H
#define GPU_CLIENT_H

#include "msg_process.h"
#include "kernel_exec_record.h"

#include <string>
#include <map>
#include <mutex>
#include <thread>
#include <pthread.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <queue>
#include <condition_variable>
#include <map>

struct ReqInfo{
    ReqInfo(ReqID rd, MsgBuffer *t_msg): req_id(rd), msg(t_msg){}
    ReqID req_id;
    MsgBuffer *msg;
};



class GPUClient {
private:
    std::string client_name;
    uint16_t client_port;
    std::string sched_ip;
    uint16_t sched_port;

public:
    GPUClient() = default;
    void config_gpu_client_server();
    void config_sched_socket();
    void establish_sched_connection();
    void process_hook_request(int sockfd);
    int update_mem_usage(const MemByte &mem_size, const int &is_allocate, const int &sockfd);
    double request_new_token(const int &sockfd, const double &overused, const double &burst);
    void get_limit_mem();

    // communication with FaST scheduler
    void scheduler_send_process();
    void scheduler_recv_process();

    //thread management
    std::vector<pthread_t> tids;

    //socket management
    int sched_sockfd;
    int client_sockfd;
    sockaddr_in client_addr;

    // memory management
    std::mutex hook_mem_mtx;
    std::map<int, MemByte> hook_mem_allocated; // <sockfd, memory_usage>
    std::mutex hook_burst_mtx;
    std::map<int, double> hook_burst_map;
    MemByte gpu_mem_limit = 0;
    MemByte gpu_mem_used = 0;

    // producer and consumer model to communicate with the FaST scheduler
    std::queue<ReqInfo> req_fifo_queue;
    std::mutex req_fifo_mtx;
    std::condition_variable req_fifo_cond;
    std::map<ReqID, MsgBuffer> resp_map;
    std::mutex resp_mtx;
    std::condition_variable token_resp_cond;
    std::condition_variable resp_proc_ready_cond;
    bool resp_proc_ready = false;

    // quota update
    std::mutex quota_update_ctx;
    int is_quota_updating = 0;
    double client_max_overused = 0.0;
    std::condition_variable quota_update_cond;
    time_point start_tp;
    double updated_quota;

    
    // exit handling
    void grace_exit();

};

void sigintHandler(int signum);

#endif