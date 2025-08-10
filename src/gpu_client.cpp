/*
 * Modified on Mon May 20 2024
 *
 * Author: KontonGu
 * Copyright (c) 2024 TUM - CAPS Cloud
 * Licensed under the Apache License, Version 2.0 (the "License")
 * 
 * Note: The gpu_client design partly follows the structure of the paper:
 * https://ieeexplore.ieee.org/abstract/document/9566822
 * 
 */


#include "gpu_client.h"
#include "log_debug.h"

#include <unistd.h>
#include <cstring>
#include <climits>
#include <atomic>
#include <signal.h>

GPUClient gpu_client; 

// signal 
std::atomic<bool> running(true);

int main(){
    signal(SIGINT, sigintHandler);
    signal(SIGKILL, sigintHandler);

    gpu_client.config_gpu_client_server();
    LOG_INFO("GPU Client Socket bind and accept are finished.");
    gpu_client.config_sched_socket();
    gpu_client.establish_sched_connection();
    LOG_INFO("Establish connection to the scheduler");

    gpu_client.get_limit_mem();
    LOG_INFO("Get memory limited to: {}", gpu_client.gpu_mem_limit);

    std::thread sched_send_t(&GPUClient::scheduler_send_process, &gpu_client);
    sched_send_t.detach();
    gpu_client.tids.push_back(sched_send_t.native_handle());

    std::thread sched_recv_t(&GPUClient::scheduler_recv_process, &gpu_client);
    sched_recv_t.detach();
    gpu_client.tids.push_back(sched_recv_t.native_handle());

    gpu_client.start_tp = steady_clock::now();

    
    // wait for the connection from hook front-end
    int addr_len = sizeof(gpu_client.client_addr);
    int hook_sockfd = 0;
    while(running){
        hook_sockfd = accept(gpu_client.client_sockfd, (sockaddr *)&gpu_client.client_addr, (socklen_t *)&addr_len);
        if(hook_sockfd == -1){
            LOG_ERROR("GPU client accept() error: {}", strerror(errno));
            break;
        }
        std::cout << "New sockfd: " << hook_sockfd << std::endl;
        std::unique_lock<std::mutex> m_lock(gpu_client.hook_mem_mtx);
        gpu_client.hook_mem_allocated[hook_sockfd] = 0;
        m_lock.unlock();
        std::unique_lock<std::mutex> b_lock(gpu_client.hook_burst_mtx);
        gpu_client.hook_burst_map[hook_sockfd] = 0.0;
        b_lock.unlock();
        std::thread hook_process(&GPUClient::process_hook_request, &gpu_client, hook_sockfd);
        hook_process.detach();
        gpu_client.tids.push_back(hook_process.native_handle());
    }
}

 // Configure the socket connection to the hook front-end, bind and listen;
 // Get socket file descriptor.
void GPUClient::config_gpu_client_server(){
    char *c_name = getenv("GPU_CLIENT_NAME");
    if(c_name != NULL){
        client_name = c_name;
    }else{
        char host_name[HOST_NAME_MAX];
        gethostname(host_name, HOST_NAME_MAX);
        client_name = std::string(host_name);
    }
    char *c_port = getenv("GPU_CLIENT_PORT");
    if(c_port == NULL){
        LOG_ERROR("Cannot get gpu client port.");
        grace_exit();
    }
    client_port = atoi(c_port);
    LOG_INFO("GPU Client Port: {}", client_port);

    client_sockfd = socket(AF_INET, SOCK_STREAM, 0);
    client_addr.sin_family = AF_INET;
    client_addr.sin_port = htons(client_port);
    client_addr.sin_addr.s_addr = INADDR_ANY;
    
    int res = multiple_try(
        [&]() -> int {
            return bind(client_sockfd, (sockaddr *)&client_addr, sizeof(client_addr));
        }, SOCKET_MAX_TRY, 1);
    if(res == -1){
        LOG_ERROR("Failed to Bind Socket: error = {}", strerror(errno));
        close(client_sockfd);
        grace_exit();
    }
    res = listen(client_sockfd, SOMAXCONN);
    if(res == -1){
        LOG_ERROR("Failed to Listen Socket: error = {}", strerror(errno));
        close(client_sockfd);
        grace_exit();
    }
}

// Configure the IP and Port for the connection to the gpu scheduler;
// Get the scheduler IP and Port.
void GPUClient::config_sched_socket(){
    char *s_ip = getenv("SCHEDULER_IP");
    if(s_ip == NULL){
        LOG_ERROR("Cannot get scheduler IP.");
        grace_exit();
    }
    sched_ip = s_ip;
    char *s_port = getenv("SCHEDULER_PORT");
    if(s_port == NULL){
        LOG_ERROR("Cannot get scheduler Port.");
        grace_exit();
    }
    sched_port = atoi(s_port);
    LOG_INFO("Scheduler IP and Port: {}:{}", sched_ip, sched_port);
}


// Establish connection to the scheduler;
// Get the scheduler socket file descriptor.
void GPUClient::establish_sched_connection(){
    sched_sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if(sched_sockfd == -1){
        LOG_ERROR("Error creating socket!");
        grace_exit();
    }
    struct sockaddr_in sched_addr;
    int addr_len = sizeof(sched_addr);
    std::memset(&sched_addr, 0, addr_len);
    sched_addr.sin_family = AF_INET;
    sched_addr.sin_port = htons(sched_port);
    sched_addr.sin_addr.s_addr = inet_addr(sched_ip.c_str());
    int res = multiple_try(
        [&]() -> int {return connect(sched_sockfd, (sockaddr *)&sched_addr, (socklen_t) addr_len);},
        SOCKET_MAX_TRY, SOCKET_RETRY_INTV
    );
    if(res == -1){
        LOG_ERROR("Error connecting scheduler socket server!, error = {}", strerror(errno));
        close(sched_sockfd);
        grace_exit();
    }
}


// Get the limit memory information of the gpu client from scheduler
void GPUClient::get_limit_mem(){
    MsgBuffer smsg, rmsg;
    new_request(smsg, MEM_LIMIT);
    char rec_buf[MSGBUFFER_MAX_LEN];
    int res =multiple_try(
        [&]() -> int {
            if(send(sched_sockfd, smsg.getData(), MSGBUFFER_MAX_LEN, 0) <= 0){
                LOG_DEBUG("socket message sending error: {}", strerror(errno));
                close(sched_sockfd);
                grace_exit();
            }
            if(recv(sched_sockfd, rec_buf, MSGBUFFER_MAX_LEN, 0) <= 0){
                LOG_DEBUG("socket message receving error: {}", strerror(errno));
                close(sched_sockfd);
                grace_exit();
            }
            return 0;
        }, SOCKET_MAX_TRY, SOCKET_RETRY_INTV);
    if(res != 0){
        LOG_DEBUG("Failed to get limit memory information from scheduler.");
        close(sched_sockfd);
        grace_exit();
    }
    rmsg.copy2MsgBuffer(rec_buf, MSGBUFFER_MAX_LEN);
    size_t offset = parse_meta_discard(rmsg);
    size_t mem_used; // ignore the memory used, only mem limit; memory is not managed by scheduler;
    parse_response_data(rmsg, offset, MEM_LIMIT, &mem_used, &gpu_mem_limit);
}



// Process the reqeust from hook front-end;
// Request type includes: REQ_TOKEN, MEM_LIMIT, MEM_UPDATE;
// Generate and send the response corresponding to the request.
void GPUClient::process_hook_request(int sockfd){
    LOG_DEBUG("Entered the process_hook_request");
    hook_mem_allocated[sockfd] = 0;
    while(running){
        char rec_buf[MSGBUFFER_MAX_LEN];
        if(recv(sockfd, rec_buf, MSGBUFFER_MAX_LEN, 0) <= 0){
            LOG_WARNING("Socket recv error/disconnection while processing hook process: {}", strerror(errno));
            gpu_mem_used -= hook_mem_allocated[sockfd];
            break;
        }
        MsgBuffer rbuf, sbuf;
        rbuf.copy2MsgBuffer(rec_buf, MSGBUFFER_MAX_LEN);
        std::string c_name;
        ReqID req_id;
        RequestType req_type;
        size_t offset = 0;
        offset = parse_meta(rbuf, c_name, req_id, req_type);
        // LOG_DEBUG("c_name: {}, req_id = {}, req_type = {}", c_name, req_id, req_type);
        if(req_type == REQ_TOKEN){
            double overused, burst;
            parse_request_data(rbuf, offset, req_type, &overused, &burst);
            LOG_DEBUG("Received [REQ_TOKEN] Request: req_id = {}, overuse = {}, burst={}", req_id, overused, burst);
            double new_token = request_new_token(sockfd, overused, burst);
            // KONTON_TEST
            // usleep(20000);
            // double new_token = 22.0;
            // KONTON_TEST_END
            new_response(sbuf, c_name, req_id, REQ_TOKEN, new_token);
        }else if(req_type == MEM_LIMIT){
            LOG_DEBUG("Received [MEM_LIMIT] Request: req_id = {}, client_name = {}", req_id, c_name);
            std::unique_lock<std::mutex> m_lock(hook_mem_mtx);
            new_response(sbuf, c_name, req_id, MEM_LIMIT, gpu_mem_used, gpu_mem_limit);
            m_lock.unlock();
        }else if(req_type == MEM_UPDATE){
            int is_allocate;
            MemByte mem_size;
            parse_request_data(rbuf, offset, req_type, &mem_size, &is_allocate);
            LOG_DEBUG("Received [MEM_UPDATE] Request: req_id = {}, client_name = {}, mem_size = {}, is_allocate = {}", req_id, client_name, mem_size, is_allocate);
            int is_success = update_mem_usage(mem_size, is_allocate, sockfd);
            MemByte remain = gpu_mem_limit - gpu_mem_used;
            new_response(sbuf, c_name, req_id, MEM_UPDATE, is_success, remain);
        }
        if(sbuf.size() > 0){
            if(send(sockfd, sbuf.getData(), MSGBUFFER_MAX_LEN, 0) <= 0){
                LOG_ERROR("Failed to send message to the hook front-end: client_name = {}, req_type = {}", c_name, req_type);
            }
        }
    }
    close(sockfd);
}


// Update the the GPU memory usage based on available total memory of the GPU client;
// is_allocate = 0 means free the memory, and whereas =1 means allocation;
// Return 1 if the update is successful;  
int GPUClient::update_mem_usage(const MemByte &mem_size, const int &is_allocate, const int &sockfd){
    int is_succeed = 1;
    std::unique_lock<std::mutex> m_lock(hook_mem_mtx);
    if(is_allocate == 1){
        if(gpu_mem_used + mem_size <= gpu_mem_limit){
            gpu_mem_used += mem_size;
            hook_mem_allocated[sockfd] += mem_size;
        }else{
            is_succeed = 0;
        }
    }else if(is_allocate == 0){
        gpu_mem_used -= mem_size;
        hook_mem_allocated[sockfd] -= mem_size;
    }
    m_lock.unlock();
    LOG_DEBUG("The state of memory: gpu_mem_used = {}", gpu_mem_used);
    return is_succeed;
}


// Get the new avaialble token based on the gpu client quota allocated from the scheduler;
// Return the remaining quota as a new token;
// double GPUClient::request_new_token(const int &sockfd, const double &overused, const double &burst){
//     client_max_overused = std::max(client_max_overused, overused);
//     std::unique_lock<std::mutex> q_lock(quota_update_ctx);
//     while(is_quota_updating != 0){
//         quota_update_cond.wait(q_lock); 
//     }
//     std::unique_lock<std::mutex> b_lock(hook_burst_mtx);
//     hook_burst_map[sockfd] = burst;
//     b_lock.unlock();

//     time_point now_tp = steady_clock::now();
//     double itv_since_updated = duration_cast<microseconds>(now_tp - start_tp).count() / 1e3;
//     q_lock.unlock();
//     if(itv_since_updated + burst > updated_quota){
//         q_lock.lock();
//         is_quota_updating = 1;
//         q_lock.unlock();
//         itv_since_updated = 0.0;
//         double max_burst;
//         // retrive the maximum burst of all hook clients
//         b_lock.lock();
//         for(auto x: hook_burst_map) max_burst = std::max(x.second, max_burst);
//         b_lock.unlock();

//         MsgBuffer *msg = new MsgBuffer;
//         ReqID req_id = new_request(*msg, REQ_TOKEN, client_max_overused, max_burst);
//         std::unique_lock<std::mutex> f_lock(req_fifo_mtx);
//         ReqInfo req_info(req_id, msg);
//         req_fifo_queue.push(req_info);
//         req_fifo_cond.notify_one();
//         f_lock.unlock();
        

//         bool received=false;
//         while(!received){
//             std::unique_lock<std::mutex> resp_lock(resp_mtx);
//             token_resp_cond.wait(resp_lock);
//             if(resp_map.find(req_id) != resp_map.end()){
//                 received = true;
//                 MsgBuffer rbuf = resp_map[req_id];
//                 size_t offset = parse_meta_discard(rbuf);
//                 parse_response_data(rbuf, offset, REQ_TOKEN, &updated_quota);
//                 LOG_DEBUG("Received quota: quota = {}", updated_quota);
//                 client_max_overused = 0.0;
//                 start_tp = steady_clock::now();
//                 resp_map.erase(req_id);
//             }
//             resp_lock.unlock();
//         }
//         is_quota_updating = 0;
//         quota_update_cond.notify_all();
//     }
//     LOG_DEBUG("Quota sent to the front-end app: {}, itv_since_updated = {}", updated_quota - itv_since_updated, itv_since_updated);
//     // return updated_quota - itv_since_updated;
//     return updated_quota;
// }

// without considering the overuse and burst
double GPUClient::request_new_token(const int &sockfd, const double &overused, const double &burst){
    client_max_overused = std::max(client_max_overused, overused);
    std::unique_lock<std::mutex> q_lock(quota_update_ctx);
    
    MsgBuffer *msg = new MsgBuffer;
    ReqID req_id = new_request(*msg, REQ_TOKEN, client_max_overused, burst);
    std::unique_lock<std::mutex> f_lock(req_fifo_mtx);
    ReqInfo req_info(req_id, msg);
    req_fifo_queue.push(req_info);
    req_fifo_cond.notify_one();
    f_lock.unlock();


    bool received=false;
    while(!received){
        std::unique_lock<std::mutex> resp_lock(resp_mtx);
        resp_proc_ready = true;
        resp_proc_ready_cond.notify_all();
        token_resp_cond.wait(resp_lock);
        if(resp_map.find(req_id) != resp_map.end()){
            received = true;
            MsgBuffer rbuf = resp_map[req_id];
            size_t offset = parse_meta_discard(rbuf);
            parse_response_data(rbuf, offset, REQ_TOKEN, &updated_quota);
            LOG_DEBUG("Received quota: quota = {}", updated_quota);
            client_max_overused = 0.0;
            start_tp = steady_clock::now();
            resp_map.erase(req_id);
        }
        resp_proc_ready = false;
        resp_lock.unlock();
    }
    q_lock.unlock();
    return updated_quota;
}



// Process the request sending to the scheduler;
void GPUClient::scheduler_send_process(){
    while(running){
        std::unique_lock<std::mutex> s_lock(req_fifo_mtx);
        LOG_DEBUG("Running scheduler_send_process, waiting for req_fifo_queue items ....");
        req_fifo_cond.wait(s_lock);
        LOG_DEBUG("Got element in the req_fifo_queue.");
        if(!req_fifo_queue.empty()){
            ReqInfo req_info = req_fifo_queue.front();
            if(send(sched_sockfd, req_info.msg->getData(), MSGBUFFER_MAX_LEN, 0) <= 0){
                 LOG_DEBUG("socket message sending error: {}", strerror(errno));
                 close(sched_sockfd);
                 grace_exit();
            }
            delete req_info.msg;
            req_fifo_queue.pop();
        }
        s_lock.unlock();
    }
}


// Process the response received from the scheduler;
void GPUClient::scheduler_recv_process(){
    char rec_buf[MSGBUFFER_MAX_LEN]; 
    while(running){
        LOG_DEBUG("Running scheduler_recv_process, socket recv waiting .....");
        if(recv(sched_sockfd, rec_buf, MSGBUFFER_MAX_LEN, 0) <= 0){
             LOG_ERROR("socket message receving error: {}", strerror(errno));
             close(sched_sockfd);
             grace_exit();
        }
        MsgBuffer rbuf;
        rbuf.copy2MsgBuffer(rec_buf, MSGBUFFER_MAX_LEN);
        std::string cn; ReqID req_id; RequestType req_t;
        parse_meta(rbuf, cn, req_id, req_t);
        LOG_DEBUG("receive message from scheduler. client_name = {}, req_id = {}, req_type = {}", cn, req_id, req_t);

        std::unique_lock<std::mutex> r_lock(resp_mtx);
        resp_map.insert(std::make_pair(req_id, rbuf));
        if(req_t == REQ_TOKEN){
            if(!resp_proc_ready){
                resp_proc_ready_cond.wait(r_lock);
            }
            token_resp_cond.notify_all();
        }
        r_lock.unlock();
    }
}


// Process exit gracefully when the process meet the error;
void GPUClient::grace_exit(){
    running.store(false);
    for(auto it=gpu_client.tids.begin(); it!=gpu_client.tids.end(); it++){
        pthread_cancel(*it);
    }
    usleep(50000);
    exit(-1);
}


//sigint stop the program 
void sigintHandler(int signum){
    LOG_DEBUG("Interrupt signal ( {} ) received. STOP the program.", signum);
    running.store(false);
    for(auto it=gpu_client.tids.begin(); it!=gpu_client.tids.end(); it++){
        pthread_cancel(*it);
    }
    usleep(50000);
    exit(-1);
}





