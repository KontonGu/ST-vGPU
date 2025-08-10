/*
 * Modified on Mon May 20 2024
 *
 * Author: KontonGu
 * Copyright (c) 2024 TUM - CAPS Cloud
 * Licensed under the Apache License, Version 2.0 (the "License")
 */

#ifndef FAST_SCHEDULER_H
#define FAST_SCHEDULER_H

#include <log_debug.h>
#include <msg_process.h>

#include <map>
#include <vector>
#include <list>
#include <pthread.h>
#include <thread>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <mutex>
#include <condition_variable>
#include <chrono>


using std::chrono::steady_clock;
using std::chrono::microseconds;
using std::chrono::milliseconds;
using std::chrono::duration_cast;
typedef std::chrono::time_point<std::chrono::steady_clock> time_point;

#define SM_GLOBAL_LIMIT 100    // 100 partitions of sm in a GPU

struct sched_entity{
    std::string c_name;
    int sockfd;
    ReqID req_id;
    double expired_tp;
    double recv_tp;
};

struct scheduable_entity{
    double used;
    double delta_req;  // request_quota - used;
    double delta_limit;  // limit quota - used;
    std::list<sched_entity>::iterator entity_iter;
};

struct sched_record{
    std::string c_name;
    double start_tp;
    double expired_tp;
};


class GPUClientInfo{
public:

    GPUClientInfo(std::string c_n, double ma_q, double mi_q, size_t sm_p, size_t mem):
                 client_name(c_n), max_quota(ma_q), min_quota(mi_q), sm_partition(sm_p), mem_limit(mem){}
    double get_max_quota() const {return max_quota;}
    double get_min_quota() const {return min_quota;}
    MemByte get_mem_limit() const {return mem_limit;}
    inline double get_sm_partition(){return sm_partition;}
    void update_overused(const double &o_u){ c_overused = o_u;}
    void update_burst(const double &bt){ c_burst = bt;}


    std::string client_name;
    double alloc_quota;
    size_t mem_usage;
    double c_burst;
    double c_overused;

private:
    double min_quota;
    double max_quota;
    size_t sm_partition;
    MemByte mem_limit;
};




class FastScheduler{
private:
    int clients_num = 0;
    time_point sched_start_tp;
public:
    FastScheduler(){
        sched_start_tp = steady_clock::now();
        sm_occupied = 0;
    }
    void read_config_file();
    // Monitor the change of the resource configuraiton file where clients' resource are defined;
    void monitor_config_file();
    // Configure the scheduler socket server for gpu clients' connection and requests;
    void config_scheduler_server();
    // Process the requests from gpu clients for token and memory limit;
    int process_client_request(int sockfd);
    // Get the time point since the program is started;
    inline double get_now_tp(){
        return duration_cast<microseconds>(steady_clock::now() - sched_start_tp).count() / 1e3;
    }

    // Select next runnable entities based on scheduling policy
    std::vector<sched_entity> pick_next_runnable_entities();

    // Schedule tokens to the runnable gpu clients based on the scheduling policy;
    void schedule(); 
    // Release the sm occupancy of gpu clients with expired token;
    bool update_tokens();
    // The priority of clients to get tokens in the schduling policy
    static bool schedule_policy_priority(const scheduable_entity& en_a, const scheduable_entity &en_b);


    // record the start time and expired time of a runnable client in the list of clients_record
    // for further scheudling policy usage.  
    void record(const std::string &cn, const double &quota);

    // Release token takers that have expired end time and relase their sm occupancy 
    bool release_token_taker(const std::string &cn);

    // Get the timepoint after itv_ms ms since now
    time_point tp_after_duration(const double &itv_ms);

    // Consider overused in the client to update its end time
    void update_actual_token_end_time(std::string c_n, double over_used);

    // Return the next allocated quota in miliseconds for client
    double get_next_quota(const GPUClientInfo & client);


    // exit handling
    void exit_handle();
    void grace_exit();

    //scheduler socket server managemnt
    uint16_t sched_port;
    int sched_sockfd;
    sockaddr_in sched_addr;
    
    //scheduler resource management
    std::string clients_conf_file;
    size_t gpu_mem_used;
    size_t sm_occupied;
    double time_window;
    std::map<std::string, GPUClientInfo*> clients_info;  // <client_name, ClientResource Info>
    std::mutex clients_info_mtx;

    //scheduling policy management
    std::list<sched_entity> ready_list;
    std::mutex ready_list_mtx;
    std::condition_variable ready_list_cond;
    std::list<sched_entity> token_takers;
    std::list<sched_entity>::iterator closest_token_entity;
    std::list<sched_record> clients_record;


    //thread management
    //key < 0 means shceudler internal thread, key >  0 are for gpu client processing threads.
    std::map<int, pthread_t> tids;
    int intern_id = -1;
};



#endif