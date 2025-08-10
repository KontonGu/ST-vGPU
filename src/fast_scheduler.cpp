/*
 * Modified on Mon May 20 2024
 *
 * Author: KontonGu
 * Copyright (c) 2024 TUM - CAPS Cloud
 * Licensed under the Apache License, Version 2.0 (the "License")
 */
#include <fast_scheduler.h>

#include <CLI11.hpp>
#include <sstream>
#include <atomic>
#include <unistd.h>
#include <libgen.h>
#include <sys/inotify.h>
#include <algorithm>

#define EVENT_SIZE (sizeof(struct inotify_event))
#define EVENT_BUF_LEN (1024 * (EVENT_SIZE + 16))


// signal 
std::atomic<bool> running(true);

FastScheduler fast_scheduler;

int main(int argc, char **argv){
    CLI::App config("FaST Scheduler Backend");
    config.add_option("-p,--sched_port", fast_scheduler.sched_port, "The socket port of the scheduler backend.")->required();
    config.add_option("-w,--window", fast_scheduler.time_window, "The time window for the scheudler.")->required();
    config.add_option("-f,--clients_config", fast_scheduler.clients_conf_file, 
                     "The resource configuration file of clients.")->required();

    config.set_help_all_flag("--help-all", "Show help for all options");

    CLI11_PARSE(config, argc, argv);

    LOG_INFO("The scheudler backend, port = {}, time_window = {}, clients_conf_file = {}", 
            fast_scheduler.sched_port, fast_scheduler.time_window, fast_scheduler.clients_conf_file);

    fast_scheduler.read_config_file();
    std::thread config_monitor_thread(&FastScheduler::monitor_config_file, &fast_scheduler);
    config_monitor_thread.detach();
    fast_scheduler.tids[fast_scheduler.intern_id--] = config_monitor_thread.native_handle();

    fast_scheduler.config_scheduler_server();

    std::thread schedule_thread(&FastScheduler::schedule, &fast_scheduler);
    schedule_thread.detach();
    fast_scheduler.tids[fast_scheduler.intern_id--] = schedule_thread.native_handle();

    int addr_len = sizeof(fast_scheduler.sched_addr);
    int client_sockfd = 0;
    while(running){
        client_sockfd = accept(fast_scheduler.sched_sockfd, (struct sockaddr *)&fast_scheduler.sched_addr, 
                               (socklen_t*)&addr_len);
        if(client_sockfd == -1){
            LOG_ERROR("GPU Scheduler accept() error: {}", strerror(errno));
            continue;
        }
        std::thread proc_client(&FastScheduler::process_client_request, &fast_scheduler, client_sockfd);
        proc_client.detach();
        fast_scheduler.tids[client_sockfd] = proc_client.native_handle();
    }
    close(fast_scheduler.sched_sockfd);
    fast_scheduler.grace_exit();
}


void FastScheduler::read_config_file(){
    std::ifstream ifile;
    ifile.open(clients_conf_file, std::ios::in);
    if(!ifile.is_open()){
        LOG_ERROR("Failed to open clients' resource configuration file");
        grace_exit();
    }
    ifile >> clients_num;
    std::string client_name; 
    double max_q, min_q;
    size_t sm_partition, mem_limit;
    // std::unique_lock<std::mutex> ci_lock(clients_info_mtx);
    for(int i=0; i<clients_num; i++){
        ifile >> client_name >> min_q >> max_q >> sm_partition >> mem_limit;
        GPUClientInfo *new_c = new GPUClientInfo(client_name, max_q, min_q, sm_partition, mem_limit);
        if(clients_info.find(client_name) != clients_info.end()){
            delete clients_info[client_name];
        }
        clients_info[client_name] = new_c;
        LOG_INFO("New Client with: name = {}, max_q = {}, min_q = {}, sm_partition = {}, mem_limit = {}", 
                client_name, max_q, min_q, sm_partition, mem_limit);
    }
    // ci_lock.unlock();
    ifile.close();
}


void FastScheduler::monitor_config_file(){
    std::string config_dir, file_name;
    char * config_file = new char[clients_conf_file.length()+1];
    strcpy(config_file, clients_conf_file.c_str());
    config_dir = dirname(config_file);
    strcpy(config_file, clients_conf_file.c_str());
    file_name = basename(config_file);
    delete [] config_file;

    char buf[EVENT_BUF_LEN];
    int ifd = inotify_init();
    if(ifd < 0){
        LOG_ERROR("Failed to initilize inotify.");
        grace_exit();
    }

    int buf_len, idx = 0;
    int wd = inotify_add_watch(ifd, config_dir.c_str(), IN_CLOSE_WRITE);
    while(running){
        idx = 0;
        buf_len = read(ifd, buf, EVENT_BUF_LEN);
        if(buf_len < 0){
            LOG_ERROR("Inotify buf read error.");
            grace_exit();
        }
        while(idx < buf_len){
            struct inotify_event *event = (struct inotify_event *) &buf[idx];
            if(event->len){
                LOG_DEBUG("Inotify new event, name = {}", event->name);
                if(event->mask & IN_CLOSE_WRITE){
                    if(strcmp((const char*)event->name, file_name.c_str()) == 0){
                        read_config_file();
                    }
                }
            }
            idx += EVENT_SIZE + event->len;
        }
    }
    inotify_rm_watch(ifd, wd);
    close(ifd);
}



void FastScheduler::config_scheduler_server(){
    sched_sockfd = socket(AF_INET, SOCK_STREAM, 0);
    sched_addr.sin_family = AF_INET;
    sched_addr.sin_port = htons(sched_port);
    sched_addr.sin_addr.s_addr = INADDR_ANY;

    int res = multiple_try(
        [&]() -> int {
            return bind(sched_sockfd, (sockaddr *)&sched_addr, sizeof(sched_addr));
        }, SOCKET_MAX_TRY, 1);
    if(res == -1){
        LOG_DEBUG("Failed to Bind Socket: error = {}", strerror(errno));
        close(sched_sockfd);
        grace_exit();
    }
    res = listen(sched_sockfd, SOMAXCONN);
    if(res == -1){
        LOG_DEBUG("Failed to Listen Socket: error = {}", strerror(errno));
        close(sched_sockfd);
        grace_exit();
    }
}


int FastScheduler::process_client_request(int sockfd){
    LOG_DEBUG("Entered the func thread(new gpu client): [process_client_request].");
    while(running){
        char rec_buf[MSGBUFFER_MAX_LEN];
        if(recv(sockfd, rec_buf, MSGBUFFER_MAX_LEN, 0) <= 0){
            LOG_WARNING("Socket recv error/disconnection while processing client request: {}", strerror(errno));
            break;
        }
        MsgBuffer rbuf;
        rbuf.copy2MsgBuffer(rec_buf, MSGBUFFER_MAX_LEN);
        std::string c_name;
        ReqID req_id;
        RequestType req_type;
        size_t offset = 0;
        offset = parse_meta(rbuf, c_name, req_id, req_type);
        // std::unique_lock<std::mutex> ci_lock(clients_info_mtx);
        if(clients_info.find(c_name) == clients_info.end()){
            LOG_INFO("The client name is unknown.");
            return -1;
        }
        GPUClientInfo *client = clients_info[c_name];
        if(req_type == REQ_TOKEN){
            double overused, burst;
            parse_request_data(rbuf, offset, req_type, &overused, &burst);
            LOG_DEBUG("Received [REQ_TOKEN] Request: client_name = {}, req_id = {}, overuse = {}, burst={}, timestamp = {}", c_name, req_id, overused, burst, get_now_tp());
            client->update_overused(overused);
            client->update_burst(burst);
            fast_scheduler.update_actual_token_end_time(c_name, overused);
            sched_entity entity = {c_name, sockfd, req_id, -1.0, get_now_tp()};
            std::unique_lock<std::mutex> r_lock(ready_list_mtx);
            ready_list.push_back(entity);
            ready_list_cond.notify_one();  
            r_lock.unlock();

            // KONTON_TEST
            // MsgBuffer sbuf;
            // double quota = client->get_max_quota() * time_window;
            // quota = 20.0;
            // usleep(20000);
            // new_response(sbuf, c_name, req_id, REQ_TOKEN, quota);
            // if(send(sockfd, sbuf.getData(), MSGBUFFER_MAX_LEN, 0) < 0){
            //     LOG_ERROR("Sending the response of REQ_TOKEN failed. client_name = {}.", c_name);
            // }
            // KONTON_TEST_END
        }else if(req_type == MEM_LIMIT){
            LOG_DEBUG("Received [MEM_LIMIT] Request: req_id = {}, client_name = {}", req_id, c_name);
            MsgBuffer sbuf;
            MemByte used = 0;
            // ignore the gpu memory usage (managed by the gpu client), only return the gpu memory limitation;
            new_response(sbuf, c_name, req_id, MEM_LIMIT, used, client->get_mem_limit()); 
            int res = multiple_try(
                [&]() -> int {
                    if(send(sockfd, sbuf.getData(), MSGBUFFER_MAX_LEN, 0) < 0) return -1;
                    else return 0;
                }, SOCKET_MAX_TRY, 1);
            if(res < 0){
                LOG_ERROR("Sending the response of memory limit failed. client_name = {}.", c_name);
                continue;
            }
        }
        // ci_lock.unlock();
    }
    close(sockfd);
    tids.erase(sockfd);
    return 0;
}

void FastScheduler::schedule(){
    double c_quota, c_sm_partition;
    LOG_INFO("Entered the func thread: [schedule].");
    while(running){
        std::unique_lock<std::mutex> r_lock(ready_list_mtx);
        // std::unique_lock<std::mutex> ci_lock(clients_info_mtx);
        if(ready_list.size() == 0){
            LOG_DEBUG("Ready List is empty: wait for entity in ready list.");
            // ci_lock.unlock();
            ready_list_cond.wait(r_lock);
            r_lock.unlock();
        }else{
            r_lock.unlock();
            update_tokens(); // remove expired token takers and update sm occupancy;
            LOG_DEBUG("starting pick_next_runnable_entities");
            std::vector<sched_entity> next_entities = pick_next_runnable_entities();
            for(auto &item: next_entities){
                c_quota = get_next_quota(*(clients_info[item.c_name]));
                c_sm_partition = clients_info[item.c_name]->get_sm_partition();
                record(item.c_name, c_quota);
                LOG_DEBUG("Starting generating response for client = {}, quota = {}, sm_partition = {}", item.c_name, c_quota, c_sm_partition);
                MsgBuffer sbuf;
                new_response(sbuf, item.c_name, item.req_id, REQ_TOKEN, c_quota);
                int rc = multiple_try(
                    [&]() -> int {
                        if(send(item.sockfd, sbuf.getData(), MSGBUFFER_MAX_LEN, 0) <= 0){
                            LOG_DEBUG("socket message sending error: {}", strerror(errno));
                            return -1;
                        }else{
                            return 0;
                        }
                    },SOCKET_MAX_TRY, 1);
                if(rc < 0){
                    LOG_ERROR("Sending quota failed.");
                    close(item.sockfd);
                    grace_exit();
                }
                item.expired_tp = get_now_tp() + c_quota;
                sm_occupied += c_sm_partition;
                token_takers.emplace_back(item);
                LOG_DEBUG("Scheduled timestamp: {}", get_now_tp());
            }

            bool should_wait = update_tokens();
            LOG_DEBUG("Schedule wait for next client request: should_wait = {}, token_taker_size = {}", should_wait, token_takers.size());
            std::unique_lock<std::mutex> c_lock(ready_list_mtx);
            while(should_wait){
                double now = get_now_tp();
                double wait_min_duration = 0.0;
                if(closest_token_entity->expired_tp > now){
                    wait_min_duration = closest_token_entity->expired_tp - now;
                }
                time_point timeout_tp = tp_after_duration(wait_min_duration);
                LOG_DEBUG("Wait for next token request, duration to expired: {}", wait_min_duration);
                auto status = ready_list_cond.wait_until(c_lock, timeout_tp);
                if(status == std::cv_status::timeout){
                    should_wait = false;
                    sm_occupied -= clients_info[closest_token_entity->c_name]->get_sm_partition();
                    token_takers.erase(closest_token_entity);
                }else{
                    for(auto &item : ready_list){
                        if(release_token_taker(item.c_name) || clients_info[item.c_name]->get_sm_partition() + sm_occupied <= SM_GLOBAL_LIMIT){
                            should_wait = false;
                            break;
                        }
                    }
                }
            }
            c_lock.unlock();
            // ci_lock.unlock();
        }
    }
}


std::vector<sched_entity> FastScheduler::pick_next_runnable_entities(){
    while(running){
        double actual_time_window = time_window;
        double now_tp = get_now_tp();
        double win_start = now_tp - time_window;
        if( win_start < 0){
            actual_time_window = now_tp;
            win_start = 0;
        }
        auto out_of_window = [=](const sched_record &recd) -> bool { return recd.expired_tp < win_start;};
        clients_record.remove_if(out_of_window);
        std::map<std::string, double> usage;
        LOG_DEBUG("Remaining clients_record size: size = {}", clients_record.size());
        for(auto &item: clients_record){
            usage[item.c_name] += item.expired_tp - std::max(win_start, item.start_tp);
            LOG_DEBUG("The time usage by client = {} is: {}, timestamp = {}", item.c_name, usage[item.c_name], get_now_tp());
        }

        std::vector<scheduable_entity> scheduable_entities;
        double wait_time = time_window;
        std::unique_lock<std::mutex> r_lock(ready_list_mtx);
        for(auto it = ready_list.begin(); it != ready_list.end(); it++){
            std::string c_name = it->c_name;
            double limit, request, delta_limit, delta_req;
            request = clients_info[c_name]->get_min_quota() * actual_time_window;
            limit = clients_info[c_name]->get_max_quota() * actual_time_window;
            if(usage.find(c_name) == usage.end()) usage[c_name] = 0.0;
            delta_req = request - usage[c_name];
            delta_limit = limit - usage[c_name];
            
            if(delta_limit > 0){
                scheduable_entities.push_back({usage[c_name], delta_req, delta_limit, it});
            }else{
                wait_time = std::min(wait_time, -delta_limit);
                // wait_time = std::min(wait_time, time_window - usage[c_name]);
                // wait_time = std::min(wait_time, clients_record.begin()->expired_tp - win_start);
            }
        }
        r_lock.unlock();
        // All clients have used up their limit resources, 
        // wait for minimum "overused" time (delta_limit) of all clients;
        if(scheduable_entities.size() == 0){
            auto wait_tp = tp_after_duration(wait_time);
            r_lock.lock();
            LOG_DEBUG("scheduable_entities is empty, start wait_until. wait for {}ms, timestamp = {}", wait_time, get_now_tp());
            ready_list_cond.wait_until(r_lock, wait_tp);
            r_lock.unlock();
            continue;
        }
        LOG_DEBUG("scheduable_entities is not empty, size = {}", scheduable_entities.size());
        std::sort(scheduable_entities.begin(), scheduable_entities.end(), schedule_policy_priority);
        LOG_DEBUG("sorted, scheduable_entities first = {}", scheduable_entities.begin()->entity_iter->c_name);
        // totoal sm occupancy evaluation and selection
        std::vector<sched_entity> runnable_entities;
        for(auto it = scheduable_entities.begin(); it != scheduable_entities.end(); it++){
            std::string name = it->entity_iter->c_name;
            size_t sm_partition = clients_info[name]->get_sm_partition();
            if(sm_occupied + sm_partition <= SM_GLOBAL_LIMIT){
                runnable_entities.push_back(*(it->entity_iter));
                r_lock.lock();
                ready_list.erase(it->entity_iter);
                r_lock.unlock();
            }
        }

        // all schedulable token clients cannot adapt to be below SM_GLOBAL_LIMIT
        if(runnable_entities.size() == 0){
            update_tokens();
            double wait_itv = closest_token_entity->expired_tp - get_now_tp();
            time_point closest_expired_tp = steady_clock::now() + microseconds((int)(wait_itv * 1e3));
            r_lock.lock();
            LOG_DEBUG("runnable_entities is empty, start wait_until");
            ready_list_cond.wait_until(r_lock, closest_expired_tp);
            r_lock.unlock();
            continue;
        }
        LOG_DEBUG("runnable_entities is not empty, size = {}", runnable_entities.size());
        return runnable_entities;
    }
    return std::vector<sched_entity>{};
}

// The priority of clients to get tokens in the schduling policy
bool FastScheduler::schedule_policy_priority(const scheduable_entity& en_a, const scheduable_entity &en_b){
    if(en_a.delta_req > 0 && en_b.delta_req > 0){
        return en_a.delta_req / (en_a.delta_req + en_a.used) >  en_b.delta_req / (en_b.delta_req + en_b.used);
    }
    if(en_a.delta_req > 0 && en_b.delta_req <= 0) return true;
    if(en_a.delta_req <= 0 && en_b.delta_req > 0) return false;
    return en_a.used < en_b.used;
}


void FastScheduler::record(const std::string &cn, const double &quota){
    sched_record sc;
    sc.c_name = cn;
    sc.start_tp = get_now_tp();
    sc.expired_tp = sc.start_tp + quota;
    clients_record.push_back(sc);
}

bool FastScheduler::release_token_taker(const std::string &cn){
    auto iter = token_takers.begin();
    while(iter != token_takers.end()){
        if(cn == iter->c_name){
            sm_occupied -= clients_info[cn]->get_sm_partition();
            token_takers.erase(iter);
            return true;
        }
        iter++;
    }
    return false;
}


bool FastScheduler::update_tokens(){
    bool should_wait = true;
    double now = get_now_tp();
    if(token_takers.size() == 0){
        should_wait = false;
    }else{
        auto iter = token_takers.begin();
        while(iter != token_takers.end()){
            if(iter->expired_tp <= now){
                sm_occupied -= clients_info[iter->c_name]->get_sm_partition();
                token_takers.erase(iter++);
                should_wait = false;
            }else{
                iter++;
            }
        }
        closest_token_entity = std::min_element(token_takers.begin(), token_takers.end(), 
                        [](const sched_entity& pairA, const sched_entity& pairB) -> bool{return pairA.expired_tp < pairB.expired_tp;});
    }
    return should_wait;
}


 time_point FastScheduler::tp_after_duration(const double &itv_ms){
    return (steady_clock::now() + microseconds((int)(itv_ms * 1e3)));
}



void FastScheduler::update_actual_token_end_time(std::string c_n, double over_used){
    double now_tp = get_now_tp();
    for(auto it=clients_record.rbegin(); it!=clients_record.rend(); it++){
        if(it->c_name == c_n){
            it->expired_tp = std::min(now_tp, it->expired_tp + over_used);
            LOG_DEBUG("Clients record found: expired_tp = {}, start_tp = {}, duration = {}", it->expired_tp, it->start_tp, (it->expired_tp - it->start_tp));
            break;
        }
    }
}

double FastScheduler::get_next_quota(const GPUClientInfo & client){
    return client.get_min_quota() * time_window;
}


// Process exit gracefully when the process meet the error;
void FastScheduler::grace_exit(){
    running.store(false);
    exit_handle();
    usleep(50000);
    exit(-1);
}


void FastScheduler::exit_handle(){
    for(auto it=tids.begin(); it!=tids.end(); it++){
        pthread_cancel(it->second);
    }
    std::map<std::string, GPUClientInfo*>::iterator iter;
    for(iter = clients_info.begin(); iter != clients_info.end(); iter++){
        delete iter->second;
        clients_info.erase(iter);
    }
}


//sigint stop the program 
void sigintHandler(int signum){
    LOG_DEBUG("Interrupt signal ( {} ) received. STOP the program.", signum);
    running.store(false);
    usleep(500);
    fast_scheduler.exit_handle();
    exit(-1);
}
