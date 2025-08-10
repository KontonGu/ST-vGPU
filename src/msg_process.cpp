/*
 * Modified on Tue May 21 2024
 *
 * Author: KontonGu
 * Copyright (c) 2024 TUM - CAPS Cloud
 * Licensed under the Apache License, Version 2.0 (the "License")
 */

#include "msg_process.h"
// #include "log_debug.h"

#include <unistd.h>
#include <climits>
#include <stdarg.h>
#include <cstring>
#include <iostream>

template<typename T>
void writeToBuffer(MsgBuffer& buffer, const T& value) {
    buffer.write(value);
}

template<>
void writeToBuffer<std::string>(MsgBuffer& buffer, const std::string& str) {
    buffer.write(str);
}

template<typename T>
void readFromBuffer(const MsgBuffer& buffer, size_t& offset, T& data) {
    data = buffer.read<T>(offset);
}

template<>
void readFromBuffer<std::string>(const MsgBuffer& buffer, size_t& offset, std::string & data) {
    data = buffer.readString(offset);
}

/*
As for the request data format:, 
REQ_TOKEN: (<client_name>(string), <request_type>(RequestType), <request_id>(ReqID), <overuse>(double), <burst>(double));
UPDATE_MEM: (<client_name>(string), <request_type>(RequestType), <request_id>(ReqID), <memory_size>(MemByte), <memory_operation>(int));
*/ 
ReqID new_request(MsgBuffer& buffer, RequestType req_t, ...){
    static std::string client_name;
    static ReqID req_id = 0;
    va_list args;

    if(client_name.empty()){
        char * c_name = getenv("GPU_CLIENT_NAME");
        if(c_name == nullptr){
            char host_name[HOST_NAME_MAX];
            gethostname(host_name, HOST_NAME_MAX);
            // LOG_WARNING("The client might be the host. Set to default host name.");
            client_name = std::string(host_name);
        }else{
            client_name = std::string(c_name);
        }
    }
    writeToBuffer(buffer, client_name);
    writeToBuffer(buffer, req_id);
    writeToBuffer(buffer, req_t);

    if(req_t == REQ_TOKEN){
        va_start(args, req_t);
        writeToBuffer(buffer, va_arg(args, double));   // overuse time
        writeToBuffer(buffer, va_arg(args, double));   // maximum burst time
    }else if (req_t == MEM_UPDATE){
        va_start(args, req_t);
        writeToBuffer(buffer, va_arg(args, MemByte));   // memory in bytes
        writeToBuffer(buffer, va_arg(args, int));     // the type of mem update, allocate = 1 or free = 0;
    }
    va_end(args);
    return req_id++;
}


size_t parse_meta(const MsgBuffer& buffer, std::string &client_name, ReqID &req_id, RequestType &req_t){
    size_t offset = 0;
    readFromBuffer(buffer, offset, client_name);
    readFromBuffer(buffer, offset, req_id);
    readFromBuffer(buffer, offset, req_t);
    return offset;
}

size_t parse_meta_discard(const MsgBuffer& buffer){
    size_t offset = 0;
    std::string client_name;
    ReqID req_id; 
    RequestType req_t;
    readFromBuffer(buffer, offset, client_name);
    readFromBuffer(buffer, offset, req_id);
    readFromBuffer(buffer, offset, req_t);
    return offset;
}

// Example MEM_UPDATE: parse_request_data(buf, offset, req_t, &recv_mem_bytes, &recv_allocate)
void parse_request_data(const MsgBuffer& buffer, size_t &offset, const RequestType &req_t, ...){
    va_list args;
    
    if(req_t == REQ_TOKEN){
        va_start(args, req_t);
        double* overuse = va_arg(args, double *);
        double* burst = va_arg(args, double *);
        readFromBuffer(buffer, offset, *overuse);
        readFromBuffer(buffer, offset, *burst);
        
    }else if (req_t == MEM_UPDATE){
        va_start(args, req_t);
        MemByte* mem_bytes = va_arg(args, MemByte*);
        int* allocate = va_arg(args, int *);
        readFromBuffer(buffer, offset, *mem_bytes);
        readFromBuffer(buffer, offset, *allocate);
    }
    va_end(args);
}


void new_response(MsgBuffer &buffer, const std::string &req_name, const ReqID &req_id, const RequestType &req_t, ...){
    // std::string resp_name = req_name;
    va_list args;
    writeToBuffer(buffer, req_name);
    writeToBuffer(buffer, req_id);
    writeToBuffer(buffer, req_t);
    if(req_t == REQ_TOKEN){
        va_start(args, req_t);
        writeToBuffer(buffer, va_arg(args, double)); // remaining token time;
    }else if(req_t == MEM_UPDATE){
        va_start(args, req_t);
        writeToBuffer(buffer, va_arg(args, int));  // indicate if the memory update is successful or not. 
        writeToBuffer(buffer, va_arg(args, MemByte));  // return the remaining memory;
    }else if(req_t == MEM_LIMIT){
        va_start(args, req_t);
        writeToBuffer(buffer, va_arg(args, MemByte));  // memory used in bytes;
        writeToBuffer(buffer, va_arg(args, MemByte));  // memory limited in bytes;
    }
    va_end(args);
}


// Example of MEM_UPDATE: parse_response_data(buf, offset, req_t, &recv_mem_bytes, &recv_allocate)
void parse_response_data(const MsgBuffer& buffer,  size_t &offset, const RequestType &req_t, ...){
    va_list args;
    if(req_t == REQ_TOKEN){
        va_start(args, req_t);
        double* remain_token = va_arg(args, double *);
        readFromBuffer(buffer, offset, *remain_token);
    }else if (req_t == MEM_UPDATE){
        va_start(args, req_t);
        int * is_success = va_arg(args, int *);
        MemByte* mem_remain = va_arg(args, MemByte*);
        readFromBuffer(buffer, offset, *is_success);
        readFromBuffer(buffer, offset, *mem_remain);
    }else if (req_t == MEM_LIMIT){
        va_start(args, req_t);
        MemByte* mem_used = va_arg(args, MemByte*);
        MemByte* mem_limit = va_arg(args, MemByte*);
        readFromBuffer(buffer, offset, *mem_used);
        readFromBuffer(buffer, offset, *mem_limit);
    }
    va_end(args);
}


int multiple_try(std::function<int()> func, int max_try,  int it_val){
    int res;
    for(int i=0; i<max_try; i++){
        res = func();
        if(res == 0) break;
        else if(res == -1) res = errno;
        // LOG_ERROR("Try the {} times: {}", i, strerror(res));
        if(it_val > 0) sleep(it_val);
    }
    return res;
}