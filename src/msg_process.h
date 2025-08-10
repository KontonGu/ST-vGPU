/*
 * Modified on Tue May 21 2024
 *
 * Author: KontonGu
 * Copyright (c) 2024 TUM - CAPS Cloud
 * Licensed under the Apache License, Version 2.0 (the "License")
 */

#ifndef MSG_PROCESS_H
#define MSG_PROCESS_H

#include <iostream>
#include <vector>
#include <cstring>
#include <string>
#include <functional>


enum RequestType {REQ_TOKEN, MEM_LIMIT, MEM_UPDATE};
#define ReqID int32_t
#define MemByte size_t

#define MSGBUFFER_MAX_LEN 1024
#define SOCKET_MAX_TRY 10
#define SOCKET_RETRY_INTV 5 

class MsgBuffer {
private:
    char data[MSGBUFFER_MAX_LEN];
    size_t offset = 0;
public:
    MsgBuffer() = default;
    
    MsgBuffer(const char * input_data, size_t len){
        strncpy(data+offset, input_data, len);
        offset += len;
    }

    size_t copy2MsgBuffer(const char * input_data, size_t len){
        offset = 0;
        memcpy(data+offset, input_data, len);
        offset += len;
        return offset;
    }

    template<typename T>
    size_t write(const T& value) {
        memcpy(data+offset, &value, sizeof(T));
        return offset = offset + sizeof(T);
    }

    // write string
    size_t write(const std::string& str) {
        size_t len = str.size();
        write(len);
        memcpy(data+offset, str.c_str(), len);
        return offset += len;
    }

    template<typename T>
    T read(size_t& offset) const {
        T value;
        std::memcpy(&value, data+offset, sizeof(T));
        offset += sizeof(T);
        return value;
    }

    // read string
    std::string readString(size_t& offset) const {
        size_t len = read<size_t>(offset);
        std::string str(data+offset, len); 
        offset += len;
        return str;
    }

    size_t size() const {
        return offset;
    }

    const char* getData() const {
        return data;
    }    

    size_t copyData(char *val){
        std::memcpy(val, data, offset);
        return offset;
    }

    void set_offset(const size_t &i_offset){
        offset = i_offset;
    }
};

template<typename T>
void writeToBuffer(MsgBuffer& buffer, const T& value);

template<>
void writeToBuffer<std::string>(MsgBuffer& buffer, const std::string& str);


template<typename T>
void readFromBuffer(const MsgBuffer& buffer, size_t& offset, T& data);

template<>
void readFromBuffer(const MsgBuffer& buffer, size_t& offset, std::string & data);


ReqID new_request(MsgBuffer& buffer, RequestType req_t, ...);
void parse_request_data(const MsgBuffer& buffer, size_t &offset, const RequestType &req_t, ...);

void new_response(MsgBuffer &buffer, const std::string &req_name, const ReqID &req_id, const RequestType &req_t, ...);
void parse_response_data(const MsgBuffer& buffer,  size_t &offset, const RequestType &req_t, ...);

size_t parse_meta(const MsgBuffer& buffer, std::string &client_name, ReqID &req_id, RequestType &req_t);
size_t parse_meta_discard(const MsgBuffer& buffer);

int multiple_try(std::function<int()> func, int max_try,  int it_val=0);

#endif