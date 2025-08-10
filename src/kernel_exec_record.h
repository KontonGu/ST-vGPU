/*
 * Modified on Thu May 23 2024
 *
 * Author: KontonGu
 * Copyright (c) 2024 TUM - CAPS Cloud
 * Licensed under the Apache License, Version 2.0 (the "License")
 * 
 * Note: The principle of the kernel execution time record and prediction is partly from the paper:
 * https://ieeexplore.ieee.org/abstract/document/9566822
 */

#ifndef KERNEL_EXEC_RECORD_H
#define KERNEL_EXEC_RECORD_H

#include <mutex>
#include <deque>
#include <chrono>

using std::chrono::steady_clock;
using std::chrono::microseconds;
using std::chrono::milliseconds;
using std::chrono::duration_cast;
typedef std::chrono::time_point<std::chrono::steady_clock> time_point;

#define DEFAULT_MIN_IDLE_ITV 2.0                 
#define DEFAULT_VALID_ITV 3000                    // in miliseconds


class MaxRecorder {
public:
    // MaxRecorder(): valid_itv(DEFAULT_VALID_ITV) {}
    MaxRecorder(const uint32_t &itv){
        valid_itv = itv;
    } 
    void set_valid_window(uint32_t itv);
    void add_record(const time_point &tp, const double &du);
    void update_window(const time_point &tp);
    double get_max();
private:
    uint32_t valid_itv;  // in milisecond
    std::deque<std::pair<time_point, double>> record;
};

#endif


class BurstRecorder{
public:
    // itv is the minimum interval between two kernels to be regarded as a burst;  
    BurstRecorder(const double &itv); 
    void record_start();
    void record_stop();
    double get_current_kernel_max();
    double get_current_burst_max();
    void reset();

private:
    MaxRecorder kernel_record, burst_record;
    std::mutex r_mtx;
    double min_burst_idle_itv;
    time_point kernel_begin, burst_begin;
    time_point burst_end;
    bool is_kernel_recording;
};