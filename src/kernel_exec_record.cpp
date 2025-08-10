/*
 * Modified on Thu May 23 2024
 *
 * Author: KontonGu
 * Copyright (c) 2024 TUM - CAPS Cloud
 * Licensed under the Apache License, Version 2.0 (the "License")
 */


#include "kernel_exec_record.h"

void MaxRecorder::set_valid_window(uint32_t itv){
    valid_itv = itv;
}

void MaxRecorder::add_record(const time_point &tp, const double &du){
    while(!record.empty() && record.back().second < du) record.pop_back();
    record.push_back(std::make_pair(tp, du));
}

double MaxRecorder::get_max(){
    if(!record.empty()){
        return record.front().second;
    }else{
        return 0.0;
    }
}

void MaxRecorder::update_window(const time_point &tp){
    while(!record.empty() && duration_cast<milliseconds>(tp - record.front().first).count() > valid_itv) record.pop_front();
}


BurstRecorder::BurstRecorder(const double &itv):kernel_record(DEFAULT_VALID_ITV), 
                    burst_record(DEFAULT_VALID_ITV){
    min_burst_idle_itv = itv;
    kernel_begin = time_point::max();
    burst_begin = time_point::max();
    is_kernel_recording = false;
    burst_end = time_point::min();
}

void BurstRecorder::record_start(){
    double idle_itv;
    std::unique_lock<std::mutex> r_lock(r_mtx);
    if(!is_kernel_recording){
        
        kernel_begin = steady_clock::now();
        idle_itv = (duration_cast<microseconds>(kernel_begin - burst_end).count()) / 1e3;
        if(idle_itv > min_burst_idle_itv){
            burst_begin = kernel_begin;
            burst_end = time_point::min();
        }
        is_kernel_recording = true;    
    }
    r_lock.unlock();
}

void BurstRecorder::record_stop(){
    std::unique_lock<std::mutex> r_lock(r_mtx);
    if(is_kernel_recording){
        time_point p_now = steady_clock::now();
        double k_itv = (duration_cast<microseconds>(p_now - kernel_begin).count()) / 1e3;
        kernel_record.add_record(p_now, k_itv);
        burst_end = p_now;
        double b_itv = (duration_cast<microseconds>(burst_end - burst_begin).count()) / 1e3;
        burst_record.add_record(p_now, b_itv); 
    }
    is_kernel_recording = false;
    r_lock.unlock();
}


double BurstRecorder::get_current_kernel_max(){
    double k_max = 0;
    std::unique_lock<std::mutex> r_lock(r_mtx);
    kernel_record.update_window(steady_clock::now());
    k_max = kernel_record.get_max();
    r_lock.unlock();
    return k_max;
}

double BurstRecorder::get_current_burst_max(){
    double b_max = 0;
    std::unique_lock<std::mutex> r_lock(r_mtx);
    burst_record.update_window(steady_clock::now());
    b_max = burst_record.get_max();
    r_lock.unlock();
    return b_max;
}

void BurstRecorder::reset(){
    kernel_begin = time_point::max();
    burst_begin = time_point::max();
    is_kernel_recording = false;
    burst_end = time_point::min();
}
