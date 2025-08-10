#
 # Modified on Sun May 19 2024
 #
 # Author: KontonGu
 # Copyright (c) 2024 TUM - CAPS Cloud
 # Licensed under the Apache License, Version 2.0 (the "License")
 #


# path to CUDA installation
CUDA_PATH ?= /usr/local/cuda-12.2
PWD := $(shell pwd)
PREFIX ?= $(PWD)/..


SMS ?=  60 61 70 75 80
GENCODE_FLAGS += $(foreach sm,$(SMS),-gencode arch=compute_$(sm),code=sm_$(sm))

CXX ?= g++ 
NVCC ?= $(CUDA_PATH)/bin/nvcc -ccbin $(CXX)

CUDA_LDFLAGS += -lcuda -L$(CUDA_PATH)/lib64 -L$(CUDA_PATH)/lib64/stubs 
LDFLAGS += -ldl -lrt 
TH_FLAGS += -lpthread
CUDA_INCLUDE += -I$(CUDA_PATH)/include -I./
UTIL_INCLUDE += -I../util/ -I./

# ENABLE_DEBUG = -DENABLE_DEBUG
ENABLE_DEBUG = -DDISABLE_DEBUG

BUILD_FOLDER += ../build

CXXFLAGS += -std=c++11 -fPIC 

CXX_OPT = -O3

ifeq ($(DEBUG),1)
# CXXFLAGS += -g -D_DEBUG -Wall 
# ENABLE_DEBUG = -DENABLE_DEBUG
CXX_OPT = -O2
else
CXXFLAGS += -O2
CXX_OPT = -O2
endif

all: libfast.so.1 gpu_client fast_scheduler
	cp *.o ${BUILD_FOLDER}

cuda_hook.o: cuda_hook.cpp cuda_hook.h log_debug.h hook_util.h
	$(NVCC) -m64 --compiler-options "$(CXXFLAGS)" $(GENCODE_FLAGS) -o $@ -c $< ${ENABLE_DEBUG} 

libfast.so.1: cuda_hook.o msg_process.o hook_util.o
	$(NVCC) -shared -m64 $(GENCODE_FLAGS) -o $@ $^ $(CUDA_LDFLAGS) $(LDFLAGS) ${TH_FLAGS} ${CXX_OPT}
	mkdir -p $(PREFIX)/lib
	cp $@ $(PREFIX)/lib
	mv $@ ${BUILD_FOLDER}/
# mv *.o ${BUILD_FOLDER}

msg_process.o: msg_process.cpp msg_process.h 
	$(CXX) -c $< -g -o $@ -std=c++11

hook_util.o: hook_util.cpp hook_util.h 
	$(CXX) -c $< -g -o $@ -std=c++11 ${CUDA_INCLUDE}

gpu_client.o: gpu_client.cpp gpu_client.h msg_process.h log_debug.h
	$(CXX) -c $< -o $@ -std=c++11 ${TH_FLAGS} ${ENABLE_DEBUG} ${CXX_OPT}

gpu_client: gpu_client.o msg_process.o
	$(CXX) -o $@ $^ ${TH_FLAGS} ${CXX_OPT}
	mv $@ ${BUILD_FOLDER}/

clean:
	rm  -f *.o
	rm  -f ${BUILD_FOLDER}/*.o
	rm  -f ${BUILD_FOLDER}/libfast.so.1 ${BUILD_FOLDER}/gpu_client ${BUILD_FOLDER}/fast_scheduler

src_clean:
	rm  -f *.o

fast_scheduler.o: fast_scheduler.cpp fast_scheduler.h log_debug.h msg_process.h
	$(CXX) -c $< -o $@ ${UTIL_INCLUDE} ${TH_FLAGS} ${ENABLE_DEBUG} ${CXX_OPT}

fast_scheduler: fast_scheduler.o msg_process.o
	$(CXX) -o $@ $^ ${TH_FLAGS} ${CXX_OPT}
	mv $@ ${BUILD_FOLDER}/
	
# libfast.so.2: cuda_hook.cpp msg_process.cpp hook_util.cpp 
# 	$(NVCC) -shared -m64 --compiler-options "$(CXXFLAGS)" $(GENCODE_FLAGS) ${CUDA_INCLUDE} $(LDFLAGS) ${TH_FLAGS} -o $@ $^

