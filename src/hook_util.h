/*
 * Modified on Mon May 20 2024
 *
 * Author: KontonGu
 * Copyright (c) 2024 TUM - CAPS Cloud
 * Licensed under the Apache License, Version 2.0 (the "License")
 */

#ifndef CUDA_HOOK_UTIL_H
#define CUDA_HOOK_UTIL_H

#include <cuda.h>

inline size_t CUarray_format_to_byte(CUarray_format Format) {
  switch (Format) {
    case CU_AD_FORMAT_UNSIGNED_INT8:
    case CU_AD_FORMAT_SIGNED_INT8:
      return 1;
    case CU_AD_FORMAT_UNSIGNED_INT16:
    case CU_AD_FORMAT_SIGNED_INT16:
    case CU_AD_FORMAT_HALF:
      return 2;
    case CU_AD_FORMAT_UNSIGNED_INT32:
    case CU_AD_FORMAT_SIGNED_INT32:
    case CU_AD_FORMAT_FLOAT:
      return 4;
  }
}

void sigintHandler(int signum);
void sigcontHandler(int signum);

#endif



