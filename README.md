## ST-vGPU: Fine-grained GPU Sharing and Isolation Sub-System

![](./figures/ST-vGPU.png)

`ST-vGPU` is an independent spatio-temporal GPU resource sharing and isolation architecture implemented entirely in the user space of the operating system.

`ST-vGPU` consists of three key components, namely `GPU Servers`, `GPU Clients (vGPUs)`, and `GPU Processes`. The `GPU Server` is the control component that manages all physical GPUs in a node, with each GPU Server bound to a specific physical GPU. The GPU Server is responsible for creating, destroying, scheduling, and allocating resources for vGPUs in the GPU client layer. In ST-vGPU, a GPU Server can dynamically instantiate multiple vGPUs at runtime and on demand according to varying spatial and temporal GPU resource requirements. A `vGPU` is a logical partition of a physical GPU and represents the allocation and isolation of spatio-temporal GPU resources when multiple applications share the GPU. The vGPU requests GPU resources in real time by sending the token to the GPU Server. Meanwhile, the GPU Server controls and coordinates resource usage among multiple vGPUs by allocating tokens. A `token` is a GPU time slice, serving as the time unit that a vGPU or application can execute on the GPU. When a vGPU or application exhausts its allocated time slice, it can apply for new execution time by continuously sending token requests to its upper layers. At the `GPU process` level, a process acquires GPU resources by sending token requests or memory requests to its associated vGPU. In most cases, a vGPU is bound one-to-one with a GPU process to ensure inter-application isolation. Moreover, when multiple processes do not require isolation and need to share the same vGPU, a multiple-to-one mapping can also be flexibly adopted.


## Citation
If you use FaST-GShare for your research, please cite our paper [paper](https://dl.acm.org/doi/abs/10.1145/3605573.3605638):
```
@inproceedings{gu2023fast,
  title={FaST-GShare: Enabling Efficient Spatio-Temporal GPU Sharing in Serverless Computing for Deep Learning Inference},
  author={Gu, Jianfeng and Zhu, Yichao and Wang, Puxuan and Chadha, Mohak and Gerndt, Michael},
  booktitle={Proceedings of the 52nd International Conference on Parallel Processing},
  pages={635--644},
  year={2023}
}
```


## License
Copyright 2024 FaST-GShare Authors, KontonGu (**Jianfeng Gu**), et. al.
@Techinical University of Munich, **CAPS Cloud Team**

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.