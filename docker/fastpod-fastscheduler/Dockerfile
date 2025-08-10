FROM nvidia/cuda:12.2.0-devel-ubuntu22.04 AS build

WORKDIR /tmp/fastpod

RUN mkdir build src util
COPY src /tmp/fastpod/src
COPY util /tmp/fastpod/util
WORKDIR /tmp/fastpod/src
RUN make all

# FROM python:3.8.1-buster
FROM nvidia/cuda:12.2.2-base-ubuntu22.04
# FROM ubuntu/python:3.10-22.04

RUN apt-get update && apt install -y python3.10 python3-pip && apt-get clean
RUN pip3 install inotify


ENV NVIDIA_VISIBLE_DEVICES      all
ENV NVIDIA_DRIVER_CAPABILITIES  utility

COPY docker/fastpod-fastscheduler/launch_schedulers.sh /launch_schedulers.sh
RUN chmod +x /launch_schedulers.sh
COPY docker/fastpod-fastscheduler/scheduler_instance.py /scheduler_instance.py
COPY --from=build /tmp/fastpod/build/fast_scheduler /fast_scheduler
COPY --from=build /tmp/fastpod/build/gpu_client /gpu_client

CMD [ "/bin/bash", "/launch_schedulers.sh", "/fastpod/scheduler/config", "/fastpod/scheduler/gpu_clients", "/fastpod/scheduler/log"]

