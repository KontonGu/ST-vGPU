FROM nvidia/cuda:12.2.0-devel-ubuntu22.04 AS build

WORKDIR /tmp/fastpod

RUN mkdir build src util
COPY src /tmp/fastpod/src
COPY util /tmp/fastpod/util
WORKDIR /tmp/fastpod/src
RUN make all

FROM alpine:3.9
# FROM busybox:1.36.1-glibc
COPY --from=build /tmp/fastpod/build/libfast.so.1 /libfast.so.1 
CMD ["cp", "/libfast.so.1 ", "/fastpod/library/libfast.so.1"]


