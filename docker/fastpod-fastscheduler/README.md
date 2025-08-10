## build the docker image
```
docker build  -t <your_docker_repository_registry>/fastpod-fastscheduler:release -f Dockerfile ../../
```
For example:
```
docker build  -t docker.io/kontonpuku666/fastpod-fastscheduler:release -f Dockerfile ../../
```
```
docker push  <your_docker_repository_registry>/fastpod-fastscheduler:release
```