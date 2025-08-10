.PHONY build_and_upload:
build_and_upload_release:
	docker build  -t docker.io/kontonpuku666/fastpod-fastscheduler:release -f Dockerfile ../../
	docker push docker.io/kontonpuku666/fastpod-fastscheduler:release

build_and_upload_debug:
	docker build  -t docker.io/kontonpuku666/fastpod-fastscheduler:debug -f Dockerfile ../../
	docker push docker.io/kontonpuku666/fastpod-fastscheduler:debug