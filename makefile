make:
	g++ -std=c++11 main.cpp -O3
build-image-perf:
	docker build . --tag perf-image --file docker/perf/Dockerfile
run-image-perf:
	docker run --mount type=bind,source=.,target=/mnt --privileged --rm perf-image
build-image:
	 docker build -t perf-valgrind-image .
run-image:
	 docker run -it --privileged --pid=host --rm perf-valgrind-image
wasm:
	emcc -o main.html main.cpp --preload-file ./train.csv --preload-file ./out.csv --preload-file ./cross_entropy.csv -O3 --shell-file application/shell_minimal.html -sFORCE_FILESYSTEM