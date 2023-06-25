make:
	g++ -std=c++11 main.cpp -O3
build-image:
	 docker build -t valgrind-image .
run-image:
	 docker run -it valgrind-image