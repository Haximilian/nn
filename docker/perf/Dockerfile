FROM alpine:latest
VOLUME ["/mnt"]
RUN apk add --no-cache make
RUN apk add --no-cache g++
RUN apk add --no-cache perf
COPY . /source
WORKDIR /source
RUN make
CMD echo 0 > /proc/sys/kernel/kptr_restrict \
    && perf record -a -F 999 -g --output=perf.data ./a.out \
    && perf script -i perf.data > profile.linux-perf.txt \
    && mv profile.linux-perf.txt /mnt