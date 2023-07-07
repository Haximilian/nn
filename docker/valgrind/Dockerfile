FROM alpine:latest
RUN apk add --no-cache make
RUN apk add --no-cache g++
RUN apk add --no-cache valgrind
RUN apk add --no-cache perf
COPY . /source
WORKDIR /source
RUN make
ENTRYPOINT ["/bin/sh"]
