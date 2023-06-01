FROM alpine:3.14
RUN apk add --no-cache valgrind
RUN apk add --no-cache make
RUN apk add --no-cache g++
COPY . /source
WORKDIR /source
RUN make
ENTRYPOINT ["/bin/sh"]
