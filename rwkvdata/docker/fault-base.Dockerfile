# Toolchain image for offline fault validation (rwkvdata faults --image). Build on the runner host:
#   docker build -t rwkv-pilot-base-test:latest -f rwkvdata/docker/fault-base.Dockerfile rwkvdata/docker
FROM ubuntu:24.04
# Hosts may inject a client-side proxy into builds; build and run without it.
ENV DEBIAN_FRONTEND=noninteractive LANG=C.UTF-8 \
    http_proxy= https_proxy= HTTP_PROXY= HTTPS_PROXY= no_proxy= NO_PROXY=
ARG APT_MIRROR=http://cn.archive.ubuntu.com
RUN unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY \
    && sed -i "s#http://archive.ubuntu.com#$APT_MIRROR#g" /etc/apt/sources.list.d/ubuntu.sources \
    && apt-get -o Acquire::http::Proxy=false update \
    && apt-get -o Acquire::http::Proxy=false install -y --no-install-recommends \
      python3 python3-pytest build-essential make sqlite3 git curl ca-certificates xz-utils jq \
    && rm -rf /var/lib/apt/lists/*
RUN unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY && curl -fsSL https://nodejs.org/dist/v22.22.1/node-v22.22.1-linux-x64.tar.xz \
      | tar -xJ -C /usr/local --strip-components=1
WORKDIR /app

