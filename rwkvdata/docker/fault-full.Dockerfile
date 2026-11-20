# Fault-validation image with every task toolchain (rwkvdata faults --image), matching harbor_export's task images:
# Go 1.22, Rust 1.75, JDK 17 and the pinned web packages in /opt/web. Build on the runner host after fault-base:
#   docker build -t rwkv-fault-full:latest -f rwkvdata/docker/fault-full.Dockerfile rwkvdata/docker
FROM rwkv-pilot-base-test:latest
ARG NPM_REGISTRY=https://registry.npmmirror.com
RUN unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY && apt-get -o Acquire::http::Proxy=false update \
    && apt-get -o Acquire::http::Proxy=false install -y --no-install-recommends golang-go rustc cargo openjdk-17-jdk-headless \
    && rm -rf /var/lib/apt/lists/*
RUN unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY && mkdir -p /opt/web && cd /opt/web \
    && echo '{"private":true}' > package.json && npm install --no-audit --no-fund --save-exact --ignore-scripts --registry "$NPM_REGISTRY" \
      @types/node@22.9.0 @types/react@18.3.12 @types/react-dom@18.3.1 jsdom@25.0.1 react@18.3.1 react-dom@18.3.1 tsx@4.19.2 typescript@5.6.3
