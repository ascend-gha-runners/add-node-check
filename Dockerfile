ARG CANN_VERSION=8.2.rc1
ARG DEVICE_TYPE=910b  # Default, overridden by workflow
ARG OS=ubuntu22.04
ARG PYTHON_VERSION=py3.11
ARG REGISTRY=swr.cn-southwest-2.myhuaweicloud.com/base_image/ascend-ci

FROM $REGISTRY/cann:$CANN_VERSION-$DEVICE_TYPE-$OS-$PYTHON_VERSION

# Define environments
ARG TARGETARCH # auto-set by Buildx (amd64/arm64)
ARG PIP_INDEX_URL="https://pypi.org/simple/"
ARG APTMIRROR="repo.huaweicloud.com"
ARG PYTORCH_VERSION=2.6.0
ARG TORCHVISION_VERSION=0.21.0
ARG VLLM_TAG=v0.8.5
ARG SGLANG_TAG=main
ARG ASCEND_CANN_PATH=/usr/local/Ascend/ascend-toolkit
ARG SGLANG_KERNEL_NPU_TAG=main

# Set environment variables according to architecture
ARG MEMFABRIC_URL_AMD64="https://sglang-ascend.obs.cn-east-3.myhuaweicloud.com/sglang/mf_adapter-1.0.0-cp311-cp311-linux_x86_64.whl"
ARG PTA_URL_AMD64="https://gitcode.com/Ascend/pytorch/releases/download/v7.1.0.2-pytorch2.6.0/torch_npu-2.6.0.post2-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
ARG TRITON_ASCEND_URL_AMD64="https://sglang-ascend.obs.cn-east-3.myhuaweicloud.com/sglang/triton_ascend-3.2.0.dev20250815-cp311-cp311-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"

ARG MEMFABRIC_URL_ARM64="https://sglang-ascend.obs.cn-east-3.myhuaweicloud.com/sglang/mf_adapter-1.0.0-cp311-cp311-linux_aarch64.whl"
ARG PTA_URL_ARM64="https://gitee.com/ascend/pytorch/releases/download/v7.1.0.1-pytorch2.6.0/torch_npu-2.6.0.post1-cp311-cp311-manylinux_2_28_aarch64.whl"
ARG TRITON_ASCEND_URL_ARM64="https://sglang-ascend.obs.cn-east-3.myhuaweicloud.com/sglang/triton_ascend-3.2.0.dev20250729-cp311-cp311-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl"

RUN if [ "$TARGETARCH" = "amd64" ]; then \
      echo "Using x86_64 dependencies"; \
      echo "MEMFABRIC_URL=$MEMFABRIC_URL_AMD64" >> /etc/environment_new; \
      echo "PTA_URL=$PTA_URL_AMD64" >> /etc/environment_new; \
      echo "TRITON_ASCEND_URL=$TRITON_ASCEND_URL_AMD64" >> /etc/environment_new; \
    elif [ "$TARGETARCH" = "arm64" ]; then \
      echo "Using aarch64 dependencies"; \
      echo "MEMFABRIC_URL=$MEMFABRIC_URL_ARM64" >> /etc/environment_new; \
      echo "PTA_URL=$PTA_URL_ARM64" >> /etc/environment_new; \
      echo "TRITON_ASCEND_URL=$TRITON_ASCEND_URL_ARM64" >> /etc/environment_new; \
    else \
      echo "Unsupported TARGETARCH: $TARGETARCH"; exit 1; \
    fi

WORKDIR /workspace
ENV DEBIAN_FRONTEND=noninteractive

# Configure caching URLs for apt and pip
RUN CACHING_URL="cache-service.nginx-pypi-cache.svc.cluster.local" && \
    # Configure apt sources to use cache \
    sed -Ei "s@(ports|archive).ubuntu.com@${CACHING_URL}:8081@g" /etc/apt/sources.list && \
    # Configure pip to use cache \
    pip config set global.index-url http://${CACHING_URL}/pypi/simple && \
    pip config set global.trusted-host ${CACHING_URL}


# Install development tools and utilities
RUN apt-get update -y && apt upgrade -y && apt-get install -y \
    build-essential \
    cmake \
    vim \
    wget \
    curl \
    net-tools \
    zlib1g-dev \
    lld \
    clang \
    locales \
    ccache \
    openssl \
    libssl-dev \
    pkg-config \
    ca-certificates \
    protobuf-compiler \
    && rm -rf /var/cache/apt/* \
    && rm -rf /var/lib/apt/lists/* \
    && update-ca-certificates \
    && locale-gen en_US.UTF-8

ENV LANG=en_US.UTF-8 \
    LANGUAGE=en_US:en \
    LC_ALL=en_US.UTF-8 \
    PATH="/root/.cargo/bin:${PATH}"

RUN . /etc/environment_new && \
    pip install $MEMFABRIC_URL --no-cache-dir

RUN pip install setuptools-rust wheel build --no-cache-dir

# install rustup from rustup.rs
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y \
    && rustc --version && cargo --version && protoc --version

# Install vLLM
RUN git clone --depth 1 https://github.com/vllm-project/vllm.git --branch $VLLM_TAG && \
    (cd vllm && VLLM_TARGET_DEVICE="empty" pip install -v . --no-cache-dir) && rm -rf vllm

# TODO: install from pypi released triton-ascend
RUN . /etc/environment_new && \
    pip install torch==$PYTORCH_VERSION torchvision==$TORCHVISION_VERSION --index-url https://download.pytorch.org/whl/cpu --no-cache-dir \
    && wget ${PTA_URL} && pip install $(basename ${PTA_URL}) --no-cache-dir \
    && python3 -m pip install --no-cache-dir attrs==24.2.0 numpy==1.26.4 scipy==1.13.1 decorator==5.1.1 psutil==6.0.0 pytest==8.3.2 pytest-xdist==3.6.1 pyyaml pybind11 \
    && pip install ${TRITON_ASCEND_URL} --no-cache-dir

# Install SGLang
RUN mkdir /workspace/sglang
COPY . /workspace/sglang
RUN (cd sglang/python && pip install -v .[srt_npu] --no-cache-dir) \
    && (cd sglang/sgl-router && python -m build && pip install --force-reinstall dist/*.whl) \
    && rm -rf sglang

# Install Deep-ep
# pin wheel to 0.45.1 ref: https://github.com/pypa/wheel/issues/662
RUN . /etc/environment_new && \
    pip install wheel==0.45.1 && git clone --branch $SGLANG_KERNEL_NPU_TAG https://github.com/sgl-project/sgl-kernel-npu.git \
    && export LD_LIBRARY_PATH=${ASCEND_CANN_PATH}/latest/runtime/lib64/stub:$LD_LIBRARY_PATH \
    && export CPLUS_INCLUDE_PATH=${ASCEND_HOME_PATH}/x86_64-linux/include/experiment/platform/ \
    && source ${ASCEND_CANN_PATH}/set_env.sh && \
    cd sgl-kernel-npu && \
    bash build.sh \
    && pip install output/deep_ep*.whl output/sgl_kernel_npu*.whl --no-cache-dir \
    && cd .. && rm -rf sgl-kernel-npu \
    && cd "$(pip show deep-ep | awk '/^Location:/ {print $2}')" && ln -s deep_ep/deep_ep_cpp*.so

CMD ["/bin/bash"]