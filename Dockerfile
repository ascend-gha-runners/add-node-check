ARG CANN_VERSION=8.3.rc1
ARG OS=ubuntu22.04
ARG DEVICE_TYPE=910b
ARG PYTHON_VERSION=py3.11
ARG REGISTRY=swr.cn-southwest-2.myhuaweicloud.com/base_image/ascend-ci

FROM $REGISTRY/cann:$CANN_VERSION-$DEVICE_TYPE-$OS-$PYTHON_VERSION

# Define environments
ARG TARGETARCH
ARG PIP_INDEX_URL="https://repo.huaweicloud.com/repository/pypi/simple"
ARG APTMIRROR="repo.huaweicloud.com"
ARG PYTORCH_VERSION=2.8.0
ARG TORCHVISION_VERSION=0.23.0
ARG VLLM_TAG=v0.8.5
ARG BISHENG_URL="https://sglang-ascend.obs.cn-east-3.myhuaweicloud.com/sglang/Ascend-BiSheng-toolkit_aarch64.run"
ARG SGLANG_TAG=main
ARG ASCEND_CANN_PATH=/usr/local/Ascend/ascend-toolkit
ARG SGLANG_KERNEL_NPU_TAG=main

# Set environment variables according to architecture
ARG PTA_URL_AMD64="https://gitcode.com/Ascend/pytorch/releases/download/v7.2.0-pytorch2.8.0/torch_npu-2.8.0-cp311-cp311-manylinux_2_28_x86_64.whl"

ARG PTA_URL_ARM64="https://gitcode.com/Ascend/pytorch/releases/download/v7.2.0-pytorch2.8.0/torch_npu-2.8.0-cp311-cp311-manylinux_2_28_aarch64.whl"

RUN if [ "$TARGETARCH" = "amd64" ]; then \
      echo "Using x86_64 dependencies"; \
      echo "PTA_URL=$PTA_URL_AMD64" >> /etc/environment_new; \
    elif [ "$TARGETARCH" = "arm64" ]; then \
      echo "Using aarch64 dependencies"; \
      echo "PTA_URL=$PTA_URL_ARM64" >> /etc/environment_new; \
    else \
      echo "Unsupported TARGETARCH: $TARGETARCH"; exit 1; \
    fi

WORKDIR /workspace
ENV DEBIAN_FRONTEND=noninteractive

# Update pip & apt sources
RUN pip config set global.index-url $PIP_INDEX_URL \
    && if [ -n "$APTMIRROR" ]; then sed -i "s|//.*.ubuntu.com|//$APTMIRROR|g" /etc/apt/sources.list; fi

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

ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US:en
ENV LC_ALL=en_US.UTF-8
ENV PATH="/root/.cargo/bin:${PATH}"

# Install dependencies
# TODO: install from pypi released memfabric
RUN pip install mf-adapter==1.0.0

RUN pip install setuptools-rust wheel build --no-cache-dir

# install rustup from rustup.rs
RUN export RUSTUP_DIST_SERVER=https://mirrors.ustc.edu.cn/rust-static \
    && export RUSTUP_UPDATE_ROOT=https://mirrors.ustc.edu.cn/rust-static/rustup \
    && curl --proto '=https' --tlsv1.2 -sSf https://mirrors.ustc.edu.cn/rust-static/rustup/rustup-init.sh | sh -s -- -y \
    && rustc --version && cargo --version && protoc --version
    
# Install vLLM
RUN git clone --depth 1 https://github.com/vllm-project/vllm.git --branch $VLLM_TAG && \
    (cd vllm && VLLM_TARGET_DEVICE="empty" pip install -v . --timeout 1000 --resume-retries 5 --no-cache-dir) && rm -rf vllm

# Install torch PTA
RUN . /etc/environment_new && \
    pip install torch==$PYTORCH_VERSION torchvision==$TORCHVISION_VERSION --no-cache-dir \
    && wget ${PTA_URL} && pip install $(basename ${PTA_URL}) --no-cache-dir \
    && python3 -m pip install --no-cache-dir attrs==24.2.0 numpy==1.26.4 scipy==1.13.1 decorator==5.1.1 psutil==6.0.0 pytest==8.3.2 pytest-xdist==3.6.1 pyyaml pybind11

# Install from test.pypi released triton-ascend
RUN pip install \
    --timeout 60 \
    --retries 3 \
    "triton-ascend < 3.2.0rc" \
    --pre \
    --no-cache-dir

# Install SGLang
RUN mkdir /workspace/sglang
COPY . /workspace/sglang
RUN (cd sglang/python && rm -rf pyproject.toml \
    && mv pyproject_other.toml pyproject.toml && pip install -v .[srt_npu] --no-cache-dir) \
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

# Install Bisheng
RUN wget ${BISHENG_URL} && chmod a+x Ascend-BiSheng-toolkit_aarch64.run && ./Ascend-BiSheng-toolkit_aarch64.run --install && rm Ascend-BiSheng-toolkit_aarch64.run

CMD ["/bin/bash"]