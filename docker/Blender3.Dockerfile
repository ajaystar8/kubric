# Compiles a docker image for blender w/ "import bpy support"
# Using Blender 3.6 LTS with CUDA 11.8 for RTX 4090 (sm_89) support
#
# Compilation happens in two stages:
# 1) Compiles blender from source.
# 2) Installs previously built bpy module along with other dependencies in a fresh image.

# #################################################################################################
# Stage 1
# #################################################################################################

FROM nvidia/cuda:11.8.0-devel-ubuntu20.04 AS build

ENV DEBIAN_FRONTEND=noninteractive
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

WORKDIR /blenderpy

# --- Install package dependencies
RUN apt-get update --yes --fix-missing && \
    apt-get install --yes --quiet --no-install-recommends \
      python3.8-dev \
      build-essential \
      ca-certificates \
      git \
      git-lfs \
      libffi-dev \
      libssl-dev \
      libx11-dev \
      libxxf86vm-dev \
      libxcursor-dev \
      libxi-dev \
      libxrandr-dev \
      libxinerama-dev \
      libglew-dev \
      libxkbcommon-dev \
      libgl1-mesa-dev \
      libglu1-mesa-dev \
      libfftw3-dev \
      libwayland-dev \
      wayland-protocols \
      libegl-dev \
      libdbus-1-dev \
      python3-dev \
      python3-numpy \
      wget \
      subversion

# Initialize git-lfs
RUN git lfs install

# Install GCC 11 (Blender 3.6 requires GCC 11+)
RUN apt-get install --yes software-properties-common && \
    add-apt-repository ppa:ubuntu-toolchain-r/test -y && \
    apt-get update && \
    apt-get install --yes gcc-11 g++-11 && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 100 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 100

# Install newer CMake (Blender 3.6 requires CMake 3.19+)
RUN wget -q https://github.com/Kitware/CMake/releases/download/v3.27.0/cmake-3.27.0-linux-x86_64.tar.gz && \
    tar -xzf cmake-3.27.0-linux-x86_64.tar.gz && \
    mv cmake-3.27.0-linux-x86_64 /opt/cmake && \
    ln -sf /opt/cmake/bin/cmake /usr/local/bin/cmake && \
    rm cmake-3.27.0-linux-x86_64.tar.gz

RUN which cmake && cmake --version

# make python3.8 the default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 10 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 10

# --- Clone Blender 3.6 LTS source and initialize submodules (for addons)
RUN git clone https://github.com/blender/blender.git --branch blender-v3.6-release --depth 1 && \
    cd blender && \
    git submodule update --init --recursive --depth 1 && \
    echo "=== Checking addons after submodule init ===" && \
    ls -la scripts/addons/ | head -20

# --- Download official precompiled libs for Blender 3.6
RUN cd blender && \
    ./build_files/utils/make_update.py --use-linux-libraries

# enable CUDA for GPU rendering
RUN echo 'set(WITH_CYCLES_DEVICE_CUDA     ON  CACHE BOOL "" FORCE)' >> /blenderpy/blender/build_files/cmake/config/bpy_module.cmake && \
    echo 'set(WITH_CYCLES_CUDA_BINARIES   ON  CACHE BOOL "" FORCE)' >> /blenderpy/blender/build_files/cmake/config/bpy_module.cmake && \
    echo 'set(CYCLES_CUDA_BINARIES_ARCH   "sm_86;sm_89" CACHE STRING "" FORCE)' >> /blenderpy/blender/build_files/cmake/config/bpy_module.cmake

# Build bpy module
RUN cd blender && make bpy

# #################################################################################################
# Stage 2
# #################################################################################################

FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

LABEL Author="kubric-team <kubric@google.com>"
LABEL Title="Blender"

ENV DEBIAN_FRONTEND=noninteractive
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# --- Install package dependencies (Ubuntu 22.04 has Python 3.10 by default)
RUN apt-get update --yes --fix-missing && \
    apt-get install --yes --quiet --no-install-recommends \
      python3.10 \
      python3.10-dev \
      python3.10-distutils \
      build-essential \
      imagemagick \
      curl \
      ca-certificates \
      git \
      libffi-dev \
      libssl-dev \
      libx11-dev \
      libxxf86vm-dev \
      libxcursor-dev \
      libxi-dev \
      libxrandr-dev \
      libxinerama-dev \
      libglew-dev \
      zlib1g-dev \
      libbz2-dev \
      libgdbm-dev \
      liblzma-dev \
      libncursesw5-dev \
      libreadline-dev \
      libsqlite3-dev \
      uuid-dev \
      libxkbcommon0 \
      libsm6 \
      libice6 \
      libgl1 \
      libglu1-mesa \
      libxi6 \
      libxrender1 \
      libxfixes3 \
      libxxf86vm1

# make python3.10 the default python and python3
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 10 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 10

# install pip for python 3.10
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3.10 get-pip.py && \
    rm get-pip.py

# Copy the entire bpy module directory (includes __init__.so and libs)
COPY --from=build /blenderpy/build_linux_bpy/bin/bpy/ /usr/local/lib/python3.10/dist-packages/bpy/

# Copy additional runtime libs (Intel OneAPI/SYCL for GPU support)
COPY --from=build /blenderpy/blender/lib/linux_x64/dpcpp/lib/ /usr/local/lib/blender-dpcpp/

# Copy Blender addons (required for OBJ/FBX/glTF import)
COPY --from=build /blenderpy/blender/scripts/addons/ /usr/local/lib/python3.10/dist-packages/bpy/3.6/scripts/addons/

# Set library path so bpy can find its bundled libs
ENV LD_LIBRARY_PATH="/usr/local/lib/python3.10/dist-packages/bpy/lib:/usr/local/lib/blender-dpcpp:${LD_LIBRARY_PATH}"