# Compiles a docker image for blender w/ "import bpy support"
# Using Blender 4.4 with CUDA 12.9 for RTX 5090 (sm_120 Blackwell) support
#
# Compilation happens in two stages:
# 1) Compiles blender from source.
# 2) Installs previously built bpy module along with other dependencies in a fresh image.

# #################################################################################################
# Stage 1: Build Stage
# #################################################################################################

FROM nvidia/cuda:12.9.0-devel-ubuntu22.04 AS build

ENV DEBIAN_FRONTEND=noninteractive
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

WORKDIR /blenderpy

# --- Install package dependencies
RUN apt-get update --yes --fix-missing && \
    apt-get install --yes --quiet --no-install-recommends \
      python3.11 \
      python3.11-dev \
      python3.11-distutils \
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
      python3-numpy \
      wget \
      subversion \
      software-properties-common \
      libepoxy-dev \
      libpulse-dev \
      libdecor-0-dev

# Initialize git-lfs
RUN git lfs install

# Install GCC 12 (compatible with CUDA 12.9)
RUN add-apt-repository ppa:ubuntu-toolchain-r/test -y && \
    apt-get update && \
    apt-get install --yes gcc-12 g++-12 && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 100 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 100

# Install newer CMake (Blender 4.4 requires CMake 3.19+)
RUN wget -q https://github.com/Kitware/CMake/releases/download/v3.28.0/cmake-3.28.0-linux-x86_64.tar.gz && \
    tar -xzf cmake-3.28.0-linux-x86_64.tar.gz && \
    mv cmake-3.28.0-linux-x86_64 /opt/cmake && \
    ln -sf /opt/cmake/bin/cmake /usr/local/bin/cmake && \
    rm cmake-3.28.0-linux-x86_64.tar.gz

RUN which cmake && cmake --version

# make python3.11 the default python (Blender 4.4 uses Python 3.11)
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 10 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 10

# Install pip for python 3.11
RUN wget https://bootstrap.pypa.io/get-pip.py && \
    python3.11 get-pip.py && \
    rm get-pip.py

# Install numpy for build
RUN pip3 install numpy

# --- Clone Blender 4.4 source (has Blackwell sm_120 support) and initialize submodules
# Skip LFS during clone to avoid 404 errors on some assets
# The make_update.py script will handle assets properly
RUN GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/blender/blender.git --branch blender-v4.4-release --depth 1 && \
    cd blender && \
    GIT_LFS_SKIP_SMUDGE=1 git submodule update --init --recursive --depth 1 && \
    echo "=== Checking addons after submodule init ===" && \
    ls -la scripts/addons/ 2>/dev/null | head -20 || echo "No addons dir yet"

# --- Download official precompiled libs for Blender 4.4
RUN cd blender && \
    ./build_files/utils/make_update.py --use-linux-libraries

# Enable CUDA for GPU rendering with Blackwell (sm_120) support
# Blender 4.4 has native support for sm_120 with CUDA 12.8+
RUN echo 'set(WITH_CYCLES_DEVICE_CUDA     ON  CACHE BOOL "" FORCE)' >> /blenderpy/blender/build_files/cmake/config/bpy_module.cmake && \
    echo 'set(WITH_CYCLES_CUDA_BINARIES   ON  CACHE BOOL "" FORCE)' >> /blenderpy/blender/build_files/cmake/config/bpy_module.cmake && \
    echo 'set(CYCLES_CUDA_BINARIES_ARCH   "sm_86;sm_89;sm_90;sm_120" CACHE STRING "" FORCE)' >> /blenderpy/blender/build_files/cmake/config/bpy_module.cmake && \
    echo 'set(WITH_CYCLES_DEVICE_OPTIX    ON  CACHE BOOL "" FORCE)' >> /blenderpy/blender/build_files/cmake/config/bpy_module.cmake

# Build bpy module with reduced parallelism to avoid OOM during linking
RUN cd blender && make bpy

# Debug: Find where the bpy module was actually built
RUN echo "=== Finding bpy .so files ===" && \
    find /blenderpy -name "*.so" -path "*bpy*" 2>/dev/null && \
    echo "=== Contents of build_linux_bpy/bin ===" && \
    ls -laR /blenderpy/build_linux_bpy/bin/ 2>/dev/null | head -100 && \
    echo "=== Looking for bpy module files ===" && \
    find /blenderpy/build_linux_bpy -name "bpy*" -type f 2>/dev/null

# Debug: Show what was built
RUN echo "=== Build output structure ===" && \
    find /blenderpy/build_linux_bpy/bin -type f -name "*.so" && \
    ls -la /blenderpy/build_linux_bpy/bin/bpy/ && \
    ls -la /blenderpy/build_linux_bpy/bin/bpy/4.4/ 2>/dev/null || true && \
    ls -la /blenderpy/build_linux_bpy/bin/bpy/4.4/scripts/ 2>/dev/null || true

# #################################################################################################
# Stage 2: Runtime Stage
# #################################################################################################

FROM nvidia/cuda:12.9.0-runtime-ubuntu22.04

LABEL Author="kubric-team <kubric@google.com>"
LABEL Title="Blender"

ENV DEBIAN_FRONTEND=noninteractive
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# --- Install package dependencies
RUN apt-get update --yes --fix-missing && \
    apt-get install --yes --quiet --no-install-recommends \
      python3.11 \
      python3.11-dev \
      python3.11-distutils \
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
      libxxf86vm1 \
      libepoxy0 \
      libpulse0 \
      libdecor-0-0

# make python3.11 the default python and python3
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 10 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 10

# install pip for python 3.11
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3.11 get-pip.py && \
    rm get-pip.py

# Copy the entire bpy module directory (includes __init__.so and libs)
COPY --from=build /blenderpy/build_linux_bpy/bin/bpy/ /usr/local/lib/python3.11/dist-packages/bpy/

# Copy Blender scripts (modules, startup, etc.) - critical for bpy.types to work
# In Blender 4.4, scripts are directly under scripts/, not release/scripts/
COPY --from=build /blenderpy/blender/scripts/modules/ /usr/local/lib/python3.11/dist-packages/bpy/4.4/scripts/modules/
COPY --from=build /blenderpy/blender/scripts/startup/ /usr/local/lib/python3.11/dist-packages/bpy/4.4/scripts/startup/

# Copy additional runtime libs (Intel OneAPI/SYCL for GPU support)
COPY --from=build /blenderpy/blender/lib/linux_x64/dpcpp/lib/ /usr/local/lib/blender-dpcpp/

# Copy Blender addons (required for OBJ/FBX/glTF import)
# In Blender 4.4, addons are in scripts/addons_core and extensions
COPY --from=build /blenderpy/blender/scripts/addons_core/ /usr/local/lib/python3.11/dist-packages/bpy/4.4/scripts/addons_core/

# Set library path so bpy can find its bundled libs
ENV LD_LIBRARY_PATH="/usr/local/lib/python3.11/dist-packages/bpy/lib:/usr/local/lib/blender-dpcpp:${LD_LIBRARY_PATH}"