# --- Image with *pre-installed* Kubric python package
# 
# docker run --rm --interactive \
#   --user $(id -u):$(id -g) \
#   --volume "$PWD:/kubric" \
#   --workdir "/kubric" \
#   kubricdockerhub/kubruntu:latest \
#   python3 examples/helloworld.py

FROM kubricdockerhub/blender:v3.6-cuda11.8
WORKDIR /kubric

# --- copy requirements in workdir
COPY requirements.txt .
COPY requirements_full.txt .

# --- install uv for faster python dependencies installation
RUN pip install uv

# --- install python dependencies
RUN uv pip install --system --upgrade pip wheel
RUN uv pip install --system --upgrade --force-reinstall -r requirements.txt
RUN uv pip install --system --upgrade --force-reinstall -r requirements_full.txt

# --- cleanup
RUN rm -f requirements.txt
RUN rm -f requirements_full.txt

# --- Silences tensorflow
ENV TF_CPP_MIN_LOG_LEVEL="3"

# --- Fix matplotlib config directory
ENV MPLCONFIGDIR="/tmp/matplotlib"

# --- Install Kubric
COPY dist/kubric*.whl .
RUN pip3 install `ls kubric*.whl`
RUN rm -f kubric*.whl

# --- Update Python path to 3.10 (matches new blender image)
COPY kubric/renderer/blender.py /usr/local/lib/python3.10/dist-packages/kubric/renderer/blender.py