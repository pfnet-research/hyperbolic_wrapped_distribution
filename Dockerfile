FROM nvcr.io/nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

RUN DEBIAN_FRONTEND=noninteractive \
  apt-get update -qq -y && \
  apt-get install -qq -y python3-pip && \
  rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir -q \
    tqdm \
    ruamel.yaml \
    mypy \
    seaborn \
    jupyterthemes \
    numba \
    Pillow \
    scikit-learn==0.20.1 \
    chainer==5.0.0 \
    cupy-cuda90==5.0.0 \
    optuna==0.5.0 \
    colorama==0.3.9 \
    pygments==2.2.0 \
    nltk==3.3 \
    h5py && \
  pip3 install -q --upgrade --user tornado terminado && \
  pip3 install -q --no-cache-dir --upgrade --force-reinstall html5lib && \
  pip3 install -q --upgrade ipywidgets jupyter_contrib_nbextensions && \
  jt -t grade3 -T -N && \
  jupyter contrib nbextension install --user > /dev/null 2>&1 && \
  jupyter nbextension enable hinterland/hinterland && \
  jupyter nbextension enable --py widgetsnbextension

CMD ["bash"]
