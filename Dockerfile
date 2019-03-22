# the first part is from FROM anibali/pytorch:cuda-8.0
# +++++++++++++++++++++START++++++++++++++++++++++++++

FROM nvidia/cuda:8.0-runtime-ubuntu16.04

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
	ipython ipython-notebook \
 && rm -rf /var/lib/apt/lists/*

# Create a working directory
RUN mkdir /app
WORKDIR /app

# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' --shell /bin/bash user \
 && chown -R user:user /app
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user

# All users can use /home/user as their home directory
ENV HOME=/home/user
RUN chmod 777 /home/user

# Install Miniconda
RUN curl -so ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-4.5.11-Linux-x86_64.sh \
 && chmod +x ~/miniconda.sh \
 && ~/miniconda.sh -b -p ~/miniconda \
 && rm ~/miniconda.sh
ENV PATH=/home/user/miniconda/bin:$PATH
ENV CONDA_AUTO_UPDATE_CONDA=false

# Create a Python 3.6 environment
RUN /home/user/miniconda/bin/conda install conda-build \
 && /home/user/miniconda/bin/conda create -y --name py36 python=3.6.5 \
 && /home/user/miniconda/bin/conda clean -ya
ENV CONDA_DEFAULT_ENV=py36
ENV CONDA_PREFIX=/home/user/miniconda/envs/$CONDA_DEFAULT_ENV
ENV PATH=$CONDA_PREFIX/bin:$PATH

# CUDA 8.0-specific steps
RUN conda install -y -c pytorch \
    cuda80=1.0 \
    magma-cuda80=2.3.0 \
    "pytorch=1.0.0=py3.6_cuda8.0.61_cudnn7.1.2_1" \
    torchvision=0.2.1 \
 && conda clean -ya

# Install HDF5 Python bindings
RUN conda install -y h5py=2.8.0 \
 && conda clean -ya
RUN pip install h5py-cache==1.0

# Install Torchnet, a high-level framework for PyTorch
RUN pip install torchnet==0.0.4

# Install Requests, a Python library for making HTTP requests
RUN conda install -y requests=2.19.1 \
 && conda clean -ya

# Install Graphviz
RUN conda install -y graphviz=2.38.0 \
 && conda clean -ya
RUN pip install graphviz==0.8.4

# Install OpenCV3 Python bindings
RUN sudo apt-get update && sudo apt-get install -y --no-install-recommends \
    libgtk2.0-0 \
    libcanberra-gtk-module \
 && sudo rm -rf /var/lib/apt/lists/*
RUN conda install -y -c menpo opencv3=3.1.0 \
 && conda clean -ya


# +++++++++++++++++++++++END+++++++++++++++++++++++++

RUN conda update -n base -c default conda

# install jupyter
RUN conda install -y jupyter 

# get additional requirements - conda
RUN conda install -y tqdm
RUN conda install -y pandas

RUN conda install -y spacy \
	&& python -m spacy download en \
	&& python -m spacy download de

RUN conda install -y -c conda-forge prettytable
RUN conda install -y -c conda-forge beautifulsoup4
RUN conda install -c conda-forge matplotlib 
RUN conda install -c anaconda seaborn
RUN conda install -y scikit-learn \
	&& conda clean -ya

# get additional requirements - pip
RUN pip install hyperopt
RUN pip install torchtext
RUN pip install stop-words
RUN pip install pyspellchecker
RUN pip install revtok
RUN pip install tensorflow
RUN pip install tensorboardX
RUN pip install colorama

# install jupyter notebook

ENTRYPOINT ["jupyter-notebook", "--ip", "0.0.0.0", "--no-browser", "--allow-root", "--port=8888"]