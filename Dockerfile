FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04
ENV HOME=/usr/src/app

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    libgl1-mesa-dev \
 && rm -rf /var/lib/apt/lists/*


COPY environment.yml .

# Install Miniconda and Python 3.8
ENV CONDA_AUTO_UPDATE_CONDA=false
ENV PATH=/miniconda/bin:$PATH
RUN curl -sLo /miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-py37_4.9.2-Linux-x86_64.sh \
 && chmod +x /miniconda.sh \
 && /miniconda.sh -b -p /miniconda \
 && rm -f /miniconda.sh \
 && conda clean -ya \
 && conda env update --name base -f environment.yml

WORKDIR $HOME
