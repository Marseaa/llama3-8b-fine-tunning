
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive PIP_PREFER_BINARY=1 \
        CUDA_HOME=/usr/local/cuda-11.8 TORCH_CUDA_ARCH_LIST="8.6"
RUN rm /bin/sh && ln -s /bin/bash /bin/sh  
    
ENV DEBIAN_FRONTEND noninteractive
    
RUN apt-get update && apt-get install -y wget 
    
# instalando python
RUN apt-get install -y python3  && apt-get install -y python3-pip

WORKDIR /app

RUN pip install torch torchvision torchaudio

### unsloth
RUN pip install --upgrade --force-reinstall --no-cache-dir torch==2.2.0 triton \
  --index-url https://download.pytorch.org/whl/cu121

COPY unsloth /app/unsloth
RUN pip install "./unsloth[cu121-torch220]"
# RUN pip install "./unsloth[cu121-ampere-torch220]" - aproveita mais a gpu
### unsloth

#outras dependencias
RUN pip install datasets trl transformers 

RUN pip install --no-deps packaging ninja einops xformers trl peft accelerate bitsandbytes

RUN pip install torch==2.3.0+cu121 -f https://download.pytorch.org/whl/torch_stable.html

RUN pip install xformers==0.0.26.post1 --no-cache-dir


# Adicionando unsloth ao PYTHONPATH
ENV PYTHONPATH "/app/unsloth"

ENV NVIDIA_VISIBLE_DEVICES all
WORKDIR /app
CMD ["python3", "data.py"]
