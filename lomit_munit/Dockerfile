FROM images.borgy.elementai.lan/eai/repl_docker_base_nvidia_pytorch:latest

LABEL maintainer="Ripples Team <ripples@elementai.com>"

WORKDIR /eai/project
ENV HOME /eai/project
ENV PYTHONPATH /eai/project/

COPY requirements.txt /eai/project/
RUN pip install --requirement requirements.txt

COPY setup.py /eai/project/
