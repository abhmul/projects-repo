FROM tensorflow/tensorflow:latest-gpu-jupyter
RUN apt-get update && \
    apt-get install -y vim
COPY requirements.txt /opt/app/requirements.txt
RUN pip install --upgrade pip && \ 
    pip install -r /opt/app/requirements.txt

