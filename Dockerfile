FROM mxnet/python:nightly_gpu_cu100_py3
#LABEL version="0.1" owner="GGMO"

WORKDIR face_web
COPY requirements.txt ./
RUN apt-get update && apt-get install -y libsm6 && apt-get  install -y libxrender1 && apt-get install -y libxext-dev && apt-get install -y libglib2.0-0 && pip install --upgrade pip && pip install -r requirements_docker.txt
COPY face_web .

CMD ["python3","faces_web.py"]