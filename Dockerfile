FROM mxnet/python:nightly_gpu_cu100_py3
#LABEL version="0.1" owner="GGMO"

WORKDIR project
COPY requirements.txt ./
RUN apt-get install -y libsm6 && apt-get  install -y libxrender1 && apt-get install -y libxext-dev && apt-get install -y libglib2.0-0 && pip install -r requirements.txt
COPY project .

CMD ["python3","faces_web.py"]