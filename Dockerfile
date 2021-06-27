FROM python:3.8

ENV PYTHONPATH=/src

WORKDIR /src
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install numpy
RUN pip install opencv-contrib-python
COPY main.py /src/main.py
COPY pixilizer/ /src/pixilizer/
RUN pip install -r requirements.txt

CMD ["python", "main.py"]