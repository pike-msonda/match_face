FROM python:3.6.5

WORKDIR /face_app

ADD . /face_app

COPY requirements.txt .
# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN pip install --upgrade pip
RUN pip install -U setuptools

RUN pip install --no-cache-dir -r requirements.txt

COPY detect_face.py /usr/local/lib/python3.6/site-packages/easyfacenet/src/align/detect_face.py

RUN python download.py

COPY . .