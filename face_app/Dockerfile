FROM python:3.6.5

WORKDIR /face_app

ADD . /face_app

COPY requirements.txt .
# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN pip install --upgrade pip
RUN pip install -U setuptools

RUN apt-get update \
    && apt-get -y install curl \
    cmake

RUN pip install --no-cache-dir -r requirements.txt


RUN python download.py

COPY . .