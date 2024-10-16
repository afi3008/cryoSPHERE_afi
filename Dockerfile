# syntax=docker/dockerfile:1

# Comments are provided throughout this file to help you get started.
# If you need more help, visit the Dockerfile reference guide at
# https://docs.docker.com/go/dockerfile-reference/

# Want to help us make this template better? Share your feedback here: https://forms.gle/ybq9Krt8jtBL3iCk7

#FROM python:${PYTHON_VERSION}-slim as base
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu20.04 as base
ARG DEBIAN_FRONTEND=noninteractive

# Prevents Python from writing pyc files.
ENV PYTHONDONTWRITEBYTECODE=1

# Keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering.
ENV PYTHONUNBUFFERED=1

WORKDIR /app
# Create a non-privileged user that the app will run under.
# See https://docs.docker.com/go/dockerfile-user-best-practices/
ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/nonexistent" \
    --shell "/sbin/nologin" \
    --no-create-home \
    --uid "${UID}" \
    appuser

RUN apt update
RUN apt-get -y install sudo
RUN apt-get -y install wget
RUN apt-get -y  install build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev libsqlite3-dev wget libbz2-dev
RUN wget https://www.python.org/ftp/python/3.9.20/Python-3.9.20.tgz
RUN tar -xf Python-3.9.20.tgz
WORKDIR Python-3.9.20/
RUN ./configure --enable-optimizations
RUN make
RUN make install
WORKDIR /app
RUN python3.9 -V
#RUN apt-get update
#RUN apt-get -y install python3.9 && ln -s /usr/bin/python3.9 /usr/bin/python3
RUN apt install python3-venv python3-pip -y
RUN apt-get -y install git
RUN pip3 install nvidia-cublas-cu12
# Download dependencies as a separate step to take advantage of Docker's caching.
# Leverage a cache mount to /root/.cache/pip to speed up subsequent builds.
# Leverage a bind mount to requirements.txt to avoid having to copy them into
# into this layer.
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=requirements.txt,target=requirements.txt \
    python3 -m pip install -r requirements.txt

#RUN pip install "git+https://github.com/facebookresearch/pytorch3d.git"
# Switch to the non-privileged user to run the application.
USER appuser

# Copy the source code into the container.
COPY . .

# Expose the port that the application listens on.
EXPOSE 8000

# Run the application.
