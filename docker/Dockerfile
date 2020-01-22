FROM pytorch/pytorch

LABEL maintainer="anamolous.behaviour@gmail.com"
LABEL version=0.1
LABEL description="PyTorch image with HuggingFace Transformers installed"

# copy the Python requirements into build
COPY requirements.txt requirements.txt

# upgrade pip and wheel
RUN pip install -U pip wheel & \
    pip install -r requirements.txt
