FROM pytorch/pytorch

COPY requirements.txt requirements.txt

# upgrade pip and wheel
RUN pip install -U pip wheel & \
    pip install -r requirements.txt
