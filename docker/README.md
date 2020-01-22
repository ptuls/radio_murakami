# PyTorch with HuggingFace Transformer Docker Image

To build, on command line run
```
docker build .
```

This should build the image by pulling the `pytorch/pytorch` image and install all requirements in `requirements.txt`. To run the built image, run
```
docker run -it <image_id>
```
