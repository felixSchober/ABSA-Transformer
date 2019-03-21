# ABSA-Transformer
This is the repository for my NLP master thesis with the title **Transfer and Multitask Learning for Aspect-Based Sentiment Analysis Using the Google Transformer Architecture**.

It is based on the Google Transformer architecture and the paper *Attention is all you need* (https://arxiv.org/abs/1706.03762)

I recommend to check out the excellent [Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html) guide from Harvard or the [Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) by Jay Alammar. Both are excellent resources on the topic.
## Run Code
### Docker

I created a docker image you can use to run the code inside a docker container.

#### CUDA
The image is based on https://github.com/anibali/docker-pytorch. In order to run it with cuda support you need to install the latest NVIDIA drivers and libraries as well as CUDA.

You will also need to install the NVIDIA docker runtime which you can find here: https://github.com/NVIDIA/nvidia-docker

#### Run prebuild image from DockerHub (recomended)
To run this image you have to login to the DockerHub wiht `docker login`. The create a docker account head over to https://cloud.docker.com

This repository can also be obtained prebuild from the DockerHub. For now the repository is private. Once authenticated the image can be run via

```
docker run -rm -it --init \
	-p 8888:8888 \
	--runtime=nvidia \
	--volume=$(pwd):/app \
	jorba/absa-transformer:latest
```

#### Build with Dockerfile

To build the docker image, navigate to the repository folder which contains the `Dockerfile`.

Next, run the docker build command:

```
docker build .
```

Make sure you include the `.` at the end of the command to specifiy that docker should search for the `Dockerfile` in the current directory.

Note down the docker image id which should have a format like `3624c152fb28`.
#### Run

To run the image simply run this command in the terminal. (Replace `image:version` with your imageId. You can also get this id by running the command `docker images`)

It will create a container and start the jupyter notebook server which you can access at http://localhost:8888
```
docker run -rm -it --init \
	-p 8888:8888 \
	--runtime=nvidia \
	--volume=$(pwd):/app \
	image:version
```

For the windows command line (not powershell), use `%cd%` instead to mount the current directory as a volume so that the run command looks like this:

```
docker run -rm -it --init \
	-p 8888:8888 \
	--runtime=nvidia \
	--volume=%cd%:/app \
	image:version
```