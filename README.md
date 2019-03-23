# ABSA-Transformer
![Docker Build Status](https://img.shields.io/docker/cloud/build/jorba/absa-transformer.svg)

This is the repository for my NLP master thesis with the title **Transfer and Multitask Learning for Aspect-Based Sentiment Analysis Using the Google Transformer Architecture**.

It is based on the Google Transformer architecture, and the paper *Attention is all you need* (https://arxiv.org/abs/1706.03762)

I recommend to check out the excellent [Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html) guide from Harvard or the [Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) by Jay Alammar. Both are excellent resources on the topic.
## Run Code

The first step to run this code is to clone the repository using `git clone `

### Docker

I created a Docker image you can use to run the code inside a Docker container. The Docker image is based on https://github.com/anibali/docker-pytorch and runs on CUDA 8.0

#### CUDA
The image is based on https://github.com/anibali/docker-pytorch. In order to run it with CUDA support you need to install the latest NVIDIA drivers and libraries as well as CUDA.

You also need to install the NVIDIA Docker runtime which you can find here: https://github.com/NVIDIA/nvidia-docker

#### Run prebuild image from DockerHub (recommended)
To run this image you have to login to the DockerHub with `docker login`. To create a Docker account head over to https://cloud.docker.com

This repository can also be obtained prebuild from the DockerHub. For now, the repository is private. Once authenticated the image can be run via

```
docker run -it --rm --init \
	-p 8888:8888 \
	--runtime=nvidia \
	--volume=$(pwd):/app \
	--name=absa
	jorba/absa-transformer:latest
```

Use this one-liner for easy copy and pasting: 

`docker run -it --rm --init -p 8888:8888 --runtime=nvidia --volume=$(pwd):/app --name=absa jorba/absa-transformer:latest`

The commands above starts the container in interactive mode which means that as soon as you close your session, the container stops. You probably don't want this behavior when running long experiments. To run the container in detached mode, use this command:

`docker run -d -p 8888:8888 --rm --runtime=nvidia --volume=$(pwd):/app --name=absa jorba/absa-transformer:latest`

To get the token of the notebook, use `docker logs absa`.

#### Upgrade image from DockerHub
The Docker runtime always tries to run the local version of the image instead of pulling a new one from the Docker hub even when including the tag `:latest`. To upgrade a container, run the following commands:

```
docker pull jorba/absa-transformer:latest
docker stop absa
docker rm absa
docker run -it --init -p 8888:8888 --runtime=nvidia --volume=$(pwd):/app --name=absa jorba/absa-transformer:latest
```

#### Manual build with Dockerfile

To build the Docker image, navigate to the repository folder which contains the `Dockerfile`.

Next, run the Docker build command:

```
docker build .
```

Make sure you include the `.` at the end of the command to specify that Docker should search for the `Dockerfile` in the current directory.

Note down the Docker image id which should have a format like `3624c152fb28`.
##### Run manual build

To run the image simply run this command in the terminal. (Replace `image:version` with your imageId. You can also get this id by running the command `docker images`)

It creates a container and start the jupyter notebook server which you can access at http://localhost:8888
```
docker run -it --init --rm \
	-p 8888:8888 \
	--runtime=nvidia \
	--name=absa \
	--volume=$(pwd):/app \
	image:version
```

For the windows command line (not powershell), use `%cd%` instead to mount the current directory as a volume so that the run command looks like this:

```
docker run -it --init --rm \
	-p 8888:8888 \
	--runtime=nvidia \
	--name=absa \
	--volume=%cd%:/app \
	image:version
```

### Connect to remote Notebook

Connecting to a notebook running on a remote machine is easy. Just run this command on your local machine. This opens an SSH-tunnel to the remote machine and maps the jupyter-ports 8888 to your local machines 8888 port. Make sure a local notebook does not occupy this port.

`ssh -N -L 8080:localhost:8080 yourname@remoteServer.com`

If you don't have an SSH key, this will ask for your password. After that don't expect any further output. It should still work. Just try to access the notebook on your local machine at http://localhost:8888 

There is one caveat regarding jupyter notebooks though. The browser tab on your local machine has to remain open the whole time. Once you lose connection or close the tab, the script does not stop but you will not get any further output.

If you do not have a local machine which runs 24/7 you can also use a VM which keeps the notebook open.