# Driver Monitering System - Triton Interference Server
## 1. Run the pipeline in local machine
You could run the pipeline in your local machine py running the file [pipeline.ipynb](pipeline.ipynb).
## 2. Run the pipeline in server, using Triton Interference Server
### 2.1 Server side
First, we run the Triton Interference Server using docker. `<xx.yy>` here is the version of Triton Interference Server.
```
docker run --shm-size=1g --ulimit memlock=-1 -p 8000:8000 -p 8001:8001 -p 8002:8002 --ulimit stack=67108864 -ti nvcr.io/nvidia/tritonserver:<xx.yy>-py3 
```
Second, inside the container, start the Triton server.
```
cd server
tritonserver --model-repository `pwd`/models
```
### 2.2 Client side
In the host machine, start the client container.
```
docker run -ti --net host nvcr.io/nvidia/tritonserver:<xx.yy>-py3-sdk /bin/bash
```
In the client container, clone this repository again.
```

```
And finally we run the python file [client/client.py](client/client.py)
```
python client/client.py
```