# DeepSeek R1 (Inference) on TIR Training Cluster

This document covers steps to deploy Deepseek R1 on training cluster for inference. This approach is quite useful when experimenting with the model or generating synthetic data.

Stages:
1) [Create a Training Cluster](https://github.com/mindhash/r1-on-tir/blob/main/r1-on-training-cluster.md#stage-1)
2) [Login to cluster nodes - master and worker](https://github.com/mindhash/r1-on-tir/blob/main/r1-on-training-cluster.md#stage-2)
3) Install sglang and other libraries from the master node
4) Start the sglang server on master and worker node
5) Test R1 endpoint


## Stage 1

In this stage, we will configure a training cluster with pytorch framework on 16xH100.

1) Go to TIR Dashboard
2) Navigate to Training cluster from left-hand sidebar
3) Click on `create cluster` and choose 16xH100 plan. For this experiment, we will also need a shared file system (SFS). So when prompted create a SFS of atleast 1TB. Wait for the cluster to be ready (Assigned status). 
4) Click on Deployments tab 
5) Create `Create Deployments`
6) Choose `Pytorch Distributed` as framework and make sure your cluster (Created in 1.3) is selected
7) Click `next` and add/select your SSH key to be added to login (master) node.
8) Choose SFS (created in 3) in next step and choose `/shared` as mount location
9) Complete the flow to start a deployment. and wait for the deployment to show `RUNNING` state
10) When deployment is in running state, you would see the list of nodes in `worker` tab on the same page. Locate master node in the table and click on `connect` icon
11) Use the instructions from right-hand sidebar drawer to log into master node. use the same instruction (e.g. `ssh admin@xx.xx.x..x`) to login from 3 terminal session.

## Stage 2

From here on, we will use 3 terminal sessions. From each of these terminal the steps will follow. 

### Terminal 1 (Master Node)

  1.1) Login to master node (use instructions from TIR):

```
# get the exact instruction from TIR 
$ ssh root@xx.xxx.xx..x
```   
      

  1.2) Install:
      
  - Confirm that the shared storage is available. If you cant find it, make sure you followed step 8 (from stage 1):

      ```
      $ cd /shared
      ```
        
  - Execute the following to install libraries and dependencies

    ```
    $ sudo apt update && sudo apt install python3-venv
    $ sudo apt update && sudo apt install screen
    
    $ cd /shared
    $ python3 -m venv hack_env
    $ source hack_env/bin/activate
    $ pip3 install huggingface_hub
    $ pip install --upgrade pip
    
    $ pip install sgl-kernel --force-reinstall --no-deps
    
    $ pip install "sglang[all]>=0.4.2.post2" --find-links https://flashinfer.ai/whl/cu124/torch2.5/flashinfer/

    ```
  
   - Lets download the model now. This step will pre-download the model to shared file system. The model is large (1tb+) so wait for 20-25 mins 

     ``` 
     $ export HF_HOME=/shared/hf_home
      
     # your token can be found here https://huggingface.co/settings/tokens
     $ huggingface-cli login --token <your huggingface token>
      
     $ huggingface-cli download deepseek-ai/DeepSeek-R1
     ```
    
  1.3) Launch Master Server:  Start a new screen to launch sglang server.  Perform the same steps on terminal 2.

  ```
  $ screen  
  
  $ cd /shared 
  $ export HF_HOME=/shared/hf_home
  $ source /shared/hack_env/bin/activate
  $ export MASTER=`hostname`
  $ python3 -m sglang.launch_server --model-path deepseek-ai/DeepSeek-R1 --tp 16 --trust-remote-code --dist-init-addr $MASTER:20000 --nnodes 2 --node-rank 0
```

### Terminal 2 (Worker Node)
2.1) Login to master node (use instructions from TIR)

```
# get the exact instruction from TIR 
ssh root@xx.xxx.xx..x
```

2.2) Continue further only after 1.2 is completed

2.3) SSH in to Worker node:

```
export WORKER=`hostname | sed  's/master/worker/g'`
ssh $WORKER

```
2.4) Execute the steps to start sglang server on worker:

```
$ screen 
```

```
# the steps may look similar to master on first glance but copy these exactly as they are actually different. 

$ export HF_HOME=/shared/hf_cache
$ source /shared/hack_env/bin/activate
$ export MASTER=`hostname | sed  's/worker/master/g'`
$ python3 -m sglang.launch_server --model-path deepseek-ai/DeepSeek-R1 --tp 16 --trust-remote-code --dist-init-addr $MASTER:20000 --nnodes 2 --node-rank 1
```
    
### Terminal 3 (API Client for testing API)

