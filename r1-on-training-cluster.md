# DeepSeek R1 (Inference) on TIR Training Cluster

This document covers steps to deploy Deepseek R1 on training cluster for inference. This approach is quite useful when experimenting with the model or generating synthetic data.

Stages:
1) Create a Training Cluster 
2) Open terminal on your laptop and Login to Master node. We will need 3 terminal sessions, so you can start 3 at the start of this stage 
3) Install sglang and other libraries from the master node 
4) Start the sglang server on master and worker node 
5) Test R1 endpoint 


### Stage 1
1) Go to TIR Dashboard
2) Navigate to Training cluster from left-hand sidebar
3) Click on `create cluster` and choose 16xH100 plan. For this experiment, we will also need a shared file system (SFS). So when prompted create a SFS of atleast 1TB. Wait for the cluster to be ready (Assigned status)
4) Click on Deployments tab 
5) Create `Create Deployments`
6) Choose `Pytorch Distributed` as framework and make sure your cluster (Created in 1.3) is selected
7) Click `next` and add/select your SSH key to be added to login (master) node
8) Complete the flow to start a deployment. and wait for the deployment to show `RUNNING` state
9) When deployment is in running state, you would see the list of nodes in `worker` tab on the same page. Locate master node in the table and click on `connect` icon
10) Use the instructions from right-hand sidebar drawer to log into master node. use the same instruction (e.g. ssh admin@xx.xx.x..x) to login from 3 terminal session.


