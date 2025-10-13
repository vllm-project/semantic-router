# vLLM Semantic Router as ExtProc server for Istio Gateway 

This guide provides step-by-step instructions for deploying the vLLM Semantic Router (vsr) with Istio Gateway on Kubernetes. Istio Gateway uses Envoy under the covers so it is possible to use vsr with it. There are multiple topologies possible to combine Istio Gateway with vsr. This document describes one of the common topologies. 
 
## Architecture Overview

The deployment consists of:

- **vLLM Semantic Router**: Provides intelligent request routing and classification
- **Istio Gateway**: Istio Gateway that uses an Envoy proxy under the covers
- **Gateway API Inference Extension**: Additional control and data plane for endpoint picking that can optionally attach to the same Istio gateway as vLLM Semantic Router. 
- **Two instances of vLLM serving 1 model each**:  Example backend LLMs for illustrating semantic routing in this topology

## Prerequisites

Before starting, ensure you have the following tools installed:

- [Docker](https://docs.docker.com/get-docker/) - Container runtime
- [minikube](https://minikube.sigs.k8s.io/docs/start/) - Local Kubernetes 
- [kind](https://kind.sigs.k8s.io/docs/user/quick-start/#installation) - Kubernetes in Docker
- [kubectl](https://kubernetes.io/docs/tasks/tools/) - Kubernetes CLI

Either minikube or kind works to deploy a local kubernetes cluster needed for this exercise so you only need one of these two. We use minikube in the description below but the same steps should work with a Kind cluster once the cluster is created in Step 1. 

We will also deploy two different LLMs in this exercise to illustrate the semantic routing and model routing function more clearly so you ideally you should run this on a machine that has GPU support to run the two models used in this exercise and adequate memory and storage for these models. You can also use equivalent steps on a smaller server that runs smaller LLMs on a CPU based server without GPUs.

## Step 1: Create Minikube Cluster

Create a local Kubernetes cluster via minikube (or equivalently via Kind). 

```bash
# Create cluster  
$ minikube start \
    --driver docker \
    --container-runtime docker \
    --gpus all \
    --memory no-limit \
    --cpus no-limit

# Verify cluster is ready
$ kubectl wait --for=condition=Ready nodes --all --timeout=300s
```

## Step 2: Deploy LLM models service 

As noted earlier in this exercise we deploy two LLMs viz. a llama3-8b model (meta-llama/Llama-3.1-8B-Instruct) and a phi4-mini model (microsoft/Phi-4-mini-instruct). In this exercise we chose to serve these models using two separate instances of the [vLLM inference server](https://docs.vllm.ai/en/latest/) running in the default namespace of the kubernetes cluster. For this exercise you may choose to use any inference server to serve these models but we have provided manifests to run these in vLLM containers as a reference.

```bash
# Create vLLM service running llama3-8b  
kubectl apply -f deploy/kubernetes/istio/vLlama3.yaml
```

This may take several (10+) minutes the first time this is run to download the model up until the vLLM pod running this model is in READY state.  Similarly also deploy the second LLM (phi4-mini) and wait for several minutes until the pod is in READY state..

```bash
# Create vLLM service running phi4-mini  
kubectl apply -f deploy/kubernetes/istio/vPhi4.yaml
```

At the end of this you should be able to see both your vLLM pods are READY and serving these LLMs using the command below. You should also see Kubernetes services explosing the IP/ port on which these models are being served. In th example below the llama3-8b model is being served via a kubernetes service with service IP of 10.108.250.109 and port 80.

```bash
# Verify that vLLM pods running the two LLMs are READY and serving  

kubectl get pods
NAME                                           READY   STATUS    RESTARTS     AGE
llama-8b-57b95475bd-ph7s4                      1/1     Running   0            9d
phi4-mini-887476b56-74twv                      1/1     Running   0            9d

# View the IP/port of the Kubernetes services on which these models are being served
 
kubectl get service
NAME                                  TYPE           CLUSTER-IP       EXTERNAL-IP      PORT(S)                        AGE
kubernetes                            ClusterIP      10.96.0.1        <none>           443/TCP                        36d
llama-8b                              ClusterIP      10.108.250.109   <none>           80/TCP                         18d
phi4-mini                             ClusterIP      10.97.252.33     <none>           80/TCP                         9d
```

## Step 3: Update vsr config if needed

The file deploy/kubernetes/istio/config.yaml will get used to configure vsr when it is installed in the next step. The example config file provided already in this repo should work f you use the same LLMs as in this exercise but you can choose to play with this config to enable or disable individual vsr features. Ensure that your vllm_endpoints in the file match the ip/ port of the llm services you are running. It is usually good to start with basic features of vsr such as prompt classification and model routing before experimenting with other features as described elsewhere in the vsr documentation. 

## Step 4: Deploy vLLM Semantic Router

Deploy the semantic router service with all required components:

```bash
# Deploy semantic router using Kustomize
kubectl apply -k deploy/kubernetes/istio/

# Wait for deployment to be ready (this may take several minutes for model downloads)
kubectl wait --for=condition=Available deployment/semantic-router -n vllm-semantic-router-system --timeout=600s

# Verify deployment status
kubectl get pods -n vllm-semantic-router-system
```

## Step 5: Install Istio Gateway, Gateway API, Inference Extension 

We will use a recent build of Istio for this exercise so that we have the option of also using  the v1.0.0 GA version of the Gateway API Inference Extension  CRDs and EPP functionality.

Follow the procedures described in the Gateway API [Inference Extensions documentation](https://gateway-api-inference-extension.sigs.k8s.io/guides/) to deploy the 1.28 (or newer) version of Istio Gateway, the Kubernetes Gateway API CRDs and the Gateway API Inference Extension v1.0.0. Do not install any of the HTTPRoute resources from that guide however, just use it to deploy the Istio gateway and CRDs.  If installed correctly you should see the api CRDs for gateway api and inference extension as well as pods running for the Istio gateway and Istiod using the commands shown below.

```bash
kubectl get crds | grep gateway
```

```bash
kubectl get crds | grep inference
```

```bash
kubectl get pods | grep istio
```

```bash
kubectl get pods -n istio-system
```

## Step 6: Install additional Istio configuration

Install the destinationrule and envoy filter needed for Istio gateway to use ExtProc based interface with vLLM Semantic router

```bash
kubectl apply -f deploy/kubernetes/istio/destinationrule.yaml
kubectl apply -f deploy/kubernetes/istio/envoyfilter.yaml
```

## Step 7: Install gateway routes 

Install HTTPRoutes in the Istio gateway.

```bash
kubectl apply -f deploy/kubernetes/istio/httproute-llama3-8b.yaml
kubectl apply -f deploy/kubernetes/istio/httproute-phi4-mini.yaml
```
 
## Testing the Deployment
To expose the IP on which the Istio gateway listens to client requests from outside the cluster, you can choose any standard kubernetes  option for external load balancing. We tested our feature by [deploying and configuring metallb](https://metallb.universe.tf/installation/) into the cluster to be the LoadBalancer provider. Please refer to metallb documentation for installation procedures if needed. Finally, for the minikube case, we get the external url as shown below.

```bash
minikube service inference-gateway-istio --url
http://192.168.49.2:30913
```

Now we can send LLM prompts via curl to http://192.168.49.2:30913 to access the Istio gateway  which will then use information from vLLM semantic router to dynamically route to one of the two LLMs we are using as backends in this case.

### Send Test Requests

Try the following cases with and without model "auto" selection to confirm that Istio + vsr together are able to route queries to the appropriate model. The query responses will include information about which model was used to serve that request.

Example queries to try include the following

```bash
# Model name llama3-8b provided explicitly, should route to this backend 
curl http://192.168.49.2:30913/v1/chat/completions   -H "Content-Type: application/json"   -d '{
        "model": "llama3-8b",
        "messages": [
          {"role": "user", "content": "Linux is said to be an open source kernel because "}
         ],
        "max_tokens": 100,
        "temperature": 0
      }'
```

```bash
# Model name set to "auto", should categorize to "computer science" & route to llama3-8b 
curl http://192.168.49.2:30913/v1/chat/completions   -H "Content-Type: application/json"   -d '{
        "model": "auto",
        "messages": [
          {"role": "user", "content": "Linux is said to be an open source kernel because "}
         ],
        "max_tokens": 100,
        "temperature": 0
      }'
```

```bash
# Model name phi4-mini provided explicitly, should route to this backend 
curl http://192.168.49.2:30913/v1/chat/completions   -H "Content-Type: application/json"   -d '{
        "model": "phi4-mini",
        "messages": [
          {"role": "user", "content": "2+2 is  "}
         ],
        "max_tokens": 100,
        "temperature": 0
      }'
```

```bash
# Model name set to "auto", should categorize to "math" & route to phi4-mini 
curl http://192.168.49.2:30913/v1/chat/completions   -H "Content-Type: application/json"   -d '{
        "model": "auto",
        "messages": [
          {"role": "user", "content": "2+2 is  "}
         ],
        "max_tokens": 100,
        "temperature": 0
      }'
```

## Troubleshooting

### Common Issues

**Gateway/ Front end not working:**

```bash
# Check istio gateway status
kubectl get gateway 

# Check istio gw service status
kubectl get svc inference-gateway-istio

# Check Istio's Envoy logs
kubectl logs deploy/inference-gateway-istio -c istio-proxy
```

**Semantic router not responding:**

```bash
# Check semantic router pod
kubectl get pods -n vllm-semantic-router-system

# Check semantic router service 
kubectl get svc -n vllm-semantic-router-system

# Check semantic router logs
kubectl logs -n vllm-semantic-router-system deployment/semantic-router
```

## Cleanup

```bash

# Remove semantic router
kubectl delete -k deploy/kubernetes/istio/

# Remove Istio 
istioctl uninstall --purge

# Remove LLMs
kubectl delete -f deploy/kubernetes/istio/vLlama3.yaml
kubectl delete -f deploy/kubernetes/istio/vPhi4.yaml

# Stop minikube cluster 
minikube stop

# Delete minikube cluster 
minikube delete
```

## Next Steps

- Test/ experiment with different features of vLLM Semantic Router
- Additional use cases/ topologies with Istio Gateway (including with EPP and LLM-D)
- Set up monitoring and observability
- Implement authentication and authorization
- Scale the semantic router deployment for production workloads
