# Install with LLM-D

This guide provides step-by-step instructions for deploying the vLLM Semantic Router (vsr) in combination with [LLM-D](https://github.com/llm-d/llm-d). This will also illustrate a key design pattern namely use of the vsr as a model picker in combination with the use of LLM-D as endpoint picker.

A model picker provides the ability to route an LLM query to one of multiple LLM models that are entirely different from each other, whereas an endpoint picker selects one of multiple endpoints that each serve an equivalent model (and most often the exact same base model). Hence this deployment shows how vLLM Semantic Router in its role as a model picker is perfectly complementary to endpoint picker solutions such as LLM-D.

Since LLM-D has a number of deployment configurations some of which require a larger hardware setup we will demonstrate a baseline version of LLM-D  working in combination with vsr to introduce the core concepts. These same core concepts will also apply when using vsr with more complex LLM-D configurations and production grade well-lit paths as described in the LLM-D repo at [this link](https://github.com/llm-d/llm-d/tree/main/guides).

Also we will use LLM-D with Istio as the Inference Gateway in order to build on the steps and hardware setup from the [Istio deployment example](istio) documented in this repo. Istio is also commonly used as the default gateway for LLM-D with or without vsr.

## Architecture Overview

The deployment consists of:

- **vLLM Semantic Router**: Provides intelligent request routing and processing decisions to Envoy based Gateways
- **LLM-D**: Distributed Inference platform used for scaleout LLM inferencing with SOTA performance.
- **Istio Gateway**: Istio's implementation of Kubernetes Gateway API that uses an Envoy proxy under the covers
- **Gateway API Inference Extension**: Additional APIs to extend the Gateway API for Inference via ExtProc servers
- **Two instances of vLLM serving 1 model each**:  Example backend LLMs for illustrating semantic routing in this topology

## Prerequisites

Before starting, ensure you have the following tools installed:

- [Docker](https://docs.docker.com/get-docker/) - Container runtime
- [minikube](https://minikube.sigs.k8s.io/docs/start/) - Local Kubernetes
- [kind](https://kind.sigs.k8s.io/docs/user/quick-start/#installation) - Kubernetes in Docker
- [kubectl](https://kubernetes.io/docs/tasks/tools/) - Kubernetes CLI
- [istioctl](https://istio.io/latest/docs/ops/diagnostic-tools/istioctl/) - Istio CLI

We use minikube in the description below. As noted above, this guide builds upon the vsr + Istio [deployment guide](istio) from this repo hence will point to that guide for the common portions of documentation and add the incremental additional steps here.

As was the case for the Istio guide, you will need a machine that has GPU support with at least 2 GPUs to run this exercise so that we can deploy and test the use of vsr to do model routing between two different LLM base models.

## Step 1: Common Steps from Istio Guide

First, follow the steps documented in the [Istio guide](istio), to create a local minikube cluster.

## Step 2: Install Istio Gateway, Gateway API, Inference Extension CRDs

Install CRDs for the Kubernetes Gateway API, Gateway API Inference Extension, Istio Control plane and an instance of the Istio Gateway exactly as described in the [Istio guide](istio). Use the same version of Istio as documented in that guide. If you were following the LLM-D well-lit paths this part would be done by the Gateway provider Helm charts from the LLM-D repo. In this guide, we set these up manually to keep things common and reusable with the Istio guide from this repo. This will also help the reader understand the parts that are common between a GIE/EPP based deployment and an LLM-D based deployment and how vsr can be used in both cases.

If installed correctly you should see the api CRDs for gateway api and inference extension as well as pods running for the Istio gateway and Istiod using the commands shown below.

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

## Step 3: Deploy LLM models

Now deploy two LLM models similar to the [Istio guide](istio) documentation. Note from the manifest file names that these example commands are to be executed from the top folder of the repo. The counterpart of this step from the LLM-D deployment documentation is the setup of the LLM-D Model Service. To keep things simple, we do not need the LLM-D Model service for this guide.

```bash
kubectl create secret generic hf-token-secret --from-literal=token=$HF_TOKEN
```

```bash
# Create vLLM service running llama3-8b
kubectl apply -f deploy/kubernetes/istio/vLlama3.yaml
```

This may take several (10+) minutes the first time this is run to download the model up until the vLLM pod running this model is in READY state.  Similarly also deploy the second LLM (phi4-mini) and wait for several minutes until the pod is in READY state.

```bash
# Create vLLM service running phi4-mini
kubectl apply -f deploy/kubernetes/istio/vPhi4.yaml
```

At the end of this you should be able to see both your vLLM pods are READY and serving these LLMs using the command below. You should also see Kubernetes services exposing the IP/ port on which these models are being served. In the example below the llama3-8b model is being served via a kubernetes service with service IP of 10.108.250.109 and port 80.

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

## Step 4: Deploy InferencePools and LLM-D schedulers

LLM-D (and Kubernetes IGW) use an API resource called InferencePool alongwith a scheduler (referred to as the LLM-D inference scheduler and sometimes equivalently as EndPoint Picker/ EPP).

Deploy the provided manifests in order to create InferencePool and LLM-D inference schedulers corresponding to the 2 base models used in this exercise.

In order to show a full combination of model picking and endpoint picking, one would normally need at least 2 inferencepools with at least 2 endpoints per pool. Since that would require 4 instances of vllm serving pods and 4 GPUs in our exercise, that would require a more complex hardware setup. This guide deploys 1 model endpoint per each of the two InferencePools in order to show the core design of vsr's model picking working with and complementing LLM-D scheduler's endpoint picking.

```bash
# Create the LLM-D scheduler and InferencePool for the Llama3-8b model
kubectl apply -f deploy/kubernetes/llmd-base/inferencepool-llama.yaml
```

```bash
# Create the LLM-D scheduler and InferencePool for the phi4-mini model
kubectl apply -f deploy/kubernetes/llmd-base/inferencepool-phi4.yaml
```

## Step 5: Additional Istio config for LLM-D connection

Add DestinationRule to allow each EPP/ LLM-D scheduler to use ExtProc without TLS (current Istio limitation).

```bash
# Istio destinationrule for the Llama3-8b pool scheduler
kubectl apply -f deploy/kubernetes/llmd-base/dest-rule-epp-llama.yaml
```

```bash
# Istio destinationrule for the phi4-mini pool scheduler
kubectl apply -f deploy/kubernetes/llmd-base/dest-rule-epp-phi4.yaml
```

## Step 6: Update vsr config

Since this guide is based on using the same backend models as in the [Istio guide](istio), we will reuse the same vsr config as from that guide and hence you do not need to update the file deploy/kubernetes/istio/config.yaml. If you were using different backend models as part of the LLM-D deployment, you would need to update this file.

## Step 7: Deploy vLLM Semantic Router

Deploy the semantic router service with all required components:

```bash
# Deploy semantic router using Kustomize
kubectl apply -k deploy/kubernetes/istio/

# Wait for deployment to be ready (this may take several minutes for model downloads)
kubectl wait --for=condition=Available deployment/semantic-router -n vllm-semantic-router-system --timeout=600s

# Verify deployment status
kubectl get pods -n vllm-semantic-router-system
```

## Step 6: Additional Istio configuration for the VSR connection

Install the destinationrule and envoy filter needed for Istio gateway to use ExtProc based interface with vLLM Semantic router.

```bash
kubectl apply -f deploy/kubernetes/istio/destinationrule.yaml
kubectl apply -f deploy/kubernetes/istio/envoyfilter.yaml
```

## Step 7: Install gateway routes

Install HTTPRoutes in the Istio gateway. Note a difference here compared to the http routes used in the prior vsr + istio guide, here the backendRefs in the route matches based on point to the InferencePools which in turn point to the LLM-D schedulers for those pools instead of the backendRefs pointing to the vllm service endpoints of the models as was done in the [istio guide without llm-d](istio).

```bash
kubectl apply -f deploy/kubernetes/llmd-base/httproute-llama-pool.yaml
kubectl apply -f deploy/kubernetes/llmd-base/httproute-phi4-pool.yaml
```

## Step 8: Testing the Deployment

To expose the IP on which the Istio gateway listens to client requests from outside the cluster, you can choose any standard kubernetes  option for external load balancing. We tested our feature by [deploying and configuring metallb](https://metallb.universe.tf/installation/) into the cluster to be the LoadBalancer provider. Please refer to metallb documentation for installation procedures if needed. Finally, for the minikube case, we get the external url as shown below.

```bash
minikube service inference-gateway-istio --url
http://192.168.49.2:32293
```

Now we can send LLM prompts via curl to http://192.168.49.2:32293 to access the Istio gateway  which will then use information from vLLM semantic router to dynamically route to one of the two LLMs we are using as backends in this case. Use the port number that you get as output from your "minikube service" command in the curl examples below.

### Send Test Requests

Try the following cases with and without model "auto" selection to confirm that Istio + vsr together are able to route queries to the appropriate model. The query responses will include information about which model was used to serve that request.

Example queries to try include the following

```bash
# Model name llama3-8b provided explicitly, no model alteration, send to llama EPP for endpoint picking
curl http://192.168.49.2:32293/v1/chat/completions   -H "Content-Type: application/json"   -d '{
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
curl http://192.168.49.2:32293/v1/chat/completions   -H "Content-Type: application/json"   -d '{
        "model": "auto",
        "messages": [
          {"role": "user", "content": "Linux is said to be an open source kernel because "}
         ],
        "max_tokens": 100,
        "temperature": 0
      }'
```

```bash
# Model name phi4-mini provided explicitly, no model alteration, send to phi4-mini EPP for endpoint picking
curl http://192.168.49.2:32293/v1/chat/completions   -H "Content-Type: application/json"   -d '{
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
curl http://192.168.49.2:32293/v1/chat/completions   -H "Content-Type: application/json"   -d '{
        "model": "auto",
        "messages": [
          {"role": "user", "content": "2+2 is  "}
         ],
        "max_tokens": 100,
        "temperature": 0
      }'
```

## Troubleshooting

### Basic Pod Validation

If you have followed the above steps, you should see pods similar to below running READY state as a quick initial validation. These include the LLM model pods, Istio gateway pod, LLM-D/EPP scheduler pods, vsr pod and istiod controller pod as shown below. You should also see the InferencePools and HTTPRoute instances as shown below with status showing routes in resolved state.

```bash
$ kubectl get pods -n default
NAME                                           READY   STATUS    RESTARTS   AGE
inference-gateway-istio-6fc8864bfb-gbcz8       1/1     Running   0          14h
llama-8b-6558848cc8-wkkxn                      1/1     Running   0          3h26m
phi4-mini-7b94bc69db-rnpkj                     1/1     Running   0          17h
vllm-llama3-8b-instruct-epp-7f7ff88677-j7lst   1/1     Running   0          134m
vllm-phi4-mini-epp-6f5dd6bbb9-8pv27            1/1     Running   0          14h
```

```bash
$ kubectl get pods -n vllm-semantic-router-system
NAME                              READY   STATUS    RESTARTS   AGE
semantic-router-bf6cdd5b9-t5hpg   1/1     Running   0          5d23h
```

```bash
$ kubectl get pods -n istio-system
NAME                     READY   STATUS    RESTARTS   AGE
istiod-6f5ccc65c-vnbg5   1/1     Running   0          15h
```

```bash
$ kubectl get inferencepools
NAME                      AGE
vllm-llama3-8b-instruct   139m
vllm-phi4-mini            15h
```

```bash
$ kubectl get httproutes
NAME            HOSTNAMES   AGE
vsr-llama8b                 13h
vsr-phi4-mini               13h
```

```bash
$ kubectl get httproute vsr-llama8b -o yaml | grep -A 1 "reason: ResolvedRefs"
      reason: ResolvedRefs
      status: "True"
```

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

**Semantic router not responding or not routing correctly:**

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
- Test/ experiment with more complex LLM-D configurations and well-lit paths
- Set up monitoring and observability
- Implement authentication and authorization
- Scale the semantic router deployment for production workloads
