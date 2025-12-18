# vLLM Semantic Router with Local LLMs + Public Cloud/ Remote LLMs and LLM-D

This guide showcases a deployment in which vSR can selectively route to some local LLMs and some publicly hosted LLMs from cloud providers such as OpenAI. This guide builds upon the baseline vSR + Istio guide which is at [this link](../../istio/README.md) and the vSR + Istio + LLM-D guide which is at [this link](../llmd-base/README.md) and deploys a combination of vSR + Istio + LLM-D + local LLMs + public cloud LLMs in order to showcase full flexibility of deployment options. Remote/ cloud LLMs can be supported by following this guide with or without the LLM-D portion being present in the deployment.

## Architecture Overview

The deployment consists of:

- **vLLM Semantic Router (vSR)**: Provides intelligent request routing and processing decisions to Envoy based Gateways
- **LLM-D**: Distributed Inference platform used for scaleout LLM inferencing with SOTA performance.
- **Istio Gateway**: Istio's implementation of Kubernetes Gateway API that uses an Envoy proxy under the covers
- **Gateway API Inference Extension**: Additional APIs to extend the Gateway API for Inference via ExtProc servers
- **Two instances of vLLM serving the same local LLM**: Two replicas serving the same local LLM targeted by semantic routing in this topology
- **One instance of an LLM served by a public cloud**: The alternate backend LLM targeted by semantic routing in this topology servng a different LLM base model via a cloud service (we use an OpenAI.com hosted LLM in the guide below but the instructions apply for any similar hosted LLM service as long as you have a cloud account to access LLMs programmatically via an OpenAI compatible api).

## Prerequisites

Before starting, ensure you have the following tools installed:

- [Docker](https://docs.docker.com/get-docker/) - Container runtime
- [minikube](https://minikube.sigs.k8s.io/docs/start/) - Local Kubernetes
- [kind](https://kind.sigs.k8s.io/docs/user/quick-start/#installation) - Kubernetes in Docker
- [kubectl](https://kubernetes.io/docs/tasks/tools/) - Kubernetes CLI
- [istioctl](https://istio.io/latest/docs/ops/diagnostic-tools/istioctl/) - Istio CLI

We use minikube in the description below. As noted above, this guide builds upon the vsr + Istio [deployment guide]((../istio/README.md)) from this repo hence will point to that guide for the common portions of documentation and add the incremental additional steps here.

You will need a machine with at least 2 GPUs for this guide since we deploy two replicas of a local model in combination with 1 remotely hosted model. You can also run it on a machine with only 1 local GPU by scaling down the local replicas since the primary purpose of this guide is to illustrate routing to remotely hosted LLMs alongwith at least 1 local LLM.

## Step 1: Common Steps from Istio Guide

First, follow the steps documented in the [Istio guide](../istio/README.md), to create a local minikube cluster.

## Step 2: Install Istio Gateway, Gateway API, Inference Extension CRDs

Install CRDs for the Kubernetes Gateway API, Gateway API Inference Extension, Istio Control plane and an instance of the Istio Gateway exactly as described in the [Istio guide](../istio/README.md). You may also install istio using istioctl directly as described in the istio web site as long as the version is 1.28.0 or newer.

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
kubectl get gateway | grep istio
```

```bash
kubectl get pods -n istio-system
```

## Step 3: Deploy local LLM model

Now deploy one local LLM model. Note from the manifest file names that these example commands are to be executed from the top folder of the repo.

```bash
kubectl create secret generic hf-token-secret --from-literal=token=$HF_TOKEN
```

```bash
# Create vLLM service running llama3-8b
kubectl apply -f deploy/kubernetes/istio/vLlama3.yaml
```

This may take several (10+) minutes the first time this is run to download the model up until the vLLM pod running this model is in READY state. In this guide we will create two replicas of the same LLM instead of 1 replica each of two separate LLMs, hence scale this deployment to 2 and wait for both LLM pods to be in READY state.

```bash
# Create a 2nd replica of the same local LLM 
kubectl scale deploy llama-8b --replicas=2
```

At the end of this you should be able to see both your vLLM pods are READY and serving these LLMs using the command below.

```bash
# Verify that vLLM pods running the two LLMs are READY and serving  

kubectl get pods
NAME                                       READY   STATUS    RESTARTS   AGE
inference-gateway-istio-667659bd77-nwr8q   1/1     Running   0          7m16s
llama-8b-6558848cc8-966b8                  1/1     Running   0          90m
llama-8b-6558848cc8-nhbgv                  1/1     Running   0          141m
```

## Step 4: Deploy InferencePool and LLM-D scheduler

LLM-D (and Kubernetes IGW) use an API resource called InferencePool alongwith a scheduler (referred to as the LLM-D inference scheduler and sometimes equivalently as EndPoint Picker/ EPP).

Deploy the provided manifest in order to create an InferencePool and an LLM-D inference schedulers corresponding to the base model used in this exercise.

```bash
# Create the LLM-D scheduler and InferencePool for the Llama3-8b model  
kubectl apply -f deploy/kubernetes/llmd-base/inferencepool-llama.yaml
```

## Step 5: Additional Istio config for LLM-D connection

Add DestinationRule to allow the EPP/ LLM-D scheduler to use ExtProc without TLS (current Istio limitation).

```bash
# Istio destinationrule for the Llama3-8b pool scheduler  
kubectl apply -f deploy/kubernetes/llmd-base/dest-rule-epp-llama.yaml
```

## Step 6: Update vSR config

For this guide, we use an updated vSR config file which sets two endpoints, one for the local LLM service and a second for the openai backend model, specifically we use the "gpt-4o-mini" in the provided example config. Take a look at deploy/kubernetes/llmd-base/llmd+public/config.yaml, copy it over to the config.yaml in the istio folder so that we can reuse the other manifests and kustomize from that folder to deploy vSR with this config as shown below.

```bash
cp deploy/kubernetes/llmd-base/llmd+public-llm/config.yaml.openai deploy/kubernetes/istio/config.yaml
```

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

## Step 8: Additional Istio configuration for the vSR connection

Install the destinationrule and envoy filter needed for Istio gateway to use ExtProc based interface with vLLM Semantic router.

```bash
kubectl apply -f deploy/kubernetes/istio/destinationrule.yaml
kubectl apply -f deploy/kubernetes/istio/envoyfilter.yaml
```

## Step 9: Create a K8S Service and an Istio ServiceEntry to represent the OpenAI target

vSR's HTTPRoute will need a Kubernetes service representation for the OpenAI connection and since this is an external service, also need an Istio ServiceEntry representation. Set these up using the provided anifests.

```bash
kubectl apply -f deploy/kubernetes/llmd-base/llmd+public-llm/svc-openai.yaml
kubectl apply -f deploy/kubernetes/llmd-base/llmd+public-llm/svc-entry-openai.yaml
```

## Step 10: Set up an Istio DestinationRule to upgrade the backend OpenAI connection to TLS

OpenAI will require TLS  on its connection, hence setup a DestinationRule to use TLS on the backend connection to OpenAI even if the front end client connection comes in without TLS.

```bash
kubectl apply -f deploy/kubernetes/llmd-base/llmd+public-llm/dest-rule-openai.yaml
```

## Step 11: Set up and check API account credentials for OpenAI api access

In order to use the OpenAI API programmatically over the internet, you will need an OpenAI developer account with credentials that allow you to make api calls. Once registered with OpenAI, store your api key into your local environment and perform a manual curl test to access the OpenAI api with an LLM query to the same model to confirm that your account and credentials are setup correctly and there are no access issues. Perform the manual access via an LLM query to the same model that we have setup in our vSR config earlier (the "gpt-40-mini" model in our case).  A valid LLM response indicates all is well with the OpenAI account and path and it can be added to the main deployment in the following step.

```bash
## Once registered, confirm that you have your OpenAI key in your env. 
## It should look something like "sk-xxxx"
echo $OPENAI_API_KEY
```

```bash
## Perform a manual curl to access the gpt-4o-mini model via the OpenAI api using the key from your env
##  
curl https://api.openai.com/v1/chat/completions   -H "Content-Type: application/json"   -H "Authorization: Bearer $OPENAI_API_KEY"   -d '{
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "2+2 is "}]
  }'
```

## Step 12: Move the OpenAI api key into Kubernetes and Istio Env

First create a Kubernetes secret using the OpenAI api key from the environment and then move it into the Istio-proxy container environment as shown next.

```bash
##  Create a Kubernetes secret for the openai api key
kubectl create secret generic openai-api-key \
  --from-literal=OPENAI_API_KEY="$OPENAI_API_KEY"
```

```bash
## Make this secret available inside the Istio Envoy container
kubectl patch deployment inference-gateway-istio   --type='json'   -p='[
    {"op": "add", "path": "/spec/template/spec/containers/0/env/-",
     "value": {"name": "OPENAI_API_KEY", "valueFrom": {"secretKeyRef": {"name": "openai-api-key", "key": "OPENAI_API_KEY"}}}
    }
  ]'
```

```bash
## Confirm that the secret is available inside the Istio/Envoy container .. you should see "sk-xxx"
kubectl exec -it deploy/inference-gateway-istio -- printenv | grep OPENAI_API_KEY
```

## Step 13: Patch the OPENAI_API_KEY into the HTTPRoute for OpenAI

Patch the OPEN_AI_API_KEY from your environment into a template file to generate the manifest for the HTTPRoute representing the OpenAI target. Note that you can skip step 12 by doing this step but for now we also listed step 12 in case you have other automation options for generating the httproute manifest while templating in the value of the OPENAI_API_KEY.

```bash
## Patch the OPENAI_API_KEY into the template to create the httproute manifest file 
sed "s/{{OPENAI_API_KEY}}/$OPENAI_API_KEY/g" deploy/kubernetes/llmd-base/llmd+public-llm/httproute-openai.template  > deploy/kubernetes/llmd-base/llmd+public-llm/httproute-openai.yaml
```

## Step 14: Create HTTPRoutes for Local LLM and for the OpenAI target

Now deploy the HTTPRoute manifest for the openai route destination. In the manifest note again that we match on the contents of the x-selected-model and also setup the injection of the OpenAI api key as a bearer token for enabling the access into OpenAI api for this route. For the local LLM we use a route similar to the llm-d guide since we want the prompt query to also get routed via the inferencepool and LLM-D scheduler for the Llama pool which will then pick one of the multiple endpoints in the pool serving the Llama LLM in this example.

```bash
##  HTTpRoute for OpenAI
kc apply -f deploy/kubernetes/llmd-base/llmd+public-llm/httproute-openai.yaml
```

```bash
##  HttpRoute for the local llama llm as before 
##  Note that we want the local routes to go through the endpoint picker for the llama pool 
kubectl apply -f deploy/kubernetes/llmd-base/httproute-llama-pool.yaml
```

## Step 15: Testing the Deployment

To expose the IP on which the Istio gateway listens to client requests from outside the cluster, you can choose any standard kubernetes  option for external load balancing. We tested our feature by [deploying and configuring metallb](https://metallb.universe.tf/installation/) into the cluster to be the LoadBalancer provider. Please refer to metallb documentation for installation procedures if needed. Finally, for the minikube case, we get the external url as shown below.

```bash
minikube service inference-gateway-istio --url
http://192.168.49.2:31275
```

Now we can send LLM prompts via curl to <http://192.168.49.2:31275> to access the Istio gateway  which will then use information from vLLM semantic router to dynamically route to one of the two LLMs we are using as backends in this case. Use the port number that you get as output from your "minikube service" command when you try the curl examples below.

### Send Test Requests

Try the following cases with and without model "auto" selection to confirm that Istio + vSR + llm-d together are able to route queries to the appropriate model by combining model picking and endpoint picking. The query responses will include information about which model was used to serve that request.

Example queries to try include the following

```bash
# Model name llama3-8b provided explicitly, no model alteration, routed via llama EPP for endpoint picking  
curl http://192.168.49.2:31275/v1/chat/completions   -H "Content-Type: application/json"   -d '{
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
curl http://192.168.49.2:31275/v1/chat/completions   -H "Content-Type: application/json"   -d '{
        "model": "auto",
        "messages": [
          {"role": "user", "content": "Linux is said to be an open source kernel because "}
         ],
        "max_tokens": 100,
        "temperature": 0
      }'
```

```bash
# Model name gpt-4o-mini provided explicitly, no model alteration, routed to OpenAI
# Confirmed response came from Openai via certain content fields and verifying OpenAI token usage   
# Some fields from the LLM response that you will get which show that it came from OpenAI include
# There will be a field similar to "model": "gpt-4o-mini-2024-07-18" indicating a specific version of
# gpt-4o-mini was used to respond.
# There will be a field similar to "id": "chatcmpl-xxxx" which is only inserted by OpenAI
# There will be a field similar to "system_fingerprint": "fp_xxxx" which is a fingerprint that 
# only OpenAI provides. Refer to the latest OpenAI API docs for the model you are testing with
#
curl http://192.168.49.2:31275/v1/chat/completions   -H "Content-Type: application/json"   -d '{
        "model": "gpt-4o-mini",
        "messages": [
          {"role": "user", "content": "2+2 is  "}
         ],
        "max_tokens": 100,
        "temperature": 0
      }'
```

```bash
# Model name set to "auto", should categorize to "math" & route to OpenAI/ gpt-4o-mini 
# Confirm the response comes from openai using similar checks as the previous example above. 
curl http://192.168.49.2:31275/v1/chat/completions   -H "Content-Type: application/json"   -d '{
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
$ kubectl get pods
NAME                                                   READY   STATUS    RESTARTS   AGE
inference-gateway-istio-846b999d8c-kv5gp               1/1     Running   0          116m
llama-8b-6558848cc8-966b8                              1/1     Running   0          5h32m
llama-8b-6558848cc8-nhbgv                              1/1     Running   0          6h23m
llm-d-inference-scheduler-llama3-8b-74854dcdf6-k6z2m   1/1     Running   0          3h59m
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
```

```bash
$ kubectl get httproutes
NAME            HOSTNAMES   AGE
vsr-llama8b                 43m
vsr-openai-g4               50m
```

```bash
$ kubectl get httproute vsr-llama8b -o yaml | grep -A 1 "reason: ResolvedRefs"
      reason: ResolvedRefs
      status: "True"
```

```bash
$ kubectl get httproute vsr-openai-g4 -o yaml | grep -A 1 "reason: ResolvedRefs"
      reason: ResolvedRefs
      status: "True"
```

Also as noted previously in Step 11 verify your OpenAI account credentials and api access separately prior to accessing via the vSR + Istio setup.

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
- Test/ experiment with different public hosted models and model providers
- Test/ experiment with more complex LLM-D configurations and well-lit paths
- Set up monitoring and observability
- Implement authentication and authorization
- Scale the semantic router deployment for production workloads
