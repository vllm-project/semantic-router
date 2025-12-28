## ❌ E2E Integration Test Report - nginx

**Status:** 🔴 **FAILED**
**Profile:** `nginx`
**Duration:** 13m35s
**Cluster:** `semantic-router-e2e`

### 📊 Test Results

| Metric | Value |
|--------|-------|
| Exit Code | `1` |
| Total Tests | 5 |
| Passed Tests | 3 |
| Failed Tests | 2 |
| Success Rate | 60.0% |

### 🔧 Cluster Statistics

| Metric | Value |
|--------|-------|
| Total Pods | 14 |
| Running Pods | 14 |
| Pending Pods | 0 |
| Failed Pods | 0 |
| Namespaces | 8 |


### 📝 Test Cases

- ✅ **nginx-proxy-health** (2.041747397s)
  <details>
  <summary>Details</summary>

  | Metric | Value |
  |--------|-------|
  | endpoint | /v1/health |
  | status_code | 200 |
  | passed | true |

  </details>
- ❌ **nginx-proxy-normal-request** (1m0.542235402s) - Error: `backend error 502: llm-katan is unhealthy or unreachable: <html>
<head><title>502 Bad Gateway</title></head>
<body>
<center><h1>502 Bad Gateway</h1></center>
<hr><center>nginx</center>
</body>
</html>
`
- ✅ **nginx-proxy-jailbreak-block** (2.221443913s)
  <details>
  <summary>Details</summary>

  | Metric | Value |
  |--------|-------|
  | jailbreak_type | jailbreak |
  | block_reason | jailbreak_detected |
  | passed | true |
  | endpoint | /v1/chat/completions |
  | status_code | 403 |
  | jailbreak_blocked | true |

  </details>
- ✅ **nginx-proxy-pii-block** (2.399467675s)
  <details>
  <summary>Details</summary>

  | Metric | Value |
  |--------|-------|
  | status_code | 403 |
  | pii_violation | true |
  | pii_types | B-US_SSN,I-PHONE_NUMBER |
  | block_reason | pii_policy_violation |
  | passed | true |
  | endpoint | /v1/chat/completions |

  </details>
- ❌ **nginx-proxy-classification** (1m2.025777184s) - Error: `failed to send request: Post "http://localhost:8080/v1/chat/completions": context deadline exceeded (Client.Timeout exceeded while awaiting headers)`

### 📦 Deployed Components

<details>
<summary>Namespace: ingress-nginx (1 pods)</summary>

| Pod | Phase | Ready | Restarts | Age |
|-----|-------|-------|----------|-----|
| ingress-nginx-controller-5d4d7ddd69-t56f2 | Running | 1/1 | 0 | 6m39s |

</details>

<details>
<summary>Namespace: kube-system (10 pods)</summary>

| Pod | Phase | Ready | Restarts | Age |
|-----|-------|-------|----------|-----|
| coredns-66bc5c9577-gpd8w | Running | 1/1 | 0 | 12m59s |
| coredns-66bc5c9577-zb9mx | Running | 1/1 | 0 | 12m59s |
| etcd-semantic-router-e2e-control-plane | Running | 1/1 | 0 | 13m17s |
| kindnet-qwmg5 | Running | 1/1 | 0 | 13m12s |
| kindnet-sg57x | Running | 1/1 | 0 | 13m9s |
| kube-apiserver-semantic-router-e2e-control-plane | Running | 1/1 | 0 | 13m17s |
| kube-controller-manager-semantic-router-e2e-control-plane | Running | 1/1 | 0 | 13m17s |
| kube-proxy-5jbwm | Running | 1/1 | 0 | 13m12s |
| kube-proxy-q8hqp | Running | 1/1 | 0 | 13m9s |
| kube-scheduler-semantic-router-e2e-control-plane | Running | 1/1 | 0 | 13m17s |

</details>

<details>
<summary>Namespace: llm-katan (1 pods)</summary>

| Pod | Phase | Ready | Restarts | Age |
|-----|-------|-------|----------|-----|
| llm-katan-7499467f44-zcwfn | Running | 1/1 | 0 | 6m14s |

</details>

<details>
<summary>Namespace: local-path-storage (1 pods)</summary>

| Pod | Phase | Ready | Restarts | Age |
|-----|-------|-------|----------|-----|
| local-path-provisioner-576879948-vqv8w | Running | 1/1 | 0 | 12m56s |

</details>

<details>
<summary>Namespace: vllm-semantic-router-system (1 pods)</summary>

| Pod | Phase | Ready | Restarts | Age |
|-----|-------|-------|----------|-----|
| semantic-router-d586cbb7-q59dr | Running | 1/1 | 0 | 3m38s |

</details>

### 🔍 Debugging

**Test failed!** Check the details above:

- Review failed test cases in the Test Cases section
- Check pod status in the Deployed Components section
- Verify all pods are in Running state

