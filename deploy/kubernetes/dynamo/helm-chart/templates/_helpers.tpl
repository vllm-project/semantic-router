{{/*
Expand the name of the chart.
*/}}
{{- define "dynamo-vllm.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "dynamo-vllm.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "dynamo-vllm.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "dynamo-vllm.labels" -}}
helm.sh/chart: {{ include "dynamo-vllm.chart" . }}
{{ include "dynamo-vllm.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "dynamo-vllm.selectorLabels" -}}
app.kubernetes.io/name: {{ include "dynamo-vllm.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the namespace
*/}}
{{- define "dynamo-vllm.namespace" -}}
{{- default .Values.global.namespace .Release.Namespace }}
{{- end }}

{{/*
Create image reference
*/}}
{{- define "dynamo-vllm.image" -}}
{{- printf "%s:%s" .Values.image.repository .Values.image.tag }}
{{- end }}

{{/*
Generate the vLLM worker command
Each worker runs exactly ONE model
Parameters:
  - .worker: Worker configuration (contains model config directly)
  - .index: Worker index for defaults
*/}}
{{- define "dynamo-vllm.workerCommand" -}}
{{- $worker := .worker -}}
{{- $index := .index | default 0 -}}
{{- $model := $worker.model -}}
{{/* GPU device defaults to index + 1 (frontend uses GPU 0, workers start from GPU 1) */}}
{{- $defaultGpu := add $index 1 -}}
{{- $gpuDevice := $worker.gpuDevice | default $defaultGpu -}}
{{/* Worker type defaults to "both" (combined prefill+decode). Set explicitly for disaggregated mode. */}}
{{- $workerType := $worker.workerType | default "both" -}}
{{- $cmd := printf "sleep 15 && export CUDA_VISIBLE_DEVICES=%d && export LD_LIBRARY_PATH=/nvidia-driver-libs:/usr/local/cuda/lib64:$LD_LIBRARY_PATH && python3 -m dynamo.vllm --model %s" (int $gpuDevice) $model.path -}}
{{- if and $model $model.tensorParallelSize -}}
{{- $cmd = printf "%s --tensor-parallel-size %d" $cmd (int $model.tensorParallelSize) -}}
{{- end -}}
{{/* enforceEager defaults to true to avoid Triton compilation issues */}}
{{- $enforceEager := true -}}
{{- if and $model (hasKey $model "enforceEager") -}}
{{- $enforceEager = $model.enforceEager -}}
{{- end -}}
{{- if $enforceEager -}}
{{- $cmd = printf "%s --enforce-eager" $cmd -}}
{{- end -}}
{{- if and $model $model.maxModelLen -}}
{{- $cmd = printf "%s --max-model-len %d" $cmd (int $model.maxModelLen) -}}
{{- end -}}
{{- if and $model $model.gpuMemoryUtilization -}}
{{- $cmd = printf "%s --gpu-memory-utilization %.2f" $cmd $model.gpuMemoryUtilization -}}
{{- end -}}
{{- if eq $workerType "prefill" -}}
{{- $cmd = printf "%s --is-prefill-worker" $cmd -}}
{{- end -}}
{{/* Connector defaults to "null" to avoid NIXL/UCX issues */}}
{{- $connector := $worker.connector | default "null" -}}
{{- $cmd = printf "%s --connector %s" $cmd $connector -}}
{{- if and $model $model.extraArgs -}}
{{- range $model.extraArgs -}}
{{- $cmd = printf "%s %s" $cmd . -}}
{{- end -}}
{{- end -}}
{{- $cmd -}}
{{- end }}

{{/*
Generate the frontend command
*/}}
{{- define "dynamo-vllm.frontendCommand" -}}
{{- $gpuDevice := .gpuDevice -}}
{{- $routerMode := .routerMode -}}
{{- $httpPort := .httpPort -}}
{{- printf "sleep 15 && export CUDA_VISIBLE_DEVICES=%d && export LD_LIBRARY_PATH=/nvidia-driver-libs:/usr/local/cuda/lib64:$LD_LIBRARY_PATH && python3 -m dynamo.frontend --http-port %d --router-mode %s" (int $gpuDevice) (int $httpPort) $routerMode -}}
{{- end }}

{{/*
Generate worker name from worker config
Parameters: dict with "worker", "index", and "workerType"
*/}}
{{- define "dynamo-vllm.workerName" -}}
{{- $worker := .worker -}}
{{- $index := .index -}}
{{- $workerType := .workerType | default "both" -}}
{{/* Only include type prefix for prefill/decode, not for "both" */}}
{{- $defaultName := "" -}}
{{- if or (eq $workerType "prefill") (eq $workerType "decode") -}}
{{- $defaultName = printf "%sworker%d" $workerType (int $index) -}}
{{- else -}}
{{- $defaultName = printf "worker%d" (int $index) -}}
{{- end -}}
{{- $name := $worker.name | default $defaultName -}}
{{- $name | replace "_" "-" | lower | trunc 63 | trimSuffix "-" -}}
{{- end }}

{{/*
Generate common environment variables for all Dynamo components
*/}}
{{- define "dynamo-vllm.commonEnv" -}}
- name: ETCD_ENDPOINTS
  value: {{ .Values.platform.etcd.endpoints | quote }}
- name: NATS_URL
  value: {{ .Values.platform.nats.url | quote }}
- name: NATS_SERVER
  value: {{ .Values.platform.nats.server | quote }}
- name: DYN_SYSTEM_ENABLED
  value: "true"
- name: DYN_SYSTEM_PORT
  value: "9090"
- name: DYN_LOG
  value: {{ .Values.global.logLevel | quote }}
- name: LD_LIBRARY_PATH
  value: "/nvidia-driver-libs:/usr/local/cuda/lib64"
- name: NVIDIA_DRIVER_CAPABILITIES
  value: "compute,utility"
{{- if .Values.huggingface.existingSecret }}
- name: HF_TOKEN
  valueFrom:
    secretKeyRef:
      name: {{ .Values.huggingface.existingSecret }}
      key: {{ .Values.huggingface.existingSecretKey | default "HF_TOKEN" }}
{{- else if .Values.huggingface.token }}
- name: HF_TOKEN
  value: {{ .Values.huggingface.token | quote }}
{{- end }}
{{- end }}

{{/*
Generate common volume mounts
*/}}
{{- define "dynamo-vllm.commonVolumeMounts" -}}
- name: nvidia-driver-libs
  mountPath: /nvidia-driver-libs
  readOnly: true
- name: dev
  mountPath: /dev
{{- end }}

{{/*
Generate common volumes
*/}}
{{- define "dynamo-vllm.commonVolumes" -}}
- name: nvidia-driver-libs
  hostPath:
    path: {{ .Values.volumes.nvidiaDriverLibs.hostPath }}
- name: dev
  hostPath:
    path: {{ .Values.volumes.dev.hostPath }}
{{- end }}
