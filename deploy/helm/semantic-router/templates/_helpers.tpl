{{/*
Expand the name of the chart.
*/}}
{{- define "semantic-router.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "semantic-router.fullname" -}}
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
{{- define "semantic-router.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "semantic-router.labels" -}}
helm.sh/chart: {{ include "semantic-router.chart" . }}
{{ include "semantic-router.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "semantic-router.selectorLabels" -}}
app.kubernetes.io/name: {{ include "semantic-router.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app: semantic-router
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "semantic-router.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "semantic-router.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Get the namespace
*/}}
{{- define "semantic-router.namespace" -}}
{{- if .Values.global.namespace }}
{{- .Values.global.namespace }}
{{- else }}
{{- .Release.Namespace }}
{{- end }}
{{- end }}

{{/*
Get the PVC name
*/}}
{{- define "semantic-router.pvcName" -}}
{{- if .Values.persistence.existingClaim }}
{{- .Values.persistence.existingClaim }}
{{- else }}
{{- printf "%s-models" (include "semantic-router.fullname" .) }}
{{- end }}
{{- end }}

{{/*
Resolve semantic cache Redis host for dependency-based deployments.
*/}}
{{- define "semantic-router.semanticCache.redisHost" -}}
{{- if .Values.dependencies.semanticCache.redis.host -}}
{{- .Values.dependencies.semanticCache.redis.host -}}
{{- else -}}
{{- printf "%s-semantic-cache-redis-master" .Release.Name -}}
{{- end -}}
{{- end }}

{{/*
Resolve semantic cache Milvus host for dependency-based deployments.
*/}}
{{- define "semantic-router.semanticCache.milvusHost" -}}
{{- if .Values.dependencies.semanticCache.milvus.host -}}
{{- .Values.dependencies.semanticCache.milvus.host -}}
{{- else -}}
{{- printf "%s-semantic-cache-milvus" .Release.Name -}}
{{- end -}}
{{- end }}

{{/*
Resolve Response API Milvus address for dependency-based deployments.
*/}}
{{- define "semantic-router.responseApi.milvusAddress" -}}
{{- $host := .Values.dependencies.responseApi.milvus.host | default (printf "%s-semantic-cache-milvus" .Release.Name) -}}
{{- printf "%s:%d" $host (int .Values.dependencies.responseApi.milvus.port) -}}
{{- end }}

{{/*
Resolve Jaeger OTLP endpoint for dependency-based deployments.
*/}}
{{- define "semantic-router.jaeger.otlpEndpoint" -}}
{{- $serviceName := .Values.dependencies.observability.jaeger.serviceName | default (printf "%s-jaeger" .Release.Name) -}}
{{- printf "%s:%d" $serviceName (int .Values.dependencies.observability.jaeger.otlpGrpcPort) -}}
{{- end }}
