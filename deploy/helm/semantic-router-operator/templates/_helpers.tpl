{{/*
Expand the name of the chart.
*/}}
{{- define "semantic-router-operator.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "semantic-router-operator.fullname" -}}
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
{{- define "semantic-router-operator.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels. The app.kubernetes.io/component and part-of labels match the
labels the kustomize install applies (deploy/operator/config/default).
*/}}
{{- define "semantic-router-operator.labels" -}}
helm.sh/chart: {{ include "semantic-router-operator.chart" . }}
app.kubernetes.io/name: {{ include "semantic-router-operator.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/component: manager
app.kubernetes.io/part-of: semantic-router
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Deployment selector and pod labels. The kustomize install's label
transformer (includeSelectors: true) puts its label pairs into the
selector, and the manager manifest adds control-plane; the set below is
identical, so the pod identity a cluster sees is the same either way.
*/}}
{{- define "semantic-router-operator.podLabels" -}}
app.kubernetes.io/name: {{ include "semantic-router-operator.name" . }}
app.kubernetes.io/component: manager
app.kubernetes.io/part-of: semantic-router
control-plane: controller-manager
{{- end }}

{{/*
Create the name of the service account to use.
*/}}
{{- define "semantic-router-operator.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (printf "%s-controller-manager" (include "semantic-router-operator.fullname" .)) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}
