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
Get the router ConfigMap name
*/}}
{{- define "semantic-router.configMapName" -}}
{{- printf "%s-config" (include "semantic-router.fullname" .) }}
{{- end }}

{{/*
After installation, management APIs may update config.yaml in the live
ConfigMap. Helm values are the install seed. A changed revision explicitly
reapplies them once; --reuse-values with the same revision preserves later
runtime edits.
*/}}
{{- define "semantic-router.liveConfig" -}}
{{- $policy := .Values.configMap | default (dict) -}}
{{- if .Release.IsUpgrade -}}
{{- $current := lookup "v1" "ConfigMap" (include "semantic-router.namespace" .) (include "semantic-router.configMapName" .) -}}
{{- if $current -}}
{{- $metadata := (get $current "metadata") | default (dict) -}}
{{- $annotations := (get $metadata "annotations") | default (dict) -}}
{{- $appliedRevision := (get $annotations "semantic-router.vllm.ai/chart-config-revision") | default "" -}}
{{- $requestedRevision := (get $policy "applyValuesRevision") | default "" -}}
{{- if eq $requestedRevision $appliedRevision -}}
{{- $data := (get $current "data") | default (dict) -}}
{{- get $data "config.yaml" | default "" -}}
{{- end -}}
{{- end -}}
{{- end }}
{{- end }}

{{/*
The decision model the chart last wrote into the live config, from the live
ConfigMap's annotation; empty on install.
*/}}
{{- define "semantic-router.appliedDecisionModel" -}}
{{- if .Release.IsUpgrade -}}
{{- $current := lookup "v1" "ConfigMap" (include "semantic-router.namespace" .) (include "semantic-router.configMapName" .) -}}
{{- if $current -}}
{{- $metadata := (get $current "metadata") | default (dict) -}}
{{- $annotations := (get $metadata "annotations") | default (dict) -}}
{{- get $annotations "semantic-router.vllm.ai/decision-model" | default "" -}}
{{- end -}}
{{- end -}}
{{- end }}

{{/*
Get the dashboard service account name
*/}}
{{- define "semantic-router.dashboardServiceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- printf "%s-dashboard" (include "semantic-router.fullname" .) }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
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
Get the dashboard local-state PVC name
*/}}
{{- define "semantic-router.dashboardPvcName" -}}
{{- if .Values.dashboard.persistence.existingClaim }}
{{- .Values.dashboard.persistence.existingClaim }}
{{- else }}
{{- printf "%s-dashboard-data" (include "semantic-router.fullname" .) }}
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

{{/*
The gateway mode: standalone (the Router serves config.listeners) or extproc
(the Router serves Envoy's ext_proc gRPC for a gateway in front of it).
*/}}
{{- define "semantic-router.gatewayMode" -}}
{{- $mode := dig "mode" "standalone" (.Values.gateway | default (dict)) -}}
{{- if not (has $mode (list "standalone" "extproc")) -}}
{{- fail (printf "gateway.mode must be standalone or extproc, not %q" $mode) -}}
{{- end -}}
{{- $mode -}}
{{- end }}

{{/*
The standalone listeners, derived once from the effective config. The
container ports, the Service, the probes, the Ingress default and the
Dashboard's target all read this list. A port the Router itself or an earlier
listener already takes fails the render, since every listener binds all of the
pod's addresses. So does the ext_proc port: a gateway that still calls it must
not reach an HTTP listener there.
*/}}
{{- define "semantic-router.listeners" -}}
{{- $config := include "semantic-router.effectiveConfig" . | fromYaml -}}
{{- $taken := dict -}}
{{- $_ := set $taken (toString (int .Values.service.api.targetPort)) "the Router API (service.api.targetPort)" -}}
{{- $_ := set $taken (toString (int .Values.service.metrics.targetPort)) "the Router metrics (service.metrics.targetPort)" -}}
{{- $_ := set $taken (toString (int .Values.service.grpc.targetPort)) "the ext_proc port (service.grpc.targetPort)" -}}
{{- $listeners := list -}}
{{- range $index, $listener := (get $config "listeners") | default (list) -}}
{{-   $name := toString ((get $listener "name") | default (printf "listeners[%d]" $index)) -}}
{{-   $port := int ((get $listener "port") | default 0) -}}
{{-   if or (lt $port 1) (gt $port 65535) -}}
{{-     fail (printf "listener %s needs a port between 1 and 65535" $name) -}}
{{-   end -}}
{{-   $key := toString $port -}}
{{-   if hasKey $taken $key -}}
{{-     fail (printf "listener %s uses port %d, which %s already takes; give the listener another port, or set gateway.mode=extproc to keep an Envoy-based gateway in front of the Router" $name $port (get $taken $key)) -}}
{{-   end -}}
{{-   $_ := set $taken $key (printf "listener %s" $name) -}}
{{-   $tls := not (empty ((get $listener "tls") | default (dict))) -}}
{{-   $scheme := ternary "https" "http" $tls -}}
{{-   $listeners = append $listeners (dict "name" $name "port" $port "portName" (printf "%s-%d" $scheme $port) "scheme" $scheme "tls" $tls) -}}
{{- end -}}
{{- if eq (len $listeners) 0 -}}
{{- fail "gateway.mode=standalone serves config.listeners, and the config has none; add one (for example http-8899 on port 8899) or set gateway.mode=extproc" -}}
{{- end -}}
{{- toYaml (dict "items" $listeners) -}}
{{- end }}

{{/*
The listener that probes and the Dashboard use: the first one without TLS, or
the first listener when all of them serve TLS.
*/}}
{{- define "semantic-router.primaryListener" -}}
{{- $listeners := (include "semantic-router.listeners" . | fromYaml).items -}}
{{- $primary := dict -}}
{{- range $listeners -}}
{{-   if and (empty $primary) (not .tls) -}}
{{-     $primary = . -}}
{{-   end -}}
{{- end -}}
{{- if empty $primary -}}
{{-   $primary = first $listeners -}}
{{- end -}}
{{- toYaml $primary -}}
{{- end }}

{{/*
Resolve the Router config once so every template consumer observes the same
atomic deployment-tooling override instead of Helm's recursive map coalescing.
*/}}
{{- define "semantic-router.effectiveConfig" -}}
{{- $config := deepCopy .Values.config -}}
{{- if and (hasKey .Values "configOverride") (ne .Values.configOverride nil) -}}
{{-   if not (kindIs "map" .Values.configOverride) -}}
{{-     fail "configOverride must be a non-empty mapping" -}}
{{-   end -}}
{{-   if eq (len .Values.configOverride) 0 -}}
{{-     fail "configOverride must be a non-empty mapping" -}}
{{-   end -}}
{{-   $config = deepCopy .Values.configOverride -}}
{{- end -}}
{{- $liveConfig := include "semantic-router.liveConfig" . -}}
{{- if ne (trim $liveConfig) "" -}}
{{-   $parsed := fromYaml $liveConfig -}}
{{-   if hasKey $parsed "Error" -}}
{{-     fail "the live Router ConfigMap contains invalid YAML; correct it or change configMap.applyValuesRevision to replace it explicitly" -}}
{{-   end -}}
{{-   $config = $parsed -}}
{{- end -}}
{{- toYaml $config -}}
{{- end }}
