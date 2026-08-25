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
Resolve the single Router manifest shared by every chart template. Deployment
tooling may supply a complete canonical manifest through configOverride so
Helm's recursive values merge cannot leak chart sample routes into it.
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
{{- toYaml $config -}}
{{- end }}

{{/*
Name the immutable Router bootstrap from its exact content. A manifest change
creates a new ConfigMap reference and therefore a normal Pod rollout.
*/}}
{{- define "semantic-router.configMapName" -}}
{{- $payload := printf "%s\n%s\n%s\n%s" (include "semantic-router.effectiveConfig" .) (toJson .Values.toolsDb) (toJson .Values.dependencies) .Chart.AppVersion -}}
{{- $base := include "semantic-router.fullname" . | trunc 43 | trimSuffix "-" -}}
{{- printf "%s-config-%s" $base (sha256sum $payload | trunc 12) -}}
{{- end }}

{{/*
Return true when the Router manifest configures durable Management state.
Capabilities come from typed service/store blocks; there is no deployment
mode field.
*/}}
{{- define "semantic-router.hasManagementStore" -}}
{{- $config := include "semantic-router.effectiveConfig" . | fromYaml -}}
{{- $global := (get $config "global") | default (dict) -}}
{{- $stores := (get $global "stores") | default (dict) -}}
{{- $management := (get $stores "management") | default (dict) -}}
{{- if hasKey $management "postgres" }}true{{ else }}false{{ end -}}
{{- end }}

{{- define "semantic-router.hasRuntimeStore" -}}
{{- $config := include "semantic-router.effectiveConfig" . | fromYaml -}}
{{- $global := (get $config "global") | default (dict) -}}
{{- $stores := (get $global "stores") | default (dict) -}}
{{- $runtime := (get $stores "runtime") | default (dict) -}}
{{- if hasKey $runtime "redis" }}true{{ else }}false{{ end -}}
{{- end }}

{{- define "semantic-router.managementAPIEnabled" -}}
{{- $config := include "semantic-router.effectiveConfig" . | fromYaml -}}
{{- $global := (get $config "global") | default (dict) -}}
{{- $services := (get $global "services") | default (dict) -}}
{{- $management := (get $services "management_api") | default (dict) -}}
{{- if ((get $management "enabled") | default false) }}true{{ else }}false{{ end -}}
{{- end }}

{{- define "semantic-router.accessEnabled" -}}
{{- $config := include "semantic-router.effectiveConfig" . | fromYaml -}}
{{- $global := (get $config "global") | default (dict) -}}
{{- $services := (get $global "services") | default (dict) -}}
{{- $access := (get $services "access") | default (dict) -}}
{{- if ((get $access "enabled") | default false) }}true{{ else }}false{{ end -}}
{{- end }}

{{/*
Project only the PostgreSQL DSN environment variable into the one-shot schema
migrator. A single envFrom Secret is converted into one secretKeyRef so the
other Router credentials in that Secret never enter the migration container.
*/}}
{{- define "semantic-router.managementMigrationEnv" -}}
{{- $root := index . 0 -}}
{{- $dsnEnv := index . 1 -}}
{{- $matches := list -}}
{{- range $variable := concat $root.Values.env $root.Values.extraEnv -}}
{{- if eq ((get $variable "name") | default "") $dsnEnv -}}
{{- $valueFrom := (get $variable "valueFrom") | default (dict) -}}
{{- $secretKeyRef := (get $valueFrom "secretKeyRef") | default (dict) -}}
{{- if or (empty (get $secretKeyRef "name")) (empty (get $secretKeyRef "key")) -}}
{{- fail (printf "Management PostgreSQL DSN environment variable %s must use a Secret key reference" $dsnEnv) -}}
{{- end -}}
{{- $matches = append $matches $variable -}}
{{- end -}}
{{- end -}}
{{- if gt (len $matches) 1 -}}
{{- fail (printf "Management PostgreSQL DSN environment variable %s is declared more than once" $dsnEnv) -}}
{{- end -}}
{{- if eq (len $matches) 0 -}}
{{- if ne (len $root.Values.envFromSecrets) 1 -}}
{{- fail (printf "Management PostgreSQL DSN environment variable %s requires one explicit env Secret reference or exactly one envFromSecrets entry" $dsnEnv) -}}
{{- end -}}
{{- $secretName := first $root.Values.envFromSecrets -}}
{{- $matches = append $matches (dict "name" $dsnEnv "valueFrom" (dict "secretKeyRef" (dict "name" $secretName "key" $dsnEnv))) -}}
{{- end -}}
{{- toYaml $matches -}}
{{- end }}

{{/*
Select the most-specific read-only Secret projection containing a PostgreSQL
DSN file. The migration Job never inherits unrelated Router volume mounts.
*/}}
{{- define "semantic-router.managementMigrationFileProjection" -}}
{{- $root := index . 0 -}}
{{- $dsnFile := index . 1 -}}
{{- $selected := dict -}}
{{- $selectedLength := 0 -}}
{{- range $mount := $root.Values.extraVolumeMounts -}}
{{- $mountPath := (get $mount "mountPath") | default "" -}}
{{- $subPath := (get $mount "subPath") | default "" -}}
{{- $subPathExpr := (get $mount "subPathExpr") | default "" -}}
{{- $prefix := printf "%s/" (trimSuffix "/" $mountPath) -}}
{{- $contains := ternary (eq $mountPath $dsnFile) (or (eq $mountPath $dsnFile) (hasPrefix $prefix $dsnFile)) (or (ne $subPath "") (ne $subPathExpr "")) -}}
{{- if and (eq ((get $mount "readOnly") | default false) true) $contains (gt (len $mountPath) $selectedLength) -}}
{{- $_ := set $selected "mount" $mount -}}
{{- $selectedLength = len $mountPath -}}
{{- end -}}
{{- end -}}
{{- if not (hasKey $selected "mount") -}}
{{- fail (printf "Management PostgreSQL DSN file %s requires a matching read-only extraVolumeMount" $dsnFile) -}}
{{- end -}}
{{- $mount := get $selected "mount" -}}
{{- $volumeName := get $mount "name" -}}
{{- $volumes := list -}}
{{- range $volume := $root.Values.extraVolumes -}}
{{- if eq ((get $volume "name") | default "") $volumeName -}}
{{- if not (or (hasKey $volume "secret") (hasKey $volume "projected") (hasKey $volume "csi")) -}}
{{- fail (printf "Management PostgreSQL DSN file volume %s must use a Secret, projected, or CSI source" $volumeName) -}}
{{- end -}}
{{- $volumes = append $volumes $volume -}}
{{- end -}}
{{- end -}}
{{- if ne (len $volumes) 1 -}}
{{- fail (printf "Management PostgreSQL DSN file mount %s requires exactly one extraVolume" $volumeName) -}}
{{- end -}}
{{- toYaml (dict "volumeMounts" (list $mount) "volumes" $volumes) -}}
{{- end }}
