/*
Copyright 2026 vLLM Semantic Router Contributors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package v1alpha1

import (
	corev1 "k8s.io/api/core/v1"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// EDIT THIS FILE!  THIS IS SCAFFOLDING FOR YOU TO OWN!
// NOTE: json tags are required.  Any new fields you add must have json tags for the fields to be serialized.

// SemanticRouterSpec defines the desired state of SemanticRouter
type SemanticRouterSpec struct {
	// INSERT ADDITIONAL SPEC FIELDS - desired state of cluster
	// Important: Run "make generate" to regenerate code after modifying this file

	// Image configuration
	// +optional
	Image ImageSpec `json:"image,omitempty"`

	// Number of replicas
	// +kubebuilder:default=1
	// +kubebuilder:validation:Minimum=0
	// +optional
	Replicas *int32 `json:"replicas,omitempty"`

	// ImagePullSecrets for private registries
	// +optional
	ImagePullSecrets []corev1.LocalObjectReference `json:"imagePullSecrets,omitempty"`

	// ServiceAccount configuration
	// +optional
	ServiceAccount ServiceAccountSpec `json:"serviceAccount,omitempty"`

	// Service configuration
	// +optional
	Service ServiceSpec `json:"service,omitempty"`

	// Resource requirements
	// +optional
	Resources corev1.ResourceRequirements `json:"resources,omitempty"`

	// Persistence configuration
	// +optional
	Persistence PersistenceSpec `json:"persistence,omitempty"`

	// Configuration for the semantic router
	// +optional
	Config ConfigSpec `json:"config,omitempty"`

	// Tools database configuration
	// +optional
	ToolsDb []ToolEntry `json:"toolsDb,omitempty"`

	// VLLMEndpoints configuration - generates vllm_endpoints and model_config in config.yaml
	// +optional
	VLLMEndpoints []VLLMEndpointSpec `json:"vllmEndpoints,omitempty"`

	// Autoscaling configuration
	// +optional
	Autoscaling AutoscalingSpec `json:"autoscaling,omitempty"`

	// Probes configuration
	// +optional
	StartupProbe *ProbeSpec `json:"startupProbe,omitempty"`
	// +optional
	LivenessProbe *ProbeSpec `json:"livenessProbe,omitempty"`
	// +optional
	ReadinessProbe *ProbeSpec `json:"readinessProbe,omitempty"`

	// Security context
	// +optional
	SecurityContext *corev1.SecurityContext `json:"securityContext,omitempty"`

	// Pod security context
	// +optional
	PodSecurityContext *corev1.PodSecurityContext `json:"podSecurityContext,omitempty"`

	// Pod annotations
	// +optional
	PodAnnotations map[string]string `json:"podAnnotations,omitempty"`

	// Node selector
	// +optional
	NodeSelector map[string]string `json:"nodeSelector,omitempty"`

	// Tolerations
	// +optional
	Tolerations []corev1.Toleration `json:"tolerations,omitempty"`

	// Affinity
	// +optional
	Affinity *corev1.Affinity `json:"affinity,omitempty"`

	// Environment variables
	// +optional
	Env []corev1.EnvVar `json:"env,omitempty"`

	// Container arguments
	// +optional
	Args []string `json:"args,omitempty"`

	// Gateway integration for reusing existing gateways
	// +optional
	Gateway *GatewaySpec `json:"gateway,omitempty"`

	// OpenShift-specific features
	// +optional
	OpenShift *OpenShiftSpec `json:"openshift,omitempty"`

	// Ingress configuration
	// +optional
	Ingress IngressSpec `json:"ingress,omitempty"`
}

// ImageSpec defines the container image configuration
type ImageSpec struct {
	// Repository is the container image repository
	// +kubebuilder:default="ghcr.io/vllm-project/semantic-router/extproc"
	// +optional
	Repository string `json:"repository,omitempty"`

	// Tag is the container image tag
	// +kubebuilder:default="latest"
	// +optional
	Tag string `json:"tag,omitempty"`

	// PullPolicy is the image pull policy
	// +kubebuilder:default="IfNotPresent"
	// +kubebuilder:validation:Enum=Always;Never;IfNotPresent
	// +optional
	PullPolicy corev1.PullPolicy `json:"pullPolicy,omitempty"`

	// ImageRegistry is an optional registry prefix
	// +optional
	ImageRegistry string `json:"imageRegistry,omitempty"`
}

// ServiceAccountSpec defines service account configuration
type ServiceAccountSpec struct {
	// Create specifies whether to create a service account
	// +kubebuilder:default=true
	// +optional
	Create *bool `json:"create,omitempty"`

	// Name of the service account to use
	// +optional
	Name string `json:"name,omitempty"`

	// Annotations for the service account
	// +optional
	Annotations map[string]string `json:"annotations,omitempty"`
}

// ServiceSpec defines the service configuration
type ServiceSpec struct {
	// Type is the service type
	// +kubebuilder:default="ClusterIP"
	// +kubebuilder:validation:Enum=ClusterIP;NodePort;LoadBalancer
	// +optional
	Type corev1.ServiceType `json:"type,omitempty"`

	// GRPC port configuration
	// +optional
	GRPC PortSpec `json:"grpc,omitempty"`

	// API port configuration
	// +optional
	API PortSpec `json:"api,omitempty"`

	// Metrics port configuration
	// +optional
	Metrics MetricsPortSpec `json:"metrics,omitempty"`
}

// PortSpec defines a service port configuration
type PortSpec struct {
	// Port is the service port
	// +kubebuilder:validation:Minimum=1
	// +kubebuilder:validation:Maximum=65535
	// +optional
	Port int32 `json:"port,omitempty"`

	// TargetPort is the container port
	// +kubebuilder:validation:Minimum=1
	// +kubebuilder:validation:Maximum=65535
	// +optional
	TargetPort int32 `json:"targetPort,omitempty"`

	// Protocol is the port protocol
	// +kubebuilder:default="TCP"
	// +optional
	Protocol corev1.Protocol `json:"protocol,omitempty"`
}

// MetricsPortSpec extends PortSpec with enable flag
type MetricsPortSpec struct {
	PortSpec `json:",inline"`

	// Enabled indicates if metrics should be exposed
	// +kubebuilder:default=true
	// +optional
	Enabled *bool `json:"enabled,omitempty"`
}

// PersistenceSpec defines persistence configuration
type PersistenceSpec struct {
	// Enabled indicates if persistence is enabled
	// +kubebuilder:default=true
	// +optional
	Enabled *bool `json:"enabled,omitempty"`

	// StorageClassName is the storage class name
	// +kubebuilder:default="standard"
	// +optional
	StorageClassName string `json:"storageClassName,omitempty"`

	// AccessMode is the access mode
	// +kubebuilder:default="ReadWriteOnce"
	// +optional
	AccessMode corev1.PersistentVolumeAccessMode `json:"accessMode,omitempty"`

	// Size is the storage size
	// +kubebuilder:default="10Gi"
	// +optional
	Size string `json:"size,omitempty"`

	// ExistingClaim is an existing PVC to use
	// +optional
	ExistingClaim string `json:"existingClaim,omitempty"`

	// Annotations for the PVC
	// +optional
	Annotations map[string]string `json:"annotations,omitempty"`
}

// ConfigSpec defines the semantic router configuration
type ConfigSpec struct {
	// BERT model configuration
	// +optional
	BertModel *BertModelConfig `json:"bert_model,omitempty"`

	// Semantic cache configuration
	// +optional
	SemanticCache *SemanticCacheConfig `json:"semantic_cache,omitempty"`

	// Tools configuration
	// +optional
	Tools *ToolsConfig `json:"tools,omitempty"`

	// Prompt guard configuration
	// +optional
	PromptGuard *PromptGuardConfig `json:"prompt_guard,omitempty"`

	// Classifier configuration
	// +optional
	Classifier *ClassifierConfig `json:"classifier,omitempty"`

	// Reasoning families
	// +optional
	ReasoningFamilies map[string]ReasoningFamily `json:"reasoning_families,omitempty"`

	// Default reasoning effort
	// +kubebuilder:validation:Enum=low;medium;high
	// +optional
	DefaultReasoningEffort string `json:"default_reasoning_effort,omitempty"`

	// API configuration
	// +optional
	API *APIConfig `json:"api,omitempty"`

	// Observability configuration
	// +optional
	Observability *ObservabilityConfig `json:"observability,omitempty"`
}

// BertModelConfig defines BERT model configuration
type BertModelConfig struct {
	// +kubebuilder:default="models/mom-embedding-light"
	// +optional
	ModelID string `json:"model_id,omitempty"`
	// Threshold for embedding similarity (0.0-1.0). Stored as string to avoid float precision issues.
	// +kubebuilder:default="0.6"
	// +kubebuilder:validation:Pattern=`^0(\.[0-9]+)?$|^1(\.0+)?$`
	// +optional
	Threshold string `json:"threshold,omitempty"`
	// +kubebuilder:default=true
	// +optional
	UseCPU bool `json:"use_cpu,omitempty"`
}

// SemanticCacheConfig defines semantic cache configuration
type SemanticCacheConfig struct {
	// +kubebuilder:default=true
	// +optional
	Enabled bool `json:"enabled,omitempty"`
	// +kubebuilder:default="memory"
	// +optional
	BackendType string `json:"backend_type,omitempty"`
	// Similarity threshold for cache hits (0.0-1.0). Stored as string to avoid float precision issues.
	// +kubebuilder:default="0.8"
	// +kubebuilder:validation:Pattern=`^0(\.[0-9]+)?$|^1(\.0+)?$`
	// +optional
	SimilarityThreshold string `json:"similarity_threshold,omitempty"`
	// +kubebuilder:default=1000
	// +optional
	MaxEntries int `json:"max_entries,omitempty"`
	// +kubebuilder:default=3600
	// +optional
	TTLSeconds int `json:"ttl_seconds,omitempty"`
	// +kubebuilder:default="fifo"
	// +optional
	EvictionPolicy string `json:"eviction_policy,omitempty"`
}

// ToolsConfig defines tools configuration
type ToolsConfig struct {
	// +kubebuilder:default=true
	// +optional
	Enabled bool `json:"enabled,omitempty"`
	// +kubebuilder:default=3
	// +optional
	TopK int `json:"top_k,omitempty"`
	// Similarity threshold for tool selection (0.0-1.0). Stored as string to avoid float precision issues.
	// +kubebuilder:default="0.2"
	// +kubebuilder:validation:Pattern=`^0(\.[0-9]+)?$|^1(\.0+)?$`
	// +optional
	SimilarityThreshold string `json:"similarity_threshold,omitempty"`
	// +kubebuilder:default="config/tools_db.json"
	// +optional
	ToolsDBPath string `json:"tools_db_path,omitempty"`
	// +kubebuilder:default=true
	// +optional
	FallbackToEmpty bool `json:"fallback_to_empty,omitempty"`
}

// PromptGuardConfig defines prompt guard configuration
type PromptGuardConfig struct {
	// +kubebuilder:default=true
	// +optional
	Enabled bool `json:"enabled,omitempty"`
	// +kubebuilder:default=false
	// +optional
	UseModernBERT bool `json:"use_modernbert,omitempty"`
	// +kubebuilder:default="models/mom-jailbreak-classifier"
	// +optional
	ModelID string `json:"model_id,omitempty"`
	// Jailbreak detection threshold (0.0-1.0). Stored as string to avoid float precision issues.
	// +kubebuilder:default="0.7"
	// +kubebuilder:validation:Pattern=`^0(\.[0-9]+)?$|^1(\.0+)?$`
	// +optional
	Threshold string `json:"threshold,omitempty"`
	// +kubebuilder:default=true
	// +optional
	UseCPU bool `json:"use_cpu,omitempty"`
	// +optional
	JailbreakMappingPath string `json:"jailbreak_mapping_path,omitempty"`
}

// ClassifierConfig defines classifier configuration
type ClassifierConfig struct {
	// +optional
	CategoryModel *CategoryModelConfig `json:"category_model,omitempty"`
	// +optional
	PIIModel *PIIModelConfig `json:"pii_model,omitempty"`
}

// CategoryModelConfig defines category model configuration
type CategoryModelConfig struct {
	// +optional
	ModelID string `json:"model_id,omitempty"`
	// +optional
	UseModernBERT bool `json:"use_modernbert,omitempty"`
	// Classification threshold (0.0-1.0). Stored as string to avoid float precision issues.
	// +kubebuilder:validation:Pattern=`^0(\.[0-9]+)?$|^1(\.0+)?$`
	// +optional
	Threshold string `json:"threshold,omitempty"`
	// +optional
	UseCPU bool `json:"use_cpu,omitempty"`
	// +optional
	CategoryMappingPath string `json:"category_mapping_path,omitempty"`
}

// PIIModelConfig defines PII model configuration
type PIIModelConfig struct {
	// +optional
	ModelID string `json:"model_id,omitempty"`
	// +optional
	UseModernBERT bool `json:"use_modernbert,omitempty"`
	// Detection threshold (0.0-1.0). Stored as string to avoid float precision issues.
	// +kubebuilder:validation:Pattern=`^0(\.[0-9]+)?$|^1(\.0+)?$`
	// +optional
	Threshold string `json:"threshold,omitempty"`
	// +optional
	UseCPU bool `json:"use_cpu,omitempty"`
	// +optional
	PIIMappingPath string `json:"pii_mapping_path,omitempty"`
}

// ReasoningFamily defines reasoning family configuration
type ReasoningFamily struct {
	// +optional
	Type string `json:"type,omitempty"`
	// +optional
	Parameter string `json:"parameter,omitempty"`
}

// APIConfig defines API configuration
type APIConfig struct {
	// +optional
	BatchClassification *BatchClassificationConfig `json:"batch_classification,omitempty"`
}

// BatchClassificationConfig defines batch classification configuration
type BatchClassificationConfig struct {
	// +kubebuilder:default=100
	// +optional
	MaxBatchSize int `json:"max_batch_size,omitempty"`
	// +kubebuilder:default=5
	// +optional
	ConcurrencyThreshold int `json:"concurrency_threshold,omitempty"`
	// +kubebuilder:default=8
	// +optional
	MaxConcurrency int `json:"max_concurrency,omitempty"`
	// +optional
	Metrics *BatchMetricsConfig `json:"metrics,omitempty"`
}

// BatchMetricsConfig defines batch classification metrics configuration
type BatchMetricsConfig struct {
	// +kubebuilder:default=true
	// +optional
	Enabled bool `json:"enabled,omitempty"`
	// +kubebuilder:default=true
	// +optional
	DetailedGoroutineTracking bool `json:"detailed_goroutine_tracking,omitempty"`
	// +kubebuilder:default=false
	// +optional
	HighResolutionTiming bool `json:"high_resolution_timing,omitempty"`
	// Sample rate for metrics (0.0-1.0). Stored as string to avoid float precision issues.
	// +kubebuilder:default="1.0"
	// +kubebuilder:validation:Pattern=`^0(\.[0-9]+)?$|^1(\.0+)?$`
	// +optional
	SampleRate string `json:"sample_rate,omitempty"`
	// Duration buckets for histograms. Stored as strings to avoid float precision issues.
	// Example: ["0.001", "0.005", "0.01", "0.025", "0.05", "0.1", "0.25", "0.5", "1", "2.5", "5", "10", "30"]
	// +optional
	DurationBuckets []string `json:"duration_buckets,omitempty"`
	// +optional
	SizeBuckets []int `json:"size_buckets,omitempty"`
}

// ObservabilityConfig defines observability configuration
type ObservabilityConfig struct {
	// +optional
	Tracing *TracingConfig `json:"tracing,omitempty"`
}

// TracingConfig defines tracing configuration
type TracingConfig struct {
	// +kubebuilder:default=false
	// +optional
	Enabled bool `json:"enabled,omitempty"`
	// +kubebuilder:default="opentelemetry"
	// +optional
	Provider string `json:"provider,omitempty"`
	// +optional
	Exporter *ExporterConfig `json:"exporter,omitempty"`
	// +optional
	Sampling *SamplingConfig `json:"sampling,omitempty"`
	// +optional
	Resource *ResourceConfig `json:"resource,omitempty"`
}

// ExporterConfig defines exporter configuration
type ExporterConfig struct {
	// +kubebuilder:default="otlp"
	// +optional
	Type string `json:"type,omitempty"`
	// +kubebuilder:default="jaeger:4317"
	// +optional
	Endpoint string `json:"endpoint,omitempty"`
	// +kubebuilder:default=true
	// +optional
	Insecure bool `json:"insecure,omitempty"`
}

// SamplingConfig defines sampling configuration
type SamplingConfig struct {
	// +kubebuilder:default="always_on"
	// +optional
	Type string `json:"type,omitempty"`
	// Sampling rate (0.0-1.0). Stored as string to avoid float precision issues.
	// +kubebuilder:default="1.0"
	// +kubebuilder:validation:Pattern=`^0(\.[0-9]+)?$|^1(\.0+)?$`
	// +optional
	Rate string `json:"rate,omitempty"`
}

// ResourceConfig defines resource configuration for tracing
type ResourceConfig struct {
	// +kubebuilder:default="vllm-semantic-router"
	// +optional
	ServiceName string `json:"service_name,omitempty"`
	// +kubebuilder:default="v0.1.0"
	// +optional
	ServiceVersion string `json:"service_version,omitempty"`
	// +kubebuilder:default="development"
	// +optional
	DeploymentEnvironment string `json:"deployment_environment,omitempty"`
}

// ToolEntry defines a tool entry in the tools database
type ToolEntry struct {
	// +optional
	Tool Tool `json:"tool,omitempty"`
	// +optional
	Description string `json:"description,omitempty"`
	// +optional
	Category string `json:"category,omitempty"`
	// +optional
	Tags []string `json:"tags,omitempty"`
}

// Tool defines a tool function
type Tool struct {
	// +kubebuilder:validation:Enum=function
	// +optional
	Type string `json:"type,omitempty"`
	// +optional
	Function ToolFunction `json:"function,omitempty"`
}

// ToolFunction defines a tool function details
type ToolFunction struct {
	// +optional
	Name string `json:"name,omitempty"`
	// +optional
	Description string `json:"description,omitempty"`
	// +optional
	Parameters ToolParameters `json:"parameters,omitempty"`
}

// ToolParameters defines tool function parameters
type ToolParameters struct {
	// +optional
	Type string `json:"type,omitempty"`
	// +optional
	// +kubebuilder:pruning:PreserveUnknownFields
	// +kubebuilder:validation:Type=object
	Properties *apiextensionsv1.JSON `json:"properties,omitempty"`
	// +optional
	Required []string `json:"required,omitempty"`
}

// AutoscalingSpec defines autoscaling configuration
type AutoscalingSpec struct {
	// Enabled indicates if HPA is enabled
	// +kubebuilder:default=false
	// +optional
	Enabled *bool `json:"enabled,omitempty"`

	// MinReplicas is the minimum number of replicas
	// +kubebuilder:default=1
	// +optional
	MinReplicas *int32 `json:"minReplicas,omitempty"`

	// MaxReplicas is the maximum number of replicas
	// +kubebuilder:default=10
	// +optional
	MaxReplicas *int32 `json:"maxReplicas,omitempty"`

	// TargetCPUUtilizationPercentage is the target CPU percentage
	// +kubebuilder:default=80
	// +optional
	TargetCPUUtilizationPercentage *int32 `json:"targetCPUUtilizationPercentage,omitempty"`

	// TargetMemoryUtilizationPercentage is the target memory percentage
	// +optional
	TargetMemoryUtilizationPercentage *int32 `json:"targetMemoryUtilizationPercentage,omitempty"`
}

// ProbeSpec defines probe configuration
type ProbeSpec struct {
	// Enabled indicates if the probe is enabled
	// +kubebuilder:default=true
	// +optional
	Enabled *bool `json:"enabled,omitempty"`

	// InitialDelaySeconds before probe starts
	// +optional
	InitialDelaySeconds *int32 `json:"initialDelaySeconds,omitempty"`

	// PeriodSeconds between probes
	// +optional
	PeriodSeconds *int32 `json:"periodSeconds,omitempty"`

	// TimeoutSeconds for probe
	// +optional
	TimeoutSeconds *int32 `json:"timeoutSeconds,omitempty"`

	// FailureThreshold for probe
	// +optional
	FailureThreshold *int32 `json:"failureThreshold,omitempty"`
}

// IngressSpec defines ingress configuration
type IngressSpec struct {
	// Enabled indicates if ingress is enabled
	// +kubebuilder:default=false
	// +optional
	Enabled *bool `json:"enabled,omitempty"`

	// ClassName is the ingress class name
	// +optional
	ClassName string `json:"className,omitempty"`

	// Annotations for ingress
	// +optional
	Annotations map[string]string `json:"annotations,omitempty"`

	// Hosts configuration
	// +optional
	Hosts []IngressHost `json:"hosts,omitempty"`

	// TLS configuration
	// +optional
	TLS []IngressTLS `json:"tls,omitempty"`
}

// IngressHost defines an ingress host
type IngressHost struct {
	// +optional
	Host string `json:"host,omitempty"`
	// +optional
	Paths []IngressPath `json:"paths,omitempty"`
}

// IngressPath defines an ingress path
type IngressPath struct {
	// +optional
	Path string `json:"path,omitempty"`
	// +optional
	PathType string `json:"pathType,omitempty"`
	// +optional
	ServicePort int32 `json:"servicePort,omitempty"`
}

// IngressTLS defines ingress TLS configuration
type IngressTLS struct {
	// +optional
	SecretName string `json:"secretName,omitempty"`
	// +optional
	Hosts []string `json:"hosts,omitempty"`
}

// VLLMEndpointSpec defines a vLLM model backend endpoint
type VLLMEndpointSpec struct {
	// Name of the endpoint (used in model_config.preferred_endpoints)
	// +kubebuilder:validation:MinLength=1
	Name string `json:"name"`

	// Model name as reported by vLLM (e.g., "Model-A", "llama3-8b")
	// +kubebuilder:validation:MinLength=1
	Model string `json:"model"`

	// Reasoning family for the model (e.g., "qwen3", "deepseek", "gpt")
	// +optional
	ReasoningFamily string `json:"reasoningFamily,omitempty"`

	// Backend configuration
	Backend VLLMBackend `json:"backend"`

	// Weight for load balancing (default: 1)
	// +optional
	// +kubebuilder:default=1
	Weight int `json:"weight,omitempty"`
}

// VLLMBackend specifies how to reach the vLLM service
type VLLMBackend struct {
	// Type of backend: kserve, llamastack, or service
	// +kubebuilder:validation:Enum=kserve;llamastack;service
	Type string `json:"type"`

	// For type=kserve: InferenceService name for auto-discovery
	// +optional
	InferenceServiceName string `json:"inferenceServiceName,omitempty"`

	// For type=llamastack: Labels to match services
	// +optional
	DiscoveryLabels map[string]string `json:"discoveryLabels,omitempty"`

	// For type=service: Direct service configuration
	// +optional
	Service *ServiceBackend `json:"service,omitempty"`
}

// ServiceBackend defines a direct Kubernetes service backend
type ServiceBackend struct {
	// Service name
	// +kubebuilder:validation:MinLength=1
	Name string `json:"name"`

	// Service namespace (defaults to same namespace)
	// +optional
	Namespace string `json:"namespace,omitempty"`

	// Service port
	// +kubebuilder:validation:Minimum=1
	Port int32 `json:"port"`
}

// GatewaySpec defines Gateway API integration configuration
type GatewaySpec struct {
	// ExistingRef references an existing Gateway to use
	// +optional
	ExistingRef *GatewayReference `json:"existingRef,omitempty"`
}

// GatewayReference references an existing Gateway
type GatewayReference struct {
	// Name of the Gateway
	// +kubebuilder:validation:MinLength=1
	Name string `json:"name"`

	// Namespace of the Gateway
	// +kubebuilder:validation:MinLength=1
	Namespace string `json:"namespace"`
}

// OpenShiftSpec defines OpenShift-specific configuration
type OpenShiftSpec struct {
	// Routes configuration for OpenShift Routes
	// +optional
	Routes *RouteConfig `json:"routes,omitempty"`
}

// RouteConfig defines OpenShift Route configuration
type RouteConfig struct {
	// Enabled specifies whether to create an OpenShift Route
	// +optional
	// +kubebuilder:default=false
	Enabled bool `json:"enabled,omitempty"`

	// Hostname for the Route (optional - OpenShift generates if empty)
	// +optional
	Hostname string `json:"hostname,omitempty"`

	// TLS configuration for the Route
	// +optional
	TLS *RouteTLSConfig `json:"tls,omitempty"`
}

// RouteTLSConfig defines TLS configuration for OpenShift Routes
type RouteTLSConfig struct {
	// Termination type (edge, passthrough, reencrypt)
	// +optional
	// +kubebuilder:default="edge"
	// +kubebuilder:validation:Enum=edge;passthrough;reencrypt
	Termination string `json:"termination,omitempty"`

	// InsecureEdgeTerminationPolicy for HTTP traffic
	// +optional
	// +kubebuilder:default="Redirect"
	// +kubebuilder:validation:Enum=Allow;Redirect;None
	InsecureEdgeTerminationPolicy string `json:"insecureEdgeTerminationPolicy,omitempty"`
}

// SemanticRouterStatus defines the observed state of SemanticRouter
type SemanticRouterStatus struct {
	// INSERT ADDITIONAL STATUS FIELD - define observed state of cluster
	// Important: Run "make generate" to regenerate code after modifying this file

	// Conditions represent the latest available observations of the SemanticRouter's state
	// +optional
	Conditions []metav1.Condition `json:"conditions,omitempty"`

	// ObservedGeneration reflects the generation of the most recently observed SemanticRouter
	// +optional
	ObservedGeneration int64 `json:"observedGeneration,omitempty"`

	// Replicas is the current number of replicas
	// +optional
	Replicas int32 `json:"replicas,omitempty"`

	// ReadyReplicas is the number of ready replicas
	// +optional
	ReadyReplicas int32 `json:"readyReplicas,omitempty"`

	// Phase represents the current phase of the SemanticRouter
	// +optional
	Phase string `json:"phase,omitempty"`

	// GatewayMode indicates deployment mode: standalone or gateway-integration
	// +optional
	GatewayMode string `json:"gatewayMode,omitempty"`

	// OpenShiftFeatures tracks OpenShift-specific feature status
	// +optional
	OpenShiftFeatures *OpenShiftFeaturesStatus `json:"openshiftFeatures,omitempty"`
}

// OpenShiftFeaturesStatus tracks OpenShift-specific feature status
type OpenShiftFeaturesStatus struct {
	// RoutesEnabled indicates if OpenShift Routes are enabled
	RoutesEnabled bool `json:"routesEnabled"`

	// RouteHostname is the hostname of the created Route
	// +optional
	RouteHostname string `json:"routeHostname,omitempty"`
}

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:resource:path=semanticrouters,scope=Namespaced,shortName=sr
// +kubebuilder:printcolumn:name="Replicas",type=integer,JSONPath=`.spec.replicas`
// +kubebuilder:printcolumn:name="Ready",type=integer,JSONPath=`.status.readyReplicas`
// +kubebuilder:printcolumn:name="Phase",type=string,JSONPath=`.status.phase`
// +kubebuilder:printcolumn:name="Age",type="date",JSONPath=".metadata.creationTimestamp"

// SemanticRouter is the Schema for the semanticrouters API
type SemanticRouter struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   SemanticRouterSpec   `json:"spec,omitempty"`
	Status SemanticRouterStatus `json:"status,omitempty"`
}

// +kubebuilder:object:root=true

// SemanticRouterList contains a list of SemanticRouter
type SemanticRouterList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []SemanticRouter `json:"items"`
}

func init() {
	SchemeBuilder.Register(&SemanticRouter{}, &SemanticRouterList{})
}
