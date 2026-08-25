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

	// Bootstrap selects the immutable v0.3 Router manifest mounted into every
	// replica. The Operator never authors or mutates Router configuration.
	Bootstrap BootstrapSpec `json:"bootstrap"`

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

	// EnvFrom adds environment sources to the Router. When the Management
	// PostgreSQL DSN uses an environment reference, the migration Job projects
	// only that key from one unprefixed Secret source.
	// +optional
	EnvFrom []corev1.EnvFromSource `json:"envFrom,omitempty"`

	// Volumes adds Secret, ConfigMap, CSI, or other deployment-owned volumes.
	// Router configuration itself must continue to use bootstrap.configMapRef.
	// +optional
	Volumes []corev1.Volume `json:"volumes,omitempty"`

	// VolumeMounts mounts deployment-owned volumes into the Router. When the
	// Management PostgreSQL DSN uses a file reference, the migration Job receives
	// only the most-specific matching read-only mount and its volume.
	// +optional
	VolumeMounts []corev1.VolumeMount `json:"volumeMounts,omitempty"`

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

	// PodDisruptionBudget protects Router availability during voluntary
	// disruptions. It defaults on when a Management store is configured.
	// +optional
	PodDisruptionBudget PodDisruptionBudgetSpec `json:"podDisruptionBudget,omitempty"`

	// TopologySpread distributes Router replicas across failure domains. It
	// defaults on when a Management store is configured.
	// +optional
	TopologySpread TopologySpreadSpec `json:"topologySpread,omitempty"`

	// NetworkPolicy isolates inference, Management, internal dispatch, and
	// metrics listeners. It defaults on when a Management store is configured.
	// +optional
	NetworkPolicy NetworkPolicySpec `json:"networkPolicy,omitempty"`
}

// BootstrapSpec selects the sole Router bootstrap manifest.
type BootstrapSpec struct {
	ConfigMapRef BootstrapConfigMapReference `json:"configMapRef"`
}

// BootstrapConfigMapReference selects one key from an immutable ConfigMap in
// the SemanticRouter namespace.
type BootstrapConfigMapReference struct {
	// +kubebuilder:validation:MinLength=1
	// +kubebuilder:validation:MaxLength=253
	Name string `json:"name"`

	// +kubebuilder:validation:MinLength=1
	// +kubebuilder:validation:MaxLength=253
	// +kubebuilder:validation:Pattern=`^[-._a-zA-Z0-9]+$`
	Key string `json:"key"`
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

	// Management configures the private Management API Service port. The target
	// port remains owned by global.services.management_api in the bootstrap.
	// +optional
	Management ManagementServiceSpec `json:"management,omitempty"`

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

// ManagementServiceSpec defines the private Management Service port.
type ManagementServiceSpec struct {
	// Port is the private ClusterIP Service port.
	// +kubebuilder:default=8080
	// +kubebuilder:validation:Minimum=1
	// +kubebuilder:validation:Maximum=65535
	// +optional
	Port int32 `json:"port,omitempty"`
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

	// GatewayMode indicates inference gateway topology: sidecar or external.
	// +kubebuilder:validation:Enum=sidecar;external
	// +optional
	GatewayMode string `json:"gatewayMode,omitempty"`

	// BootstrapRevision is the content digest of the selected immutable
	// bootstrap manifest observed by the controller.
	// +optional
	BootstrapRevision string `json:"bootstrapRevision,omitempty"`

	// PublicService is the inference-only Service name.
	// +optional
	PublicService string `json:"publicService,omitempty"`

	// ManagementService is the private Management API Service name.
	// +optional
	ManagementService string `json:"managementService,omitempty"`

	// Migration reports the Management schema gate when a durable store is configured.
	// +optional
	Migration *MigrationStatus `json:"migration,omitempty"`

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
