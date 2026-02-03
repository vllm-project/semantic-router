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
	"k8s.io/apimachinery/pkg/runtime"
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

	// Embedding models configuration (qwen3, gemma, mmbert)
	// +optional
	EmbeddingModels *EmbeddingModelsConfig `json:"embedding_models,omitempty"`

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

	// Complexity rules for complexity-aware routing
	// +optional
	ComplexityRules []ComplexityRulesConfig `json:"complexity_rules,omitempty"`

	// Decision routing strategy ("priority" for priority-based matching)
	// +kubebuilder:validation:Enum=priority
	// +optional
	Strategy string `json:"strategy,omitempty" yaml:"strategy,omitempty"`

	// Routing decisions based on signals (domain, complexity, etc.)
	// +optional
	Decisions []DecisionConfig `json:"decisions,omitempty"`

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
	// Enabled controls whether semantic caching is active
	// +kubebuilder:default=true
	// +optional
	Enabled bool `json:"enabled,omitempty"`

	// BackendType specifies the cache backend to use
	// Options: "memory" (default), "redis", "milvus", "hybrid"
	// +kubebuilder:default="memory"
	// +kubebuilder:validation:Enum=memory;redis;milvus;hybrid
	// +optional
	BackendType string `json:"backend_type,omitempty"`

	// Similarity threshold for cache hits (0.0-1.0). Stored as string to avoid float precision issues.
	// +kubebuilder:default="0.8"
	// +kubebuilder:validation:Pattern=`^0(\.[0-9]+)?$|^1(\.0+)?$`
	// +optional
	SimilarityThreshold string `json:"similarity_threshold,omitempty"`

	// MaxEntries is the maximum number of cache entries (for memory/hybrid backends)
	// +kubebuilder:default=1000
	// +optional
	MaxEntries int `json:"max_entries,omitempty"`

	// TTLSeconds is the time-to-live for cache entries in seconds
	// +kubebuilder:default=3600
	// +optional
	TTLSeconds int `json:"ttl_seconds,omitempty"`

	// EvictionPolicy for in-memory cache ("fifo", "lru", "lfu")
	// +kubebuilder:default="fifo"
	// +kubebuilder:validation:Enum=fifo;lru;lfu
	// +optional
	EvictionPolicy string `json:"eviction_policy,omitempty"`

	// Redis configuration (required when backend_type is "redis")
	// +optional
	Redis *RedisCacheConfig `json:"redis,omitempty"`

	// Milvus configuration (required when backend_type is "milvus")
	// +optional
	Milvus *MilvusCacheConfig `json:"milvus,omitempty"`

	// EmbeddingModel specifies which embedding model to use for semantic similarity
	// Options: "bert" (default), "qwen3", "gemma", "mmbert"
	// +kubebuilder:default="bert"
	// +kubebuilder:validation:Enum=bert;qwen3;gemma;mmbert
	// +optional
	EmbeddingModel string `json:"embedding_model,omitempty"`

	// HNSW configuration for hybrid/in-memory backends
	// +optional
	HNSW *HNSWCacheConfig `json:"hnsw,omitempty"`
}

// RedisCacheConfig defines Redis cache backend configuration.
// Configure these settings when using Redis as the semantic cache backend.
type RedisCacheConfig struct {
	// Connection settings for Redis server
	// +optional
	Connection RedisCacheConnection `json:"connection,omitempty"`

	// Index settings for Redis vector search
	// +optional
	Index RedisCacheIndex `json:"index,omitempty"`

	// Search settings for Redis queries
	// +optional
	Search RedisCacheSearch `json:"search,omitempty"`

	// Development settings for Redis cache
	// +optional
	Development RedisCacheDevelopment `json:"development,omitempty"`
}

// RedisCacheConnection defines Redis connection parameters.
type RedisCacheConnection struct {
	// Host is the Redis server hostname or IP address
	// Example: "redis.default.svc.cluster.local"
	// +optional
	Host string `json:"host,omitempty"`

	// Port is the Redis server port
	// +kubebuilder:default=6379
	// +kubebuilder:validation:Minimum=1
	// +kubebuilder:validation:Maximum=65535
	// +optional
	Port int `json:"port,omitempty"`

	// Database is the Redis database number to use
	// +kubebuilder:default=0
	// +kubebuilder:validation:Minimum=0
	// +optional
	Database int `json:"database,omitempty"`

	// Password for Redis authentication (plaintext - consider using PasswordSecretRef instead)
	// +optional
	Password string `json:"password,omitempty"`

	// PasswordSecretRef references a Secret containing the Redis password
	// Preferred over plaintext Password field for security
	// +optional
	PasswordSecretRef *corev1.SecretKeySelector `json:"password_secret_ref,omitempty"`

	// Timeout for Redis operations in seconds
	// +kubebuilder:default=30
	// +kubebuilder:validation:Minimum=0
	// +optional
	Timeout int `json:"timeout,omitempty"`

	// TLS configuration for secure Redis connections
	// +optional
	TLS RedisCacheTLS `json:"tls,omitempty"`
}

// RedisCacheTLS defines TLS settings for Redis connections.
type RedisCacheTLS struct {
	// Enabled controls whether to use TLS for Redis connection
	// +kubebuilder:default=false
	// +optional
	Enabled bool `json:"enabled,omitempty"`

	// CertFile is the path to client certificate file
	// +optional
	CertFile string `json:"cert_file,omitempty"`

	// KeyFile is the path to client key file
	// +optional
	KeyFile string `json:"key_file,omitempty"`

	// CAFile is the path to CA certificate file
	// +optional
	CAFile string `json:"ca_file,omitempty"`
}

// RedisCacheIndex defines Redis vector index configuration.
type RedisCacheIndex struct {
	// Name of the Redis index
	// +kubebuilder:default="semantic_cache_idx"
	// +optional
	Name string `json:"name,omitempty"`

	// Prefix for Redis keys
	// +kubebuilder:default="doc:"
	// +optional
	Prefix string `json:"prefix,omitempty"`

	// VectorField configuration for embeddings
	// +optional
	VectorField RedisCacheVectorField `json:"vector_field,omitempty"`

	// IndexType specifies the index algorithm
	// Options: "HNSW" (recommended), "FLAT"
	// +kubebuilder:default="HNSW"
	// +kubebuilder:validation:Enum=HNSW;FLAT
	// +optional
	IndexType string `json:"index_type,omitempty"`

	// Params for HNSW index
	// +optional
	Params RedisCacheIndexParams `json:"params,omitempty"`
}

// RedisCacheVectorField defines vector field configuration.
type RedisCacheVectorField struct {
	// Name of the vector field
	// +kubebuilder:default="embedding"
	// +optional
	Name string `json:"name,omitempty"`

	// Dimension of the embedding vectors
	// For BERT: 384, for Qwen3: 1024, for Gemma: 768
	// +kubebuilder:validation:Minimum=1
	// +optional
	Dimension int `json:"dimension,omitempty"`

	// MetricType for vector similarity
	// Options: "COSINE", "IP" (inner product), "L2" (Euclidean)
	// +kubebuilder:default="COSINE"
	// +kubebuilder:validation:Enum=COSINE;IP;L2
	// +optional
	MetricType string `json:"metric_type,omitempty"`
}

// RedisCacheIndexParams defines HNSW index parameters.
type RedisCacheIndexParams struct {
	// M is the number of bi-directional links per node
	// Higher values = better recall, more memory
	// +kubebuilder:default=16
	// +kubebuilder:validation:Minimum=2
	// +optional
	M int `json:"M,omitempty"`

	// EfConstruction is the size of dynamic candidate list during construction
	// Higher values = better quality, slower indexing
	// +kubebuilder:default=64
	// +kubebuilder:validation:Minimum=1
	// +optional
	EfConstruction int `json:"efConstruction,omitempty"`
}

// RedisCacheSearch defines Redis search parameters.
type RedisCacheSearch struct {
	// TopK is the number of results to return from vector search
	// +kubebuilder:default=1
	// +kubebuilder:validation:Minimum=1
	// +optional
	TopK int `json:"topk,omitempty"`
}

// RedisCacheDevelopment defines development-mode settings.
type RedisCacheDevelopment struct {
	// DropIndexOnStartup clears the index when router starts (for testing)
	// +kubebuilder:default=false
	// +optional
	DropIndexOnStartup bool `json:"drop_index_on_startup,omitempty"`

	// AutoCreateIndex automatically creates the index if it doesn't exist
	// +kubebuilder:default=true
	// +optional
	AutoCreateIndex bool `json:"auto_create_index,omitempty"`

	// VerboseErrors includes detailed error messages in logs
	// +kubebuilder:default=true
	// +optional
	VerboseErrors bool `json:"verbose_errors,omitempty"`
}

// MilvusCacheConfig defines Milvus cache backend configuration.
// Configure these settings when using Milvus as the semantic cache backend.
type MilvusCacheConfig struct {
	// Connection settings for Milvus server
	// +optional
	Connection MilvusCacheConnection `json:"connection,omitempty"`

	// Collection settings for Milvus
	// +optional
	Collection MilvusCacheCollection `json:"collection,omitempty"`

	// Search settings for Milvus queries
	// +optional
	Search MilvusCacheSearch `json:"search,omitempty"`

	// Performance tuning for Milvus
	// +optional
	Performance MilvusCachePerformance `json:"performance,omitempty"`

	// DataManagement settings for TTL and compaction
	// +optional
	DataManagement MilvusCacheDataManagement `json:"data_management,omitempty"`

	// Development settings for Milvus cache
	// +optional
	Development MilvusCacheDevelopment `json:"development,omitempty"`
}

// MilvusCacheConnection defines Milvus connection parameters.
type MilvusCacheConnection struct {
	// Host is the Milvus server hostname or IP address
	// +optional
	Host string `json:"host,omitempty"`

	// Port is the Milvus server port
	// +kubebuilder:default=19530
	// +kubebuilder:validation:Minimum=1
	// +kubebuilder:validation:Maximum=65535
	// +optional
	Port int `json:"port,omitempty"`

	// Database name in Milvus
	// +kubebuilder:default="semantic_router_cache"
	// +optional
	Database string `json:"database,omitempty"`

	// Timeout for Milvus operations in seconds
	// +kubebuilder:default=30
	// +kubebuilder:validation:Minimum=0
	// +optional
	Timeout int `json:"timeout,omitempty"`

	// Auth configuration for Milvus authentication
	// +optional
	Auth MilvusCacheAuth `json:"auth,omitempty"`

	// TLS configuration for secure Milvus connections
	// +optional
	TLS MilvusCacheTLS `json:"tls,omitempty"`
}

// MilvusCacheAuth defines Milvus authentication.
type MilvusCacheAuth struct {
	// Enabled controls whether to use authentication
	// +kubebuilder:default=false
	// +optional
	Enabled bool `json:"enabled,omitempty"`

	// Username for Milvus authentication
	// +optional
	Username string `json:"username,omitempty"`

	// Password for Milvus authentication (plaintext - consider using PasswordSecretRef instead)
	// +optional
	Password string `json:"password,omitempty"`

	// PasswordSecretRef references a Secret containing the Milvus password
	// Preferred over plaintext Password field for security
	// +optional
	PasswordSecretRef *corev1.SecretKeySelector `json:"password_secret_ref,omitempty"`
}

// MilvusCacheTLS defines TLS settings for Milvus connections.
type MilvusCacheTLS struct {
	// Enabled controls whether to use TLS
	// +kubebuilder:default=false
	// +optional
	Enabled bool `json:"enabled,omitempty"`

	// CertFile is the path to client certificate file
	// +optional
	CertFile string `json:"cert_file,omitempty"`

	// KeyFile is the path to client key file
	// +optional
	KeyFile string `json:"key_file,omitempty"`

	// CAFile is the path to CA certificate file
	// +optional
	CAFile string `json:"ca_file,omitempty"`
}

// MilvusCacheCollection defines Milvus collection configuration.
type MilvusCacheCollection struct {
	// Name of the Milvus collection
	// +kubebuilder:default="semantic_cache"
	// +optional
	Name string `json:"name,omitempty"`

	// Description of the collection
	// +kubebuilder:default="Semantic cache for LLM request-response pairs"
	// +optional
	Description string `json:"description,omitempty"`

	// VectorField configuration for embeddings
	// +optional
	VectorField MilvusCacheVectorField `json:"vector_field,omitempty"`

	// Index configuration for the collection
	// +optional
	Index MilvusCacheCollectionIndex `json:"index,omitempty"`
}

// MilvusCacheVectorField defines vector field configuration.
type MilvusCacheVectorField struct {
	// Name of the vector field
	// +kubebuilder:default="embedding"
	// +optional
	Name string `json:"name,omitempty"`

	// Dimension of the embedding vectors
	// +kubebuilder:validation:Minimum=1
	// +optional
	Dimension int `json:"dimension,omitempty"`

	// MetricType for vector similarity
	// Options: "IP" (inner product), "L2", "COSINE"
	// +kubebuilder:default="IP"
	// +kubebuilder:validation:Enum=IP;L2;COSINE
	// +optional
	MetricType string `json:"metric_type,omitempty"`
}

// MilvusCacheCollectionIndex defines collection index settings.
type MilvusCacheCollectionIndex struct {
	// Type of index algorithm
	// +kubebuilder:default="HNSW"
	// +kubebuilder:validation:Enum=HNSW;IVF_FLAT;IVF_SQ8;IVF_PQ
	// +optional
	Type string `json:"type,omitempty"`

	// Params for the index
	// +optional
	Params MilvusCacheIndexParams `json:"params,omitempty"`
}

// MilvusCacheIndexParams defines index parameters.
type MilvusCacheIndexParams struct {
	// M is the number of bi-directional links for HNSW
	// +kubebuilder:default=16
	// +kubebuilder:validation:Minimum=2
	// +optional
	M int `json:"M,omitempty"`

	// EfConstruction for HNSW index building
	// +kubebuilder:default=64
	// +kubebuilder:validation:Minimum=1
	// +optional
	EfConstruction int `json:"efConstruction,omitempty"`
}

// MilvusCacheSearch defines Milvus search parameters.
type MilvusCacheSearch struct {
	// Params for search operations
	// +optional
	Params MilvusCacheSearchParams `json:"params,omitempty"`

	// TopK is the number of results to return
	// +kubebuilder:default=10
	// +kubebuilder:validation:Minimum=1
	// +optional
	TopK int `json:"topk,omitempty"`

	// ConsistencyLevel for search operations
	// Options: "Strong", "Session", "Bounded", "Eventually"
	// +kubebuilder:default="Session"
	// +kubebuilder:validation:Enum=Strong;Session;Bounded;Eventually
	// +optional
	ConsistencyLevel string `json:"consistency_level,omitempty"`
}

// MilvusCacheSearchParams defines search-time parameters.
type MilvusCacheSearchParams struct {
	// Ef is the search-time HNSW parameter
	// +kubebuilder:default=64
	// +kubebuilder:validation:Minimum=1
	// +optional
	Ef int `json:"ef,omitempty"`
}

// MilvusCachePerformance defines performance tuning.
type MilvusCachePerformance struct {
	// ConnectionPool settings
	// +optional
	ConnectionPool MilvusCacheConnectionPool `json:"connection_pool,omitempty"`

	// Batch settings for operations
	// +optional
	Batch MilvusCacheBatch `json:"batch,omitempty"`
}

// MilvusCacheConnectionPool defines connection pool settings.
type MilvusCacheConnectionPool struct {
	// MaxConnections in the pool
	// +kubebuilder:default=10
	// +kubebuilder:validation:Minimum=1
	// +optional
	MaxConnections int `json:"max_connections,omitempty"`

	// MaxIdleConnections to keep
	// +kubebuilder:default=5
	// +kubebuilder:validation:Minimum=0
	// +optional
	MaxIdleConnections int `json:"max_idle_connections,omitempty"`

	// AcquireTimeout in seconds
	// +kubebuilder:default=30
	// +kubebuilder:validation:Minimum=0
	// +optional
	AcquireTimeout int `json:"acquire_timeout,omitempty"`
}

// MilvusCacheBatch defines batch operation settings.
type MilvusCacheBatch struct {
	// InsertBatchSize for bulk inserts
	// +kubebuilder:default=100
	// +kubebuilder:validation:Minimum=1
	// +optional
	InsertBatchSize int `json:"insert_batch_size,omitempty"`

	// Timeout for batch operations in seconds
	// +kubebuilder:default=60
	// +kubebuilder:validation:Minimum=0
	// +optional
	Timeout int `json:"timeout,omitempty"`
}

// MilvusCacheDataManagement defines data lifecycle settings.
type MilvusCacheDataManagement struct {
	// TTL settings for automatic expiration
	// +optional
	TTL MilvusCacheTTL `json:"ttl,omitempty"`

	// Compaction settings
	// +optional
	Compaction MilvusCacheCompaction `json:"compaction,omitempty"`
}

// MilvusCacheTTL defines time-to-live settings.
type MilvusCacheTTL struct {
	// Enabled controls whether TTL is active
	// +kubebuilder:default=false
	// +optional
	Enabled bool `json:"enabled,omitempty"`

	// TimestampField is the field used for TTL calculation
	// +kubebuilder:default="created_at"
	// +optional
	TimestampField string `json:"timestamp_field,omitempty"`

	// CleanupInterval in seconds between cleanup runs
	// +kubebuilder:default=3600
	// +kubebuilder:validation:Minimum=0
	// +optional
	CleanupInterval int `json:"cleanup_interval,omitempty"`
}

// MilvusCacheCompaction defines compaction settings.
type MilvusCacheCompaction struct {
	// Enabled controls whether auto-compaction is active
	// +kubebuilder:default=false
	// +optional
	Enabled bool `json:"enabled,omitempty"`

	// Interval in seconds between compaction runs
	// +kubebuilder:default=86400
	// +kubebuilder:validation:Minimum=0
	// +optional
	Interval int `json:"interval,omitempty"`
}

// MilvusCacheDevelopment defines development-mode settings.
type MilvusCacheDevelopment struct {
	// DropCollectionOnStartup clears the collection when router starts (for testing)
	// +kubebuilder:default=false
	// +optional
	DropCollectionOnStartup bool `json:"drop_collection_on_startup,omitempty"`

	// AutoCreateCollection automatically creates the collection if it doesn't exist
	// +kubebuilder:default=true
	// +optional
	AutoCreateCollection bool `json:"auto_create_collection,omitempty"`

	// VerboseErrors includes detailed error messages in logs
	// +kubebuilder:default=true
	// +optional
	VerboseErrors bool `json:"verbose_errors,omitempty"`
}

// HNSWCacheConfig defines HNSW index configuration for hybrid/in-memory backends.
type HNSWCacheConfig struct {
	// UseHNSW enables HNSW indexing for faster similarity search
	// +kubebuilder:default=false
	// +optional
	UseHNSW bool `json:"use_hnsw,omitempty"`

	// M is the number of bi-directional links per node
	// +kubebuilder:default=16
	// +kubebuilder:validation:Minimum=2
	// +optional
	M int `json:"hnsw_m,omitempty"`

	// EfConstruction is the size of dynamic candidate list during construction
	// +kubebuilder:default=200
	// +kubebuilder:validation:Minimum=1
	// +optional
	EfConstruction int `json:"hnsw_ef_construction,omitempty"`

	// MaxMemoryEntries limits in-memory entries for hybrid backend
	// +kubebuilder:default=1000
	// +kubebuilder:validation:Minimum=0
	// +optional
	MaxMemoryEntries int `json:"max_memory_entries,omitempty"`
}

// EmbeddingModelsConfig defines configuration for embedding models
type EmbeddingModelsConfig struct {
	// Path to Qwen3-Embedding-0.6B model directory
	// Qwen3 provides 32K context and high quality embeddings (1024 dimensions)
	// +optional
	Qwen3ModelPath string `json:"qwen3_model_path,omitempty"`

	// Path to EmbeddingGemma-300M model directory
	// Gemma provides 8K context and fast embeddings (768 dimensions)
	// +optional
	GemmaModelPath string `json:"gemma_model_path,omitempty"`

	// Path to mmBERT 2D Matryoshka embedding model directory
	// Supports layer early exit (3/6/11/22) and dimension reduction (64-768)
	// +optional
	MmBertModelPath string `json:"mmbert_model_path,omitempty"`

	// Use CPU for inference (default: true)
	// +kubebuilder:default=true
	// +optional
	UseCPU bool `json:"use_cpu,omitempty"`

	// HNSW configuration for embedding-based classification
	// +optional
	HNSWConfig *HNSWEmbeddingConfig `json:"hnsw_config,omitempty"`
}

// HNSWEmbeddingConfig contains settings for embedding classification with HNSW indexing
type HNSWEmbeddingConfig struct {
	// ModelType specifies which embedding model to use
	// Options: "qwen3" (1024-dim, 32K context), "gemma" (768-dim, 8K context), "mmbert" (64-768-dim, multilingual)
	// +kubebuilder:validation:Enum=qwen3;gemma;mmbert
	// +optional
	ModelType string `json:"model_type,omitempty"`

	// PreloadEmbeddings enables precomputing candidate embeddings at startup
	// +kubebuilder:default=true
	// +optional
	PreloadEmbeddings bool `json:"preload_embeddings,omitempty"`

	// TargetDimension is the embedding dimension to use (default: 768)
	// Supported dimensions: 64, 128, 256, 512, 768
	// For mmBERT, lower dimensions provide faster performance with slight accuracy trade-off
	// +kubebuilder:validation:Enum=64;128;256;512;768
	// +optional
	TargetDimension int `json:"target_dimension,omitempty"`

	// TargetLayer is the layer for mmBERT early exit (only used when ModelType is "mmbert")
	// Layer 3: ~7x speedup, Layer 6: ~3.6x speedup, Layer 11: ~2x speedup, Layer 22: full accuracy
	// +kubebuilder:validation:Enum=3;6;11;22
	// +optional
	TargetLayer int `json:"target_layer,omitempty"`

	// EnableSoftMatching enables soft matching mode
	// +kubebuilder:default=true
	// +optional
	EnableSoftMatching bool `json:"enable_soft_matching,omitempty"`

	// MinScoreThreshold for matching (0.0-1.0). Stored as string to avoid float precision issues.
	// +kubebuilder:default="0.5"
	// +kubebuilder:validation:Pattern=`^0(\.[0-9]+)?$|^1(\.0+)?$`
	// +optional
	MinScoreThreshold string `json:"min_score_threshold,omitempty"`
}

// ComplexityRulesConfig defines complexity-based signal classification
type ComplexityRulesConfig struct {
	// Name of the complexity rule (e.g., "code-complexity", "reasoning-complexity")
	Name string `json:"name"`

	// Description of what this rule classifies
	// +optional
	Description string `json:"description,omitempty"`

	// Threshold for difficulty classification (0.0-1.0). Stored as string to avoid float precision issues.
	// Queries scoring above this threshold are classified as "hard"
	// +kubebuilder:validation:Pattern=`^0(\.[0-9]+)?$|^1(\.0+)?$`
	// +optional
	Threshold string `json:"threshold,omitempty"`

	// Hard candidates represent complex/difficult examples
	Hard ComplexityCandidates `json:"hard"`

	// Easy candidates represent simple/easy examples
	Easy ComplexityCandidates `json:"easy"`

	// Composer allows filtering based on other signals (e.g., only apply this rule if domain:medical)
	// +optional
	Composer *RuleComposition `json:"composer,omitempty"`
}

// ComplexityCandidates defines candidate examples for complexity classification
type ComplexityCandidates struct {
	// List of candidate phrases or examples
	Candidates []string `json:"candidates"`
}

// RuleComposition defines how to compose/filter rules based on other signals
type RuleComposition struct {
	// Operator for combining conditions (AND, OR)
	// +kubebuilder:validation:Enum=AND;OR
	Operator string `json:"operator"`

	// List of conditions that must be met
	Conditions []CompositionCondition `json:"conditions"`
}

// CompositionCondition defines a single composition condition
type CompositionCondition struct {
	// Type of signal to check (e.g., "domain", "language", "category")
	Type string `json:"type"`

	// Name of the specific signal/rule value to match
	Name string `json:"name"`
}

// DecisionConfig defines a routing decision
type DecisionConfig struct {
	// Name is the unique identifier for this decision
	Name string `json:"name" yaml:"name"`

	// Description provides information about what this decision handles
	// +optional
	Description string `json:"description,omitempty" yaml:"description,omitempty"`

	// Priority is used for decision ordering - higher priority decisions are evaluated first
	// +optional
	Priority int `json:"priority,omitempty" yaml:"priority,omitempty"`

	// Rules defines the combination of conditions using AND/OR logic
	Rules RuleCombinationConfig `json:"rules" yaml:"rules"`

	// ModelRefs contains model references for this decision
	// +optional
	ModelRefs []ModelRefConfig `json:"modelRefs,omitempty" yaml:"modelRefs,omitempty"`

	// PreferredEndpoints specifies which vLLM endpoints to prefer for this decision
	// +optional
	PreferredEndpoints []string `json:"preferred_endpoints,omitempty" yaml:"preferred_endpoints,omitempty"`

	// Plugins contains policy configurations applied after rule matching
	// +optional
	Plugins []runtime.RawExtension `json:"plugins,omitempty" yaml:"plugins,omitempty"`
}

// RuleCombinationConfig defines how to combine multiple rule conditions
type RuleCombinationConfig struct {
	// Operator specifies how to combine conditions: "AND" or "OR"
	// +kubebuilder:validation:Enum=AND;OR
	Operator string `json:"operator" yaml:"operator"`

	// Conditions is the list of rule references to evaluate
	Conditions []RuleConditionConfig `json:"conditions" yaml:"conditions"`
}

// RuleConditionConfig references a specific rule by type and name
type RuleConditionConfig struct {
	// Type specifies the rule type: "keyword", "embedding", "domain", "complexity", or "fact_check"
	// +kubebuilder:validation:Enum=keyword;embedding;domain;complexity;fact_check;context
	Type string `json:"type" yaml:"type"`

	// Name is the name of the rule to reference
	Name string `json:"name" yaml:"name"`
}

// ModelRefConfig defines a model reference for routing
type ModelRefConfig struct {
	// Model name to route to
	Model string `json:"model" yaml:"model"`

	// LoRAName is the optional LoRA adapter name
	// +optional
	LoRAName string `json:"lora_name,omitempty" yaml:"lora_name,omitempty"`

	// UseReasoning enables reasoning mode for this model
	// +optional
	UseReasoning *bool `json:"use_reasoning,omitempty" yaml:"use_reasoning,omitempty"`

	// ReasoningEffort specifies the reasoning effort level (low, medium, high)
	// +optional
	ReasoningEffort string `json:"reasoning_effort,omitempty" yaml:"reasoning_effort,omitempty"`
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
