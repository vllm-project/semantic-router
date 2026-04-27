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
