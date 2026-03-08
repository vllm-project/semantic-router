package config

type APIConfig struct {
	BatchClassification struct {
		Metrics BatchClassificationMetricsConfig `yaml:"metrics,omitempty"`
	} `yaml:"batch_classification"`
}

type ObservabilityConfig struct {
	Tracing TracingConfig `yaml:"tracing"`
	Metrics MetricsConfig `yaml:"metrics"`
}

type MetricsConfig struct {
	Enabled         *bool                 `yaml:"enabled,omitempty"`
	WindowedMetrics WindowedMetricsConfig `yaml:"windowed_metrics"`
}

type WindowedMetricsConfig struct {
	Enabled              bool     `yaml:"enabled"`
	TimeWindows          []string `yaml:"time_windows,omitempty"`
	UpdateInterval       string   `yaml:"update_interval,omitempty"`
	ModelMetrics         bool     `yaml:"model_metrics"`
	QueueDepthEstimation bool     `yaml:"queue_depth_estimation"`
	MaxModels            int      `yaml:"max_models,omitempty"`
}

type TracingConfig struct {
	Enabled  bool                  `yaml:"enabled"`
	Provider string                `yaml:"provider,omitempty"`
	Exporter TracingExporterConfig `yaml:"exporter"`
	Sampling TracingSamplingConfig `yaml:"sampling"`
	Resource TracingResourceConfig `yaml:"resource"`
}

type TracingExporterConfig struct {
	Type     string `yaml:"type"`
	Endpoint string `yaml:"endpoint,omitempty"`
	Insecure bool   `yaml:"insecure,omitempty"`
}

type TracingSamplingConfig struct {
	Type string  `yaml:"type"`
	Rate float64 `yaml:"rate,omitempty"`
}

type TracingResourceConfig struct {
	ServiceName           string `yaml:"service_name"`
	ServiceVersion        string `yaml:"service_version,omitempty"`
	DeploymentEnvironment string `yaml:"deployment_environment,omitempty"`
}

type BatchClassificationMetricsConfig struct {
	SampleRate                float64                `yaml:"sample_rate,omitempty"`
	BatchSizeRanges           []BatchSizeRangeConfig `yaml:"batch_size_ranges,omitempty"`
	DurationBuckets           []float64              `yaml:"duration_buckets,omitempty"`
	SizeBuckets               []float64              `yaml:"size_buckets,omitempty"`
	Enabled                   bool                   `yaml:"enabled,omitempty"`
	DetailedGoroutineTracking bool                   `yaml:"detailed_goroutine_tracking,omitempty"`
	HighResolutionTiming      bool                   `yaml:"high_resolution_timing,omitempty"`
}

type BatchSizeRangeConfig struct {
	Min   int    `yaml:"min"`
	Max   int    `yaml:"max"`
	Label string `yaml:"label"`
}
