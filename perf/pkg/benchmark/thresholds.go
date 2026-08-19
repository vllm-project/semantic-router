package benchmark

import (
	"bytes"
	"fmt"
	"os"
	"regexp"

	"gopkg.in/yaml.v3"
)

type ThresholdsConfig struct {
	ComponentBenchmarks ComponentBenchmarksThresholds `yaml:"component_benchmarks"`
}

type ComponentBenchmarksThresholds struct {
	Default    RegressionThreshold            `yaml:"default"`
	Benchmarks []BenchmarkRegressionThreshold `yaml:"benchmarks"`
}

// RegressionThreshold separates portable component allocation gates from
// advisory wall-clock timing and external serving metrics. A fixed,
// same-class runner may opt into a blocking ns/op threshold explicitly.
type RegressionThreshold struct {
	MaxAllocsRegressionPercent             float64 `json:"max_allocs_regression_percent" yaml:"max_allocs_regression_percent"`
	MaxBytesRegressionPercent              float64 `json:"max_bytes_regression_percent" yaml:"max_bytes_regression_percent"`
	MaxNsRegressionPercent                 float64 `json:"max_ns_regression_percent,omitempty" yaml:"max_ns_regression_percent,omitempty"`
	MaxP95LatencyRegressionPercent         float64 `json:"max_p95_latency_regression_percent,omitempty" yaml:"max_p95_latency_regression_percent,omitempty"`
	MaxThroughputRegressionPercent         float64 `json:"max_throughput_regression_percent,omitempty" yaml:"max_throughput_regression_percent,omitempty"`
	MaxUpstreamCallsRegressionPercent      float64 `json:"max_upstream_calls_regression_percent,omitempty" yaml:"max_upstream_calls_regression_percent,omitempty"`
	MaxTokenAmplificationRegressionPercent float64 `json:"max_token_amplification_regression_percent,omitempty" yaml:"max_token_amplification_regression_percent,omitempty"`
	GateNsPerOp                            bool    `json:"gate_ns_per_op,omitempty" yaml:"gate_ns_per_op,omitempty"`
}

type BenchmarkRegressionThreshold struct {
	Name                string `yaml:"name"`
	Pattern             string `yaml:"pattern"`
	RegressionThreshold `yaml:",inline"`
}

func LoadThresholds(path string) (*ThresholdsConfig, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read thresholds: %w", err)
	}
	decoder := yaml.NewDecoder(bytes.NewReader(data))
	decoder.KnownFields(true)
	var thresholds ThresholdsConfig
	if err := decoder.Decode(&thresholds); err != nil {
		return nil, fmt.Errorf("parse thresholds: %w", err)
	}
	if thresholds.ComponentBenchmarks.Default.MaxAllocsRegressionPercent <= 0 ||
		thresholds.ComponentBenchmarks.Default.MaxBytesRegressionPercent <= 0 {
		return nil, fmt.Errorf("default allocation and byte thresholds must be positive")
	}
	for _, item := range thresholds.ComponentBenchmarks.Benchmarks {
		if item.Name == "" || item.Pattern == "" {
			return nil, fmt.Errorf("benchmark thresholds require non-empty name and pattern")
		}
		if _, err := regexp.Compile(item.Pattern); err != nil {
			return nil, fmt.Errorf("invalid regexp for benchmark %q: %w", item.Name, err)
		}
	}
	return &thresholds, nil
}
