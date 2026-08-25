// Package accesscapacity implements the opt-in Router access-control capacity
// gate. It deliberately uses the production publication, access-runtime,
// quota-runtime, and usage-stream implementations against an isolated Redis
// key prefix.
package accesscapacity

import (
	"fmt"
	"runtime"
	"strings"
	"time"
)

const (
	DefaultKeyCount    = 10_000
	DefaultReplicas    = 4
	DefaultConcurrency = 64
)

// Config contains only non-secret workload controls. Redis connection options
// are supplied separately so reports can never serialize credentials.
type Config struct {
	KeyCount          int
	Replicas          int
	Concurrency       int
	RequestLimit      int
	OperationTimeout  time.Duration
	UsageDrainTimeout time.Duration
	KeyPrefix         string
	OutputRoot        string
	KeepData          bool
	Thresholds        Thresholds
}

type Thresholds struct {
	MaxAdmissionP99       time.Duration
	MaxUsageLagP99        time.Duration
	MinProjectionKeysPerS float64
	MaxProjectionBytesKey int64
	MaxEventBytes         int64
}

func DefaultConfig() Config {
	return Config{
		KeyCount:          DefaultKeyCount,
		Replicas:          DefaultReplicas,
		Concurrency:       DefaultConcurrency,
		RequestLimit:      12,
		OperationTimeout:  20 * time.Minute,
		UsageDrainTimeout: 30 * time.Second,
		OutputRoot:        ".agent-harness/access-capacity",
		Thresholds: Thresholds{
			MaxAdmissionP99:       100 * time.Millisecond,
			MaxUsageLagP99:        5 * time.Second,
			MinProjectionKeysPerS: 100,
			MaxProjectionBytesKey: 32 * 1024,
			MaxEventBytes:         16 * 1024,
		},
	}
}

func (c Config) Validate() error {
	if c.KeyCount < 1 || c.KeyCount > 1_000_000 {
		return fmt.Errorf("key count must be between 1 and 1,000,000")
	}
	if c.Replicas < 2 || c.Replicas > 128 {
		return fmt.Errorf("replica count must be between 2 and 128")
	}
	if c.Concurrency < c.Replicas || c.Concurrency > 16_384 {
		return fmt.Errorf("concurrency must be between the replica count and 16,384")
	}
	if c.RequestLimit < 4 || c.RequestLimit > 1_000_000 {
		return fmt.Errorf("request limit must be between 4 and 1,000,000")
	}
	if c.OperationTimeout < time.Minute || c.OperationTimeout > 24*time.Hour {
		return fmt.Errorf("operation timeout must be between one minute and 24 hours")
	}
	if c.UsageDrainTimeout < time.Second || c.UsageDrainTimeout > time.Hour {
		return fmt.Errorf("usage drain timeout must be between one second and one hour")
	}
	if c.KeyPrefix != "" && (!canonicalPrefix(c.KeyPrefix) || len(c.KeyPrefix) > 96) {
		return fmt.Errorf("key prefix must be a bounded canonical colon-separated identifier")
	}
	if strings.TrimSpace(c.OutputRoot) == "" {
		return fmt.Errorf("output root is required")
	}
	if c.Thresholds.MaxAdmissionP99 <= 0 || c.Thresholds.MaxUsageLagP99 <= 0 ||
		c.Thresholds.MinProjectionKeysPerS <= 0 || c.Thresholds.MaxProjectionBytesKey <= 0 ||
		c.Thresholds.MaxEventBytes <= 0 {
		return fmt.Errorf("every capacity threshold must be positive")
	}
	return nil
}

func canonicalPrefix(value string) bool {
	if value == "" || strings.TrimSpace(value) != value {
		return false
	}
	for _, segment := range strings.Split(value, ":") {
		if segment == "" {
			return false
		}
		for index, character := range segment {
			if (character >= 'a' && character <= 'z') ||
				(character >= 'A' && character <= 'Z') ||
				(character >= '0' && character <= '9') ||
				(index > 0 && (character == '.' || character == '_' || character == '-')) {
				continue
			}
			return false
		}
	}
	return true
}

func workloadEnvironment() Environment {
	return Environment{GoVersion: runtime.Version()}
}
