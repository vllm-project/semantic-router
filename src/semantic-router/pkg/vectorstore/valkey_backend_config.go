/*
Copyright 2025 vLLM Semantic Router.

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

package vectorstore

import (
	"fmt"
	"strings"
)

// ValkeyBackendConfig holds configuration for the Valkey vector store backend.
type ValkeyBackendConfig struct {
	// Host is the Valkey server hostname (default "localhost").
	Host string
	// Port is the Valkey server port (default 6379).
	Port int
	// Password for Valkey authentication (optional).
	Password string
	// Database number (default 0).
	Database int
	// CollectionPrefix is the prefix for hash keys and index names (default "vsr_vs_").
	CollectionPrefix string
	// IndexM is the HNSW M parameter (default 16).
	IndexM int
	// IndexEf is the HNSW efConstruction parameter (default 200).
	IndexEf int
	// MetricType is the distance metric: "COSINE", "L2", or "IP" (default "COSINE").
	MetricType string
	// ConnectTimeout in seconds (default 10).
	ConnectTimeout int
}

// valkeyDefaults applies default values to a ValkeyBackendConfig, returning
// the resolved host, port, prefix, indexM, indexEf, metricType, and timeout.
// Returns an error if the metric type is unsupported.
func valkeyDefaults(cfg ValkeyBackendConfig) (string, int, string, int, int, string, int, error) {
	host := cfg.Host
	if host == "" {
		host = "localhost"
	}
	port := cfg.Port
	if port <= 0 {
		port = 6379
	}
	prefix := cfg.CollectionPrefix
	if prefix == "" {
		prefix = "vsr_vs_"
	}
	indexM := cfg.IndexM
	if indexM <= 0 {
		indexM = 16
	}
	indexEf := cfg.IndexEf
	if indexEf <= 0 {
		indexEf = 200
	}
	metricType := strings.ToUpper(cfg.MetricType)
	if metricType == "" {
		metricType = "COSINE"
	}
	switch metricType {
	case "COSINE", "L2", "IP":
		// valid
	default:
		return "", 0, "", 0, 0, "", 0, fmt.Errorf("unsupported metric type: %s (supported: COSINE, L2, IP)", metricType)
	}
	timeout := cfg.ConnectTimeout
	if timeout <= 0 {
		timeout = 10
	}
	return host, port, prefix, indexM, indexEf, metricType, timeout, nil
}
