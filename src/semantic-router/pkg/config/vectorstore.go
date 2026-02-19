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

package config

import "fmt"

// VectorStoreConfig holds configuration for the vector store feature.
type VectorStoreConfig struct {
	// Enabled controls whether vector store functionality is active.
	Enabled bool `json:"enabled" yaml:"enabled"`

	// BackendType selects the storage backend: "memory", "milvus", or "llama_stack".
	BackendType string `json:"backend_type" yaml:"backend_type"`

	// FileStorageDir is the base directory for uploaded file storage.
	// Default: "/var/lib/vsr/data"
	FileStorageDir string `json:"file_storage_dir,omitempty" yaml:"file_storage_dir,omitempty"`

	// MaxFileSizeMB limits the maximum file upload size in megabytes.
	// Default: 50
	MaxFileSizeMB int `json:"max_file_size_mb,omitempty" yaml:"max_file_size_mb,omitempty"`

	// EmbeddingModel specifies the model for document embeddings.
	// Options: "bert" (default), "qwen3", "gemma", "mmbert"
	EmbeddingModel string `json:"embedding_model,omitempty" yaml:"embedding_model,omitempty"`

	// EmbeddingDimension is the dimensionality of the embedding vectors.
	// Default: 768
	EmbeddingDimension int `json:"embedding_dimension,omitempty" yaml:"embedding_dimension,omitempty"`

	// IngestionWorkers is the number of concurrent ingestion pipeline workers.
	// Default: 2
	IngestionWorkers int `json:"ingestion_workers,omitempty" yaml:"ingestion_workers,omitempty"`

	// SupportedFormats lists the allowed file extensions for upload.
	// Default: [".txt", ".md", ".json", ".csv", ".html"]
	SupportedFormats []string `json:"supported_formats,omitempty" yaml:"supported_formats,omitempty"`

	// Milvus holds Milvus-specific configuration (reuses existing MilvusConfig).
	Milvus *MilvusConfig `json:"milvus,omitempty" yaml:"milvus,omitempty"`

	// Memory holds in-memory backend configuration.
	Memory *VectorStoreMemoryConfig `json:"memory,omitempty" yaml:"memory,omitempty"`

	// LlamaStack holds Llama Stack backend configuration.
	// When backend_type is "llama_stack", vSR delegates vector storage and
	// embedding to a locally-running Llama Stack instance via its REST API.
	// Llama Stack handles embedding internally, so vSR's CandleEmbedder is
	// not used for insert/search — only Llama Stack's configured model is used.
	LlamaStack *LlamaStackVectorStoreConfig `json:"llama_stack,omitempty" yaml:"llama_stack,omitempty"`
}

// VectorStoreMemoryConfig holds configuration for the in-memory backend.
type VectorStoreMemoryConfig struct {
	// MaxEntriesPerStore limits entries per vector store collection.
	// Default: 100000
	MaxEntriesPerStore int `json:"max_entries_per_store,omitempty" yaml:"max_entries_per_store,omitempty"`
}

// LlamaStackVectorStoreConfig holds configuration for the Llama Stack backend.
type LlamaStackVectorStoreConfig struct {
	// Endpoint is the base URL of the Llama Stack server (e.g. "http://localhost:8321").
	Endpoint string `json:"endpoint" yaml:"endpoint"`

	// AuthToken is an optional bearer token for Llama Stack authentication.
	AuthToken string `json:"auth_token,omitempty" yaml:"auth_token,omitempty"`

	// EmbeddingModel is the embedding model ID registered in Llama Stack
	// (e.g. "all-MiniLM-L6-v2"). Llama Stack uses this to embed chunks and queries.
	// If empty, Llama Stack uses its configured default embedding model.
	EmbeddingModel string `json:"embedding_model,omitempty" yaml:"embedding_model,omitempty"`

	// RequestTimeoutSeconds is the HTTP request timeout in seconds. Default: 30.
	RequestTimeoutSeconds int `json:"request_timeout_seconds,omitempty" yaml:"request_timeout_seconds,omitempty"`
}

// Validate checks the vector store configuration for errors.
func (c *VectorStoreConfig) Validate() error {
	if !c.Enabled {
		return nil
	}

	switch c.BackendType {
	case "memory", "milvus", "llama_stack":
		// valid
	case "":
		return fmt.Errorf("vector_store.backend_type is required when enabled")
	default:
		return fmt.Errorf("vector_store.backend_type must be 'memory', 'milvus', or 'llama_stack', got '%s'", c.BackendType)
	}

	if c.BackendType == "milvus" && c.Milvus == nil {
		return fmt.Errorf("vector_store.milvus configuration is required when backend_type is 'milvus'")
	}

	if c.BackendType == "llama_stack" {
		if c.LlamaStack == nil {
			return fmt.Errorf("vector_store.llama_stack configuration is required when backend_type is 'llama_stack'")
		}
		if c.LlamaStack.Endpoint == "" {
			return fmt.Errorf("vector_store.llama_stack.endpoint is required")
		}
	}

	return nil
}

// ApplyDefaults fills in default values for unset fields.
func (c *VectorStoreConfig) ApplyDefaults() {
	if c.FileStorageDir == "" {
		c.FileStorageDir = "/var/lib/vsr/data"
	}
	if c.MaxFileSizeMB <= 0 {
		c.MaxFileSizeMB = 50
	}
	if c.EmbeddingModel == "" {
		c.EmbeddingModel = "bert"
	}
	if c.EmbeddingDimension <= 0 {
		// Default dimension depends on model: BERT (all-MiniLM) = 384, others = 768
		if c.EmbeddingModel == "bert" {
			c.EmbeddingDimension = 384
		} else {
			c.EmbeddingDimension = 768
		}
	}
	if c.IngestionWorkers <= 0 {
		c.IngestionWorkers = 2
	}
	if len(c.SupportedFormats) == 0 {
		c.SupportedFormats = []string{".txt", ".md", ".json", ".csv", ".html"}
	}
}
