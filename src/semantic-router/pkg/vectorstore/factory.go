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

import "fmt"

// BackendType constants for selecting a vector store backend.
const (
	BackendTypeMemory     = "memory"
	BackendTypeMilvus     = "milvus"
	BackendTypeLlamaStack = "llama_stack"
)

// BackendConfigs aggregates configuration for all supported backends.
// Only the config matching the selected backendType is used; others are ignored.
type BackendConfigs struct {
	Memory     MemoryBackendConfig
	Milvus     MilvusBackendConfig
	LlamaStack LlamaStackBackendConfig
}

// NewBackend creates a VectorStoreBackend based on the given type.
// Only the config field matching backendType is used; others are ignored.
func NewBackend(backendType string, cfgs BackendConfigs) (VectorStoreBackend, error) {
	switch backendType {
	case BackendTypeMemory:
		return NewMemoryBackend(cfgs.Memory), nil
	case BackendTypeMilvus:
		return NewMilvusBackend(cfgs.Milvus)
	case BackendTypeLlamaStack:
		return NewLlamaStackBackend(cfgs.LlamaStack)
	default:
		return nil, fmt.Errorf("unsupported backend type: %s (supported: %s, %s, %s)",
			backendType, BackendTypeMemory, BackendTypeMilvus, BackendTypeLlamaStack)
	}
}
