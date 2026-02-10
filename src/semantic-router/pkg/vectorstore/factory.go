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
	BackendTypeMemory = "memory"
	BackendTypeMilvus = "milvus"
)

// NewBackend creates a VectorStoreBackend based on the given type.
// For "memory", milvusCfg is ignored. For "milvus", memoryCfg is ignored.
func NewBackend(backendType string, memoryCfg MemoryBackendConfig, milvusCfg MilvusBackendConfig) (VectorStoreBackend, error) {
	switch backendType {
	case BackendTypeMemory:
		return NewMemoryBackend(memoryCfg), nil
	case BackendTypeMilvus:
		return NewMilvusBackend(milvusCfg)
	default:
		return nil, fmt.Errorf("unsupported backend type: %s (supported: %s, %s)", backendType, BackendTypeMemory, BackendTypeMilvus)
	}
}
