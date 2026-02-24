//go:build !windows && cgo

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

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
)

// CandleEmbedder implements the Embedder interface using the Candle FFI
// bindings for embedding generation. It delegates to the appropriate
// candle_binding function based on the configured model type.
type CandleEmbedder struct {
	modelType string
	dimension int
}

// NewCandleEmbedder creates a new CandleEmbedder for the given model and dimension.
func NewCandleEmbedder(modelType string, dimension int) *CandleEmbedder {
	if modelType == "" {
		modelType = "bert"
	}
	return &CandleEmbedder{
		modelType: modelType,
		dimension: dimension,
	}
}

// Embed generates an embedding vector for the given text using the configured model.
func (e *CandleEmbedder) Embed(text string) ([]float32, error) {
	switch e.modelType {
	case "bert":
		return candle_binding.GetEmbedding(text, 0)
	case "qwen3":
		output, err := candle_binding.GetEmbeddingBatched(text, "qwen3", e.dimension)
		if err != nil {
			return nil, fmt.Errorf("qwen3 embedding failed: %w", err)
		}
		return output.Embedding, nil
	case "gemma":
		output, err := candle_binding.GetEmbeddingWithModelType(text, "gemma", e.dimension)
		if err != nil {
			return nil, fmt.Errorf("gemma embedding failed: %w", err)
		}
		return output.Embedding, nil
	case "mmbert":
		output, err := candle_binding.GetEmbeddingWithModelType(text, "mmbert", e.dimension)
		if err != nil {
			return nil, fmt.Errorf("mmbert embedding failed: %w", err)
		}
		return output.Embedding, nil
	default:
		return nil, fmt.Errorf("unsupported embedding model: %s (must be bert, qwen3, gemma, or mmbert)", e.modelType)
	}
}

// Dimension returns the configured embedding dimension.
func (e *CandleEmbedder) Dimension() int {
	return e.dimension
}

// Verify CandleEmbedder satisfies the Embedder interface at compile time.
var _ Embedder = (*CandleEmbedder)(nil)
