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

	// Embedding configuration for embedding-based classification
	// +optional
	EmbeddingConfig *HNSWEmbeddingConfig `json:"embedding_config,omitempty"`
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
