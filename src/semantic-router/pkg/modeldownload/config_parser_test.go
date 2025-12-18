package modeldownload

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestExtractModelPaths(t *testing.T) {
	tests := []struct {
		name     string
		config   *config.RouterConfig
		expected []string
	}{
		{
			name: "Extract Qwen3ModelPath",
			config: &config.RouterConfig{
				InlineModels: config.InlineModels{
					EmbeddingModels: config.EmbeddingModels{
						Qwen3ModelPath: "models/Qwen3-Embedding-0.6B",
					},
				},
			},
			expected: []string{"models/Qwen3-Embedding-0.6B"},
		},
		{
			name: "Extract GemmaModelPath",
			config: &config.RouterConfig{
				InlineModels: config.InlineModels{
					EmbeddingModels: config.EmbeddingModels{
						GemmaModelPath: "models/embeddinggemma-300m",
					},
				},
			},
			expected: []string{"models/embeddinggemma-300m"},
		},
		{
			name: "Extract both embedding models",
			config: &config.RouterConfig{
				InlineModels: config.InlineModels{
					EmbeddingModels: config.EmbeddingModels{
						Qwen3ModelPath: "models/Qwen3-Embedding-0.6B",
						GemmaModelPath: "models/embeddinggemma-300m",
					},
				},
			},
			expected: []string{"models/Qwen3-Embedding-0.6B", "models/embeddinggemma-300m"},
		},
		{
			name: "Extract ModelID from classifier",
			config: &config.RouterConfig{
				InlineModels: config.InlineModels{
					Classifier: config.Classifier{
						CategoryModel: config.CategoryModel{
							ModelID: "models/lora_intent_classifier_bert-base-uncased_model",
						},
					},
				},
			},
			expected: []string{"models/lora_intent_classifier_bert-base-uncased_model"},
		},
		{
			name: "Extract multiple model paths",
			config: &config.RouterConfig{
				InlineModels: config.InlineModels{
					EmbeddingModels: config.EmbeddingModels{
						Qwen3ModelPath: "models/mom-embedding",
					},
					Classifier: config.Classifier{
						CategoryModel: config.CategoryModel{
							ModelID: "models/mom-domain-classifier",
						},
					},
				},
			},
			expected: []string{"models/mom-embedding", "models/mom-domain-classifier"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			paths := ExtractModelPaths(tt.config)

			if len(paths) != len(tt.expected) {
				t.Errorf("Expected %d paths, got %d: %v", len(tt.expected), len(paths), paths)
				return
			}

			// Check if all expected paths are present (order doesn't matter)
			pathMap := make(map[string]bool)
			for _, p := range paths {
				pathMap[p] = true
			}

			for _, expected := range tt.expected {
				if !pathMap[expected] {
					t.Errorf("Expected path %s not found in result: %v", expected, paths)
				}
			}
		})
	}
}

func TestIsModelDirectory(t *testing.T) {
	tests := []struct {
		path     string
		expected bool
	}{
		{"models/bert-base-uncased", true},
		{"models/lora_model/adapter_config.json", false},
		{"models/mapping.json", false},
		{"config/tools_db.json", false},
		{"models/Qwen3-Embedding-0.6B", true},
		{"models/embeddinggemma-300m", true},
	}

	for _, tt := range tests {
		t.Run(tt.path, func(t *testing.T) {
			result := isModelDirectory(tt.path)
			if result != tt.expected {
				t.Errorf("isModelDirectory(%s) = %v, expected %v", tt.path, result, tt.expected)
			}
		})
	}
}
