package config

import (
	"reflect"
	"testing"

	modelcatalog "github.com/vllm-project/semantic-router/src/semantic-router/pkg/catalog"
)

const tokenizerInferenceDoc = `
version: v0.5
providers:
  models:
    - name: open
      provider_model_id: served-open
      api_format: chat_completions
      display_name: Open model
      distribution:
        type: open_weights
        source: https://huggingface.co/Org/Open-Model/blob/0123456789abcdef0123456789abcdef01234567/README.md
        license: Apache-2.0
      context_window_size: 131072
      backend_refs:
        - endpoint: 127.0.0.1:8000
          provider: vllm
    - name: served
      provider_model_id: Org/Served-Model
      api_format: chat_completions
      backend_refs:
        - endpoint: 127.0.0.1:8001
          provider: vllm
    - name: hosted
      provider_model_id: org/hosted-model
      api_format: chat_completions
      backend_refs:
        - endpoint: https://api.example.com/v1
          provider: openai
          auth:
            api_key_env: HOSTED_API_KEY
    - name: named
      provider_model_id: Org/Named-Model
      tokenizer: cl100k_base
      api_format: chat_completions
      backend_refs:
        - endpoint: 127.0.0.1:8002
          provider: vllm
recipes:
  - name: default
    decisions:
      - name: route
        priority: 10
        rules:
          operator: AND
          conditions: []
        modelRefs:
          - model: open
            use_reasoning: false
`

// A model that names no tokenizer counts with the one in the Hugging Face
// repo it comes from: its card's open-weights source, else the id a
// self-hosted engine serves it under. A hosted API's id is not a repo.
func TestModelsCountWithTheTokenizerOfTheirHuggingFaceRepo(t *testing.T) {
	cfg, err := ParseYAMLBytes([]byte(tokenizerInferenceDoc))
	if err != nil {
		t.Fatal(err)
	}
	for model, want := range map[string]string{
		"open":   "Org/Open-Model",
		"served": "Org/Served-Model",
		"hosted": "",
		"named":  "cl100k_base",
	} {
		if got := cfg.ModelConfig[model].EffectiveTokenizer(); got != want {
			t.Errorf("%s counts with %q, want %q", model, got, want)
		}
	}
	want := []string{"o200k_base", "Org/Open-Model", "Org/Served-Model", "cl100k_base"}
	if got := cfg.ContextWindowTokenizers(); !reflect.DeepEqual(got, want) {
		t.Errorf("ContextWindowTokenizers() = %q, want %q", got, want)
	}
}

func TestHubRepoOfAnOpenWeightsSource(t *testing.T) {
	for source, want := range map[string]string{
		"https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash":                        "deepseek-ai/DeepSeek-V4.1-Flash",
		"https://huggingface.co/Qwen/Qwen3-8B/blob/0123456789abcdef/README.md":          "Qwen/Qwen3-8B",
		"https://huggingface.co/datasets/org/data":                                      "",
		"https://huggingface.co/spaces/org/demo":                                        "",
		"https://huggingface.co/gpt2":                                                   "",
		"http://huggingface.co/org/model":                                               "",
		"https://github.com/org/model":                                                  "",
		"https://huggingface.co/-org/model":                                             "",
		"https://huggingface.co/org/model%20name":                                       "",
		"https://huggingface.co.example.com/org/model":                                  "",
		"https://user@huggingface.co/org/model":                                         "",
		"https://huggingface.co/org/../model":                                           "",
		"https://huggingface.co/org/model@main":                                         "",
		"https://huggingface.co/org/model/tree/main":                                    "org/model",
		"https://huggingface.co/org/model?library=transformers":                         "org/model",
		"https://huggingface.co/org/model.v2/resolve/main/tokenizer.json?download=true": "org/model.v2",
	} {
		if got := hubRepoOf(modelcatalog.ModelDistribution{Type: "open_weights", Source: source}); got != want {
			t.Errorf("hubRepoOf(%q) = %q, want %q", source, got, want)
		}
	}
	if got := hubRepoOf(modelcatalog.ModelDistribution{Type: "api", Source: "https://huggingface.co/org/model"}); got != "" {
		t.Errorf("a closed model's source names repo %q", got)
	}
}
