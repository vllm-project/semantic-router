package modelruntime

import (
	"context"
	"errors"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
)

type audioCapabilityProvider struct {
	embedding.Provider
	info embedding.ModelInfo
}

func (p audioCapabilityProvider) EmbeddingInfo() embedding.ModelInfo { return p.info }

func TestOriginalAudioRequirementRejectsFeatureOnlyEncoder(t *testing.T) {
	provider := audioCapabilityProvider{info: embedding.ModelInfo{Modalities: []string{"text", "audio"}}}
	requirements := []config.EmbeddingRequirement{{Model: "multimodal", Consumer: "audio signal", Modality: "audio"}}
	providers := map[string]embedding.Provider{"multimodal": provider}
	base, err := embedding.NewFuncProvider("ort", 384, func(context.Context, string) ([]float32, error) {
		t.Fatal("capability validation must precede inference")
		return nil, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	provider.Provider = base
	providers["multimodal"] = provider
	if err := validatePreparedEmbeddings(context.Background(), requirements, providers); !errors.Is(err, binding.ErrCapability) {
		t.Fatalf("feature-only audio encoder accepted for original PCM: %v", err)
	}
	provider.info.Audio = &binding.AudioCapability{MaxSeconds: 30, MaxChannels: 8}
	providers["multimodal"] = provider
	if err := validatePreparedEmbeddings(context.Background(), requirements, providers); err != nil {
		t.Fatalf("original-PCM capability rejected: %v", err)
	}
}
