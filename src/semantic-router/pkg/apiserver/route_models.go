//go:build !windows && cgo

package apiserver

import (
	"net/http"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// handleOpenAIModels handles OpenAI-compatible model listing at /v1/models
// It returns the configured auto model name and optionally the underlying models from config.
// Whether to include configured models is controlled by the config's IncludeConfigModelsInList setting (default: false)
func (s *ClassificationAPIServer) handleOpenAIModels(w http.ResponseWriter, _ *http.Request) {
	now := time.Now().Unix()
	cfg := s.currentConfig()

	builder := newOpenAIModelListBuilder(now)
	builder.appendAutoAliases(cfg)
	builder.appendLooperAliases(cfg)
	builder.appendBackendModels(cfg)

	resp := OpenAIModelList{
		Object: "list",
		Data:   builder.models,
	}

	s.writeJSONResponse(w, http.StatusOK, resp)
}

type openAIModelListBuilder struct {
	now    int64
	models []OpenAIModel
	seen   map[string]bool
}

func newOpenAIModelListBuilder(now int64) *openAIModelListBuilder {
	return &openAIModelListBuilder{
		now:  now,
		seen: map[string]bool{},
	}
}

func (b *openAIModelListBuilder) appendAutoAliases(cfg *config.RouterConfig) {
	autoModelNames := config.DefaultAutoModelNames()
	if cfg != nil {
		autoModelNames = cfg.EffectiveAutoModelNames()
	}
	for _, model := range autoModelNames {
		b.append(model, "vllm-semantic-router", "Intelligent Router for Mixture-of-Models")
	}
}

func (b *openAIModelListBuilder) appendLooperAliases(cfg *config.RouterConfig) {
	if cfg == nil || !cfg.Looper.IsEnabled() {
		return
	}
	b.appendAll(cfg.ExposedReMoMModelNames(), "vllm-semantic-router", "ReMoM multi-round reasoning")
	b.appendAll(cfg.ExposedFusionModelNames(), "vllm-semantic-router", "Fusion multi-model deliberation")
	b.appendAll(cfg.ExposedFlowModelNames(), "vllm-semantic-router", "Router Flow micro-agent workflows")
}

func (b *openAIModelListBuilder) appendBackendModels(cfg *config.RouterConfig) {
	if cfg == nil || !cfg.IncludeConfigModelsInList {
		return
	}
	b.appendAll(cfg.GetAllModels(), "upstream-endpoint", "")
}

func (b *openAIModelListBuilder) appendAll(models []string, owner string, description string) {
	for _, model := range models {
		b.append(model, owner, description)
	}
}

func (b *openAIModelListBuilder) append(id string, owner string, description string) {
	if id == "" || b.seen[id] {
		return
	}
	b.seen[id] = true
	b.models = append(b.models, OpenAIModel{
		ID:          id,
		Object:      "model",
		Created:     b.now,
		OwnedBy:     owner,
		Description: description,
	})
}
