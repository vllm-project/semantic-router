package extproc

import (
	"time"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// OpenAIModel represents a single model in the OpenAI /v1/models response
type OpenAIModel struct {
	ID          string `json:"id"`
	Object      string `json:"object"`
	Created     int64  `json:"created"`
	OwnedBy     string `json:"owned_by"`
	Description string `json:"description,omitempty"` // Optional description for Chat UI
	LogoURL     string `json:"logo_url,omitempty"`    // Optional logo URL for Chat UI
}

// OpenAIModelList is the container for the models list response
type OpenAIModelList struct {
	Object string        `json:"object"`
	Data   []OpenAIModel `json:"data"`
}

// handleModelsRequest handles GET /v1/models requests and returns a direct response
// Whether to include configured models is controlled by the config's IncludeConfigModelsInList setting (default: false)
func (r *OpenAIRouter) handleModelsRequest(_ string) (*ext_proc.ProcessingResponse, error) {
	now := time.Now().Unix()

	builder := newExtProcOpenAIModelListBuilder(now)
	builder.appendAutoAliases(r.Config)
	builder.appendLooperAliases(r.Config)
	builder.appendBackendModels(r.Config)

	resp := OpenAIModelList{
		Object: "list",
		Data:   builder.models,
	}

	return r.createJSONResponse(200, resp), nil
}

type extProcOpenAIModelListBuilder struct {
	now    int64
	models []OpenAIModel
	seen   map[string]bool
}

func newExtProcOpenAIModelListBuilder(now int64) *extProcOpenAIModelListBuilder {
	return &extProcOpenAIModelListBuilder{
		now:  now,
		seen: map[string]bool{},
	}
}

func (b *extProcOpenAIModelListBuilder) appendAutoAliases(cfg *config.RouterConfig) {
	autoModelNames := config.DefaultAutoModelNames()
	if cfg != nil {
		autoModelNames = cfg.EffectiveAutoModelNames()
	}
	for _, model := range autoModelNames {
		b.append(model, "Intelligent Router for Mixture-of-Models")
	}
}

func (b *extProcOpenAIModelListBuilder) appendLooperAliases(cfg *config.RouterConfig) {
	if cfg == nil || !cfg.Looper.IsEnabled() {
		return
	}
	b.appendAll(cfg.ExposedReMoMModelNames(), "ReMoM multi-round reasoning")
	b.appendAll(cfg.ExposedFusionModelNames(), "Fusion multi-model deliberation")
	b.appendAll(cfg.ExposedFlowModelNames(), "Router Flow micro-agent workflows")
}

func (b *extProcOpenAIModelListBuilder) appendBackendModels(cfg *config.RouterConfig) {
	if cfg == nil || !cfg.IncludeConfigModelsInList {
		return
	}
	b.appendAll(cfg.GetAllModels(), "")
}

func (b *extProcOpenAIModelListBuilder) appendAll(models []string, description string) {
	for _, model := range models {
		b.append(model, description)
	}
}

func (b *extProcOpenAIModelListBuilder) append(id string, description string) {
	if id == "" || b.seen[id] {
		return
	}
	b.seen[id] = true
	b.models = append(b.models, OpenAIModel{
		ID:          id,
		Object:      "model",
		Created:     b.now,
		OwnedBy:     "vllm-semantic-router",
		Description: description,
	})
}
