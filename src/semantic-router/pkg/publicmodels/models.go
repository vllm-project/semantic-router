// Package publicmodels builds the OpenAI-compatible model catalog exposed by
// the router. It keeps request-facing routing metadata independent from model
// names, descriptions, and provider branding.
package publicmodels

import "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"

const (
	routerOwner            = "vllm-semantic-router"
	upstreamEndpointOwner  = "upstream-endpoint"
	autoModelDescription   = "Intelligent Router for Mixture-of-Models"
	remomModelDescription  = "ReMoM multi-round reasoning"
	fusionModelDescription = "Fusion multi-model deliberation"
	flowModelDescription   = "Router Flow micro-agent workflows"
)

// RoutingType identifies how a request-facing model ID is handled by the
// semantic router. Clients should use this field instead of inferring behavior
// from IDs, owners, or display descriptions.
type RoutingType string

const (
	RoutingTypeAutoAlias  RoutingType = "auto_alias"
	RoutingTypeEntrypoint RoutingType = "entrypoint"
	RoutingTypeLooper     RoutingType = "looper"
	RoutingTypeBackend    RoutingType = "backend"
)

// OpenAIModel represents a single model in the OpenAI /v1/models response.
type OpenAIModel struct {
	ID          string      `json:"id"`
	Object      string      `json:"object"`
	Created     int64       `json:"created"`
	OwnedBy     string      `json:"owned_by"`
	Description string      `json:"description,omitempty"`
	LogoURL     string      `json:"logo_url,omitempty"`
	RoutingType RoutingType `json:"routing_type"`
}

// OpenAIModelList is the container for the models list response.
type OpenAIModelList struct {
	Object string        `json:"object"`
	Data   []OpenAIModel `json:"data"`
}

// NewOpenAIModelList builds the public model catalog from the effective router
// configuration. The source of each model determines its routing type.
func NewOpenAIModelList(cfg *config.RouterConfig, created int64) OpenAIModelList {
	builder := modelListBuilder{
		created: created,
		seen:    map[string]bool{},
	}
	builder.appendAutoAliases(cfg)
	builder.appendEntrypointAliases(cfg)
	builder.appendLooperAliases(cfg)
	builder.appendBackendModels(cfg)

	return OpenAIModelList{
		Object: "list",
		Data:   builder.models,
	}
}

type modelListBuilder struct {
	created int64
	models  []OpenAIModel
	seen    map[string]bool
}

func (b *modelListBuilder) appendAutoAliases(cfg *config.RouterConfig) {
	autoModelNames := config.DefaultAutoModelNames()
	if cfg != nil {
		autoModelNames = cfg.EffectiveAutoModelNames()
	}
	b.appendAll(
		autoModelNames,
		routerOwner,
		autoModelDescription,
		RoutingTypeAutoAlias,
	)
}

func (b *modelListBuilder) appendEntrypointAliases(cfg *config.RouterConfig) {
	if cfg == nil {
		return
	}
	for _, entrypoint := range cfg.Entrypoints {
		description := cfg.EntrypointRecipeDescription(entrypoint.Recipe)
		b.appendAll(entrypoint.ModelNames, routerOwner, description, RoutingTypeEntrypoint)
	}
}

func (b *modelListBuilder) appendLooperAliases(cfg *config.RouterConfig) {
	if cfg == nil || !cfg.Looper.IsEnabled() {
		return
	}
	b.appendAll(cfg.ExposedReMoMModelNames(), routerOwner, remomModelDescription, RoutingTypeLooper)
	b.appendAll(cfg.ExposedFusionModelNames(), routerOwner, fusionModelDescription, RoutingTypeLooper)
	b.appendAll(cfg.ExposedFlowModelNames(), routerOwner, flowModelDescription, RoutingTypeLooper)
}

func (b *modelListBuilder) appendBackendModels(cfg *config.RouterConfig) {
	if cfg == nil || !cfg.IncludeConfigModelsInList {
		return
	}
	b.appendAll(cfg.GetAllModels(), upstreamEndpointOwner, "", RoutingTypeBackend)
}

func (b *modelListBuilder) appendAll(
	models []string,
	owner string,
	description string,
	routingType RoutingType,
) {
	for _, model := range models {
		b.append(model, owner, description, routingType)
	}
}

func (b *modelListBuilder) append(
	id string,
	owner string,
	description string,
	routingType RoutingType,
) {
	if id == "" || b.seen[id] {
		return
	}
	b.seen[id] = true
	b.models = append(b.models, OpenAIModel{
		ID:          id,
		Object:      "model",
		Created:     b.created,
		OwnedBy:     owner,
		Description: description,
		RoutingType: routingType,
	})
}
