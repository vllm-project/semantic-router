// Package publicmodels builds the OpenAI-compatible model catalog exposed by
// the router. It keeps request-facing routing metadata independent from model
// names, descriptions, and provider branding.
package publicmodels

import "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"

const (
	routerOwner           = "vllm-semantic-router"
	upstreamEndpointOwner = "upstream-endpoint"
)

// ResolutionKind describes the stable request-handling boundary exposed to
// clients. Internal routing and orchestration modes intentionally remain out
// of this contract so they can evolve without requiring client changes.
type ResolutionKind string

const (
	ResolutionVirtual     ResolutionKind = "virtual"
	ResolutionPassthrough ResolutionKind = "passthrough"
)

// RoutingMetadata describes the public behavior clients need for model
// discovery. Display metadata and internal execution modes are not control
// signals.
type RoutingMetadata struct {
	API          config.InferenceAPI     `json:"api"`
	Source       config.EntrypointSource `json:"source,omitempty"`
	Resolution   ResolutionKind          `json:"resolution"`
	Selectable   bool                    `json:"selectable"`
	DefaultRoute bool                    `json:"default_route,omitempty"`
	Recipe       config.RecipeName       `json:"recipe,omitempty"`
}

// OpenAIModel represents a single model in the OpenAI /v1/models response.
type OpenAIModel struct {
	ID          string          `json:"id"`
	Object      string          `json:"object"`
	Created     int64           `json:"created"`
	OwnedBy     string          `json:"owned_by"`
	Description string          `json:"description,omitempty"`
	LogoURL     string          `json:"logo_url,omitempty"`
	Routing     RoutingMetadata `json:"routing"`
}

// OpenAIModelList is the container for the models list response.
type OpenAIModelList struct {
	Object string        `json:"object"`
	Data   []OpenAIModel `json:"data"`
}

// NewOpenAIModelList builds the public model catalog from the effective router
// configuration. The source determines only its stable public behavior.
func NewOpenAIModelList(cfg *config.RouterConfig, created int64) OpenAIModelList {
	builder := modelListBuilder{
		created: created,
		seen:    map[string]struct{}{},
	}
	builder.appendEntrypointAliases(cfg)
	builder.appendBackendModels(cfg)

	return OpenAIModelList{
		Object: "list",
		Data:   builder.models,
	}
}

type modelListBuilder struct {
	created int64
	models  []OpenAIModel
	seen    map[string]struct{}
}

func (b *modelListBuilder) appendEntrypointAliases(cfg *config.RouterConfig) {
	for _, entrypoint := range cfg.EffectiveEntrypoints(config.ChatAPI) {
		description := "Entrypoint for the default routing recipe"
		if cfg != nil {
			description = cfg.EntrypointRecipeDescription(entrypoint.Recipe)
		}
		metadata := selectableVirtualRoute(entrypoint.Recipe, entrypoint.Recipe == config.DefaultRecipeName)
		metadata.API, metadata.Source = entrypoint.API, entrypoint.Source
		b.appendAll(entrypoint.ModelNames, routerOwner, description, metadata)
	}
}

func (b *modelListBuilder) appendBackendModels(cfg *config.RouterConfig) {
	if cfg == nil || !cfg.ListBackendModels {
		return
	}
	for _, model := range cfg.GetAllModels() {
		if !cfg.IsSystemOneBackend(model) {
			b.append(model, upstreamEndpointOwner, "", passthroughRoute())
		}
	}
}

func (b *modelListBuilder) appendAll(
	models []string,
	owner string,
	description string,
	routing RoutingMetadata,
) {
	for _, model := range models {
		b.append(model, owner, description, routing)
	}
}

func (b *modelListBuilder) append(
	id string,
	owner string,
	description string,
	routing RoutingMetadata,
) {
	if _, exists := b.seen[id]; id == "" || exists {
		return
	}
	b.seen[id] = struct{}{}
	b.models = append(b.models, OpenAIModel{
		ID:          id,
		Object:      "model",
		Created:     b.created,
		OwnedBy:     owner,
		Description: description,
		Routing:     routing,
	})
}

func selectableVirtualRoute(recipe config.RecipeName, defaultRoute bool) RoutingMetadata {
	return RoutingMetadata{
		Resolution:   ResolutionVirtual,
		Selectable:   true,
		DefaultRoute: defaultRoute,
		Recipe:       recipe,
	}
}

func passthroughRoute() RoutingMetadata {
	return RoutingMetadata{API: config.ChatAPI, Resolution: ResolutionPassthrough, Selectable: true}
}
