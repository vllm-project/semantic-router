package config

import (
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/extension"
)

// DecisionPluginType is a route-local plugin type. Its payload's Go type is
// the schema of the plugin's configuration; how the plugin runs is up to the
// runtime that consumes the decoded payload, so a payload that implements a
// runtime's plugin interface runs without changes to the request pipeline.
type DecisionPluginType struct {
	Catalog DecisionPluginCatalogEntry
	// NewPayload returns an empty payload to decode the configuration into.
	NewPayload func() interface{}
	// Strict rejects configuration fields the payload does not declare.
	Strict bool
	// Defaults, when set, fills a decoded payload's unset fields.
	Defaults func(payload interface{})
	// Validate, when set, checks a decoded payload after its defaults.
	Validate func(at PluginAt, payload interface{}) error
	// Aliases are other accepted spellings of the type.
	Aliases []string
}

// PluginAt locates a plugin among a decision's plugins.
type PluginAt struct {
	Decision string
	Index    int
	Type     string
}

// Errorf scopes a message to the plugin.
func (at PluginAt) Errorf(format string, args ...interface{}) error {
	return fmt.Errorf("decision %q plugins[%d] (%s): %w", at.Decision, at.Index, at.Type, fmt.Errorf(format, args...))
}

// DecisionPlugins holds the route-local plugin types. The built-in types
// register first, through the same call as any other package.
var DecisionPlugins = newDecisionPluginRegistry()

func newDecisionPluginRegistry() *extension.Registry[DecisionPluginType] {
	registry := extension.NewRegistry[DecisionPluginType]("decision plugin")
	for _, spec := range builtinDecisionPlugins() {
		if err := registerDecisionPlugin(registry, spec); err != nil {
			panic(err)
		}
	}
	return registry
}

// RegisterDecisionPlugin adds a route-local plugin type, typically from an
// init function.
func RegisterDecisionPlugin(spec DecisionPluginType) error {
	return registerDecisionPlugin(DecisionPlugins, spec)
}

func registerDecisionPlugin(registry *extension.Registry[DecisionPluginType], spec DecisionPluginType) error {
	if spec.NewPayload == nil {
		return fmt.Errorf("decision plugin %q: a payload type is required", spec.Catalog.Type)
	}
	return registry.Register(spec.Catalog.Type, spec, spec.Aliases...)
}

// DecodeDecisionPluginAt decodes plugin's configuration into its type's
// payload, applies the type's defaults and validates the result.
func DecodeDecisionPluginAt(at PluginAt, plugin DecisionPlugin) (interface{}, error) {
	// image_gen was removed when #3076 unified inference protocol translation:
	// the router no longer executes image-generation backends. Route image
	// generation through a vllm-omni modality route speaking the Responses-API
	// hosted image_generation tool instead. See issue #3129.
	if plugin.Type == "image_gen" {
		return nil, fmt.Errorf(
			"decision %q plugins[%d]: plugin %q is unsupported: the image_gen route plugin was removed; use the Responses-API hosted image_generation tool with a vllm-omni modality route",
			at.Decision, at.Index, plugin.Type,
		)
	}
	spec, ok := DecisionPlugins.Lookup(plugin.Type)
	if !ok {
		return nil, fmt.Errorf("decision %q plugins[%d]: unsupported plugin type %q", at.Decision, at.Index, plugin.Type)
	}
	if plugin.Configuration == nil {
		return nil, at.Errorf("configuration is required")
	}
	payload := spec.NewPayload()
	decode := plugin.Configuration.DecodeInto
	if spec.Strict {
		decode = plugin.Configuration.DecodeIntoStrict
	}
	if err := decode(payload); err != nil {
		return nil, at.Errorf("%w", err)
	}
	if spec.Defaults != nil {
		spec.Defaults(payload)
	}
	if spec.Validate != nil {
		if err := spec.Validate(at, payload); err != nil {
			return nil, err
		}
	}
	return payload, nil
}

// PluginOptions configure a plugin type whose payload is a *P.
type PluginOptions[P any] struct {
	// Strict rejects configuration fields P does not declare.
	Strict bool
	// Defaults, when set, fills a decoded payload's unset fields.
	Defaults func(payload *P)
	// Validate, when set, checks a decoded payload after its defaults.
	Validate func(at PluginAt, payload *P) error
	// Aliases are other accepted spellings of the type.
	Aliases []string
}

// NewDecisionPluginType is the plugin type of catalog whose payload is a *P,
// so its defaults and validator take the payload typed.
func NewDecisionPluginType[P any](catalog DecisionPluginCatalogEntry, opts PluginOptions[P]) DecisionPluginType {
	spec := DecisionPluginType{
		Catalog: catalog, NewPayload: func() interface{} { return new(P) }, Strict: opts.Strict, Aliases: opts.Aliases,
	}
	if opts.Defaults != nil {
		spec.Defaults = func(payload interface{}) { opts.Defaults(payload.(*P)) }
	}
	if opts.Validate != nil {
		spec.Validate = func(at PluginAt, payload interface{}) error { return opts.Validate(at, payload.(*P)) }
	}
	return spec
}

// builtinDecisionPlugins are the plugin types the Router runs itself, in
// catalog order.
func builtinDecisionPlugins() []DecisionPluginType {
	return []DecisionPluginType{
		NewDecisionPluginType(DecisionPluginCatalogEntry{Type: DecisionPluginResponseCache, DisplayName: "Response Cache", Description: "Reuse exact or semantically compatible responses."}, PluginOptions[ResponseCachePluginConfig]{
			Strict: true, Aliases: []string{DecisionPluginSemanticCache, "semantic_cache", "response-cache"},
			Validate: func(at PluginAt, p *ResponseCachePluginConfig) error {
				return validateResponseCachePlugin(at.Decision, at.Index, at.Type, p)
			},
		}),
		NewDecisionPluginType(DecisionPluginCatalogEntry{Type: DecisionPluginMemory, DisplayName: "Memory", Description: "Retrieve and store persistent conversation memory."}, PluginOptions[MemoryPluginConfig]{}),
		NewDecisionPluginType(DecisionPluginCatalogEntry{Type: DecisionPluginSystemPrompt, DisplayName: "System Prompt", Description: "Insert or replace the system prompt."}, PluginOptions[SystemPromptPluginConfig]{}),
		NewDecisionPluginType(DecisionPluginCatalogEntry{Type: DecisionPluginHeaderMutation, DisplayName: "Header Mutation", Description: "Add, update, or remove provider-bound headers."}, PluginOptions[HeaderMutationPluginConfig]{}),
		NewDecisionPluginType(DecisionPluginCatalogEntry{Type: DecisionPluginHallucination, DisplayName: "Hallucination", Description: "Apply response hallucination handling."}, PluginOptions[HallucinationPluginConfig]{}),
		NewDecisionPluginType(DecisionPluginCatalogEntry{Type: DecisionPluginRouterReplay, DisplayName: "Router Replay", Description: "Capture bounded request and response replay evidence."}, PluginOptions[RouterReplayPluginConfig]{}),
		NewDecisionPluginType(DecisionPluginCatalogEntry{Type: DecisionPluginRAG, DisplayName: "RAG", Description: "Retrieve external context and inject it into the request."}, PluginOptions[RAGPluginConfig]{}),
		NewDecisionPluginType(DecisionPluginCatalogEntry{Type: DecisionPluginFastResponse, DisplayName: "Fast Response", Description: "Return a fixed response without calling an upstream model."}, PluginOptions[FastResponsePluginConfig]{
			Validate: func(at PluginAt, p *FastResponsePluginConfig) error {
				return validateFastResponsePlugin(at.Decision, at.Index, at.Type, p)
			},
		}),
		NewDecisionPluginType(DecisionPluginCatalogEntry{Type: DecisionPluginTools, DisplayName: "Tools", Description: "Apply route-local tool filtering and selection."}, PluginOptions[ToolsPluginConfig]{}),
		NewDecisionPluginType(DecisionPluginCatalogEntry{Type: DecisionPluginToolSelection, DisplayName: "Tool Selection", Description: "Add or filter tools using semantic retrieval."}, PluginOptions[ToolSelectionPluginConfig]{}),
		NewDecisionPluginType(DecisionPluginCatalogEntry{Type: DecisionPluginRequestParams, DisplayName: "Request Parameters", Description: "Constrain or remove provider request parameters."}, PluginOptions[RequestParamsPluginConfig]{
			Strict: true,
			Validate: func(at PluginAt, p *RequestParamsPluginConfig) error {
				if err := ValidateRequestParamsPluginConfig(p); err != nil {
					return at.Errorf("%w", err)
				}
				return nil
			},
		}),
		NewDecisionPluginType(DecisionPluginCatalogEntry{Type: DecisionPluginResponseJailbreak, DisplayName: "Response Jailbreak", Description: "Screen generated responses for jailbreak-like output."}, PluginOptions[ResponseJailbreakPluginConfig]{
			Strict: true,
			Validate: func(at PluginAt, p *ResponseJailbreakPluginConfig) error {
				return validateResponseJailbreakPlugin(at.Decision, at.Index, at.Type, p)
			},
		}),
		NewDecisionPluginType(DecisionPluginCatalogEntry{Type: DecisionPluginContextCompression, DisplayName: "Context Compression", Description: "Compress selected context before provider dispatch."}, PluginOptions[ContextCompressionPluginConfig]{
			Strict: true,
			Validate: func(at PluginAt, p *ContextCompressionPluginConfig) error {
				return validateContextCompressionPlugin(at.Decision, at.Index, at.Type, p)
			},
		}),
		NewDecisionPluginType(DecisionPluginCatalogEntry{Type: DecisionPluginPromptCache, DisplayName: "Prompt Cache", Description: "Add bounded Anthropic prompt-cache markers after route selection."}, PluginOptions[PromptCachePluginConfig]{
			Strict: true,
			Validate: func(at PluginAt, p *PromptCachePluginConfig) error {
				return validatePromptCachePlugin(at.Decision, at.Index, at.Type, p)
			},
		}),
		NewDecisionPluginType(DecisionPluginCatalogEntry{Type: DecisionPluginShadowDispatch, DisplayName: "Shadow Dispatch", Description: "Send a bounded asynchronous copy to a secondary model."}, PluginOptions[ShadowDispatchPluginConfig]{
			Strict: true,
			Validate: func(at PluginAt, p *ShadowDispatchPluginConfig) error {
				return validateShadowDispatchPlugin(at.Decision, at.Index, at.Type, p)
			},
		}),
		NewDecisionPluginType(DecisionPluginCatalogEntry{Type: DecisionPluginMasking, DisplayName: "Masking", Description: "Replace detected PII in the provider-bound request with placeholders."}, PluginOptions[MaskingPluginConfig]{
			Strict: true,
			Validate: func(at PluginAt, p *MaskingPluginConfig) error {
				return validateMaskingPlugin(at.Decision, at.Index, at.Type, p)
			},
		}),
	}
}
