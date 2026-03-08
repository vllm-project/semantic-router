package authoring

import (
	"bytes"
	"fmt"
	"os"
	"strings"

	"gopkg.in/yaml.v3"

	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// Parse reads the canonical authoring slice from YAML bytes.
func Parse(data []byte) (*Config, error) {
	cfg := &Config{}
	decoder := yaml.NewDecoder(bytes.NewReader(data))
	decoder.KnownFields(true)
	if err := decoder.Decode(cfg); err != nil {
		return nil, fmt.Errorf("parse authoring config: %w", err)
	}
	if err := validateVersion(cfg.Version); err != nil {
		return nil, err
	}
	return cfg, nil
}

// ParseFile reads and parses an authoring config file.
func ParseFile(path string) (*Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read authoring config: %w", err)
	}
	return Parse(data)
}

// LoadRuntimeConfig loads the authoring config and compiles it into the runtime
// RouterConfig form.
func LoadRuntimeConfig(path string) (*routerconfig.RouterConfig, error) {
	cfg, err := ParseFile(path)
	if err != nil {
		return nil, err
	}
	return CompileRuntime(cfg)
}

// CompileRuntime compiles the authoring contract into the runtime RouterConfig
// form used by the router today.
func CompileRuntime(cfg *Config) (*routerconfig.RouterConfig, error) {
	if err := validateCompileConfig(cfg); err != nil {
		return nil, err
	}

	runtime := newRuntimeConfig(cfg)
	modelNames, err := compileProvidersIntoRuntime(runtime, cfg.Providers)
	if err != nil {
		return nil, err
	}
	if err := validateDecisionModelRefs(runtime.Decisions, modelNames); err != nil {
		return nil, err
	}

	return runtime, nil
}

func validateCompileConfig(cfg *Config) error {
	if cfg == nil {
		return fmt.Errorf("authoring config is nil")
	}
	return validateVersion(cfg.Version)
}

func newRuntimeConfig(cfg *Config) *routerconfig.RouterConfig {
	return &routerconfig.RouterConfig{
		MoMRegistry: routerconfig.ToLegacyRegistry(),
		APIServer: routerconfig.APIServer{
			Listeners: compileListeners(cfg.Listeners),
		},
		IntelligentRouting: routerconfig.IntelligentRouting{
			Signals: routerconfig.Signals{
				KeywordRules: compileKeywordRules(cfg.Signals.Keywords),
			},
			Decisions: compileDecisions(cfg.Decisions),
		},
		BackendModels: routerconfig.BackendModels{
			ModelConfig:      map[string]routerconfig.ModelParams{},
			DefaultModel:     cfg.Providers.DefaultModel,
			VLLMEndpoints:    []routerconfig.VLLMEndpoint{},
			ProviderProfiles: map[string]routerconfig.ProviderProfile{},
			ImageGenBackends: map[string]routerconfig.ImageGenBackendEntry{},
		},
	}
}

func compileProvidersIntoRuntime(runtime *routerconfig.RouterConfig, providers Providers) (map[string]struct{}, error) {
	modelNames := make(map[string]struct{}, len(providers.Models))
	seenEndpoints := make(map[string]struct{})

	for _, model := range providers.Models {
		if err := registerModelName(modelNames, model.Name); err != nil {
			return nil, err
		}

		params, endpoints, err := compileModel(model)
		if err != nil {
			return nil, err
		}
		if err := appendCompiledModel(runtime, model.Name, params, endpoints, seenEndpoints); err != nil {
			return nil, err
		}
	}

	if err := validateDefaultModel(providers.DefaultModel, modelNames); err != nil {
		return nil, err
	}

	runtime.ReasoningFamilies = compileReasoningFamilies(providers.ReasoningFamilies)
	runtime.DefaultReasoningEffort = providers.DefaultReasoningEffort

	return modelNames, nil
}

func registerModelName(modelNames map[string]struct{}, modelName string) error {
	if strings.TrimSpace(modelName) == "" {
		return fmt.Errorf("providers.models[].name is required")
	}
	if _, exists := modelNames[modelName]; exists {
		return fmt.Errorf("duplicate provider model %q", modelName)
	}
	modelNames[modelName] = struct{}{}
	return nil
}

func appendCompiledModel(
	runtime *routerconfig.RouterConfig,
	modelName string,
	params routerconfig.ModelParams,
	endpoints []routerconfig.VLLMEndpoint,
	seenEndpoints map[string]struct{},
) error {
	for _, endpoint := range endpoints {
		if _, exists := seenEndpoints[endpoint.Name]; exists {
			return fmt.Errorf("duplicate compiled endpoint %q", endpoint.Name)
		}
		seenEndpoints[endpoint.Name] = struct{}{}
	}

	runtime.ModelConfig[modelName] = params
	runtime.VLLMEndpoints = append(runtime.VLLMEndpoints, endpoints...)
	return nil
}

func validateDefaultModel(defaultModel string, modelNames map[string]struct{}) error {
	if defaultModel == "" {
		return nil
	}
	if _, exists := modelNames[defaultModel]; !exists {
		return fmt.Errorf("providers.default_model %q is not defined in providers.models", defaultModel)
	}
	return nil
}

func validateDecisionModelRefs(decisions []routerconfig.Decision, modelNames map[string]struct{}) error {
	for _, decision := range decisions {
		for _, modelRef := range decision.ModelRefs {
			if _, exists := modelNames[modelRef.Model]; !exists {
				return fmt.Errorf("decision %q references unknown model %q", decision.Name, modelRef.Model)
			}
		}
	}
	return nil
}

func validateVersion(version string) error {
	if strings.TrimSpace(version) == "" {
		return fmt.Errorf("authoring config version is required")
	}
	if version != CurrentVersion {
		return fmt.Errorf("unsupported authoring config version %q", version)
	}
	return nil
}

func compileListeners(listeners []Listener) []routerconfig.Listener {
	if len(listeners) == 0 {
		return nil
	}

	compiled := make([]routerconfig.Listener, 0, len(listeners))
	for _, listener := range listeners {
		compiled = append(compiled, routerconfig.Listener{
			Name:    listener.Name,
			Address: listener.Address,
			Port:    listener.Port,
			Timeout: listener.Timeout,
		})
	}
	return compiled
}

func compileKeywordRules(signals []KeywordSignal) []routerconfig.KeywordRule {
	if len(signals) == 0 {
		return nil
	}

	compiled := make([]routerconfig.KeywordRule, 0, len(signals))
	for _, signal := range signals {
		compiled = append(compiled, routerconfig.KeywordRule{
			Name:          signal.Name,
			Operator:      signal.Operator,
			Keywords:      append([]string(nil), signal.Keywords...),
			CaseSensitive: signal.CaseSensitive,
		})
	}
	return compiled
}

func compileDecisions(decisions []Decision) []routerconfig.Decision {
	if len(decisions) == 0 {
		return nil
	}

	compiled := make([]routerconfig.Decision, 0, len(decisions))
	for _, decision := range decisions {
		compiled = append(compiled, routerconfig.Decision{
			Name:        decision.Name,
			Description: decision.Description,
			Priority:    decision.Priority,
			Rules:       compileRules(decision.Rules),
			ModelRefs:   compileModelRefs(decision.ModelRefs),
		})
	}
	return compiled
}

func compileRules(rules Rules) routerconfig.RuleCombination {
	operator := rules.Operator
	if operator == "" {
		operator = "AND"
	}

	return routerconfig.RuleCombination{
		Operator:   operator,
		Conditions: compileConditions(rules.Conditions),
	}
}

func compileConditions(conditions []Condition) []routerconfig.RuleNode {
	if len(conditions) == 0 {
		return nil
	}

	compiled := make([]routerconfig.RuleNode, 0, len(conditions))
	for _, condition := range conditions {
		compiled = append(compiled, routerconfig.RuleNode{
			Type:       condition.Type,
			Name:       condition.Name,
			Operator:   condition.Operator,
			Conditions: compileConditions(condition.Conditions),
		})
	}
	return compiled
}

func compileModelRefs(modelRefs []ModelRef) []routerconfig.ModelRef {
	if len(modelRefs) == 0 {
		return nil
	}

	compiled := make([]routerconfig.ModelRef, 0, len(modelRefs))
	for _, modelRef := range modelRefs {
		useReasoning := modelRef.UseReasoning
		compiled = append(compiled, routerconfig.ModelRef{
			Model:    modelRef.Model,
			LoRAName: modelRef.LoRAName,
			ModelReasoningControl: routerconfig.ModelReasoningControl{
				UseReasoning:    &useReasoning,
				ReasoningEffort: modelRef.ReasoningEffort,
			},
		})
	}
	return compiled
}
