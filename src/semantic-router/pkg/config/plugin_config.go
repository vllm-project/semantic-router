package config

import (
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// DecisionPlugin represents a plugin configuration for a decision.
// Type is the plugin identifier; the authoritative supported set is registered in
// routing_surface_catalog (and DSL/compiler surfaces), not duplicated here.
type DecisionPlugin struct {
	Type string `yaml:"type" json:"type"`

	// Configuration stores the plugin payload as normalized structured bytes.
	Configuration *StructuredPayload `yaml:"configuration,omitempty" json:"configuration,omitempty"`
}

// SemanticCachePluginConfig represents configuration for semantic-cache plugin.
type SemanticCachePluginConfig struct {
	Enabled             bool     `json:"enabled" yaml:"enabled"`
	SimilarityThreshold *float32 `json:"similarity_threshold,omitempty" yaml:"similarity_threshold,omitempty"`
	TTLSeconds          *int     `json:"ttl_seconds,omitempty" yaml:"ttl_seconds,omitempty"`
}

// MemoryPluginConfig is per-decision memory config (overrides global MemoryConfig).
type MemoryPluginConfig struct {
	Enabled             bool                    `json:"enabled" yaml:"enabled"`
	RetrievalLimit      *int                    `json:"retrieval_limit,omitempty" yaml:"retrieval_limit,omitempty"`
	SimilarityThreshold *float32                `json:"similarity_threshold,omitempty" yaml:"similarity_threshold,omitempty"`
	AutoStore           *bool                   `json:"auto_store,omitempty" yaml:"auto_store,omitempty"`
	HybridSearch        bool                    `json:"hybrid_search,omitempty" yaml:"hybrid_search,omitempty"`
	HybridMode          string                  `json:"hybrid_mode,omitempty" yaml:"hybrid_mode,omitempty"`
	Reflection          *MemoryReflectionConfig `json:"reflection,omitempty" yaml:"reflection,omitempty"`
}

// FastResponsePluginConfig represents configuration for fast_response plugin.
type FastResponsePluginConfig struct {
	Message string `json:"message" yaml:"message"`
}

// RequestParamsPluginConfig represents configuration for request_params plugin.
// This plugin validates and strips request body parameters per decision.
type RequestParamsPluginConfig struct {
	// BlockedParams is a list of parameters that should be blocked/stripped.
	BlockedParams []string `json:"blocked_params,omitempty" yaml:"blocked_params,omitempty"`
	// MaxTokensLimit sets the maximum allowed value for max_tokens.
	MaxTokensLimit *int `json:"max_tokens_limit,omitempty" yaml:"max_tokens_limit,omitempty"`
	// MaxN is the maximum allowed value for n (number of completions).
	MaxN *int `json:"max_n,omitempty" yaml:"max_n,omitempty"`
	// StripUnknown if true, removes fields not in the OpenAI spec.
	StripUnknown bool `json:"strip_unknown,omitempty" yaml:"strip_unknown,omitempty"`
}

// SystemPromptPluginConfig represents configuration for system_prompt plugin.
type SystemPromptPluginConfig struct {
	Enabled      *bool  `json:"enabled,omitempty" yaml:"enabled,omitempty"`
	SystemPrompt string `json:"system_prompt,omitempty" yaml:"system_prompt,omitempty"`
	Mode         string `json:"mode,omitempty" yaml:"mode,omitempty"`
}

// HeaderMutationPluginConfig represents configuration for header_mutation plugin.
type HeaderMutationPluginConfig struct {
	Add    []HeaderPair `json:"add,omitempty" yaml:"add,omitempty"`
	Update []HeaderPair `json:"update,omitempty" yaml:"update,omitempty"`
	Delete []string     `json:"delete,omitempty" yaml:"delete,omitempty"`
}

// HeaderPair represents a header name-value pair.
type HeaderPair struct {
	Name  string `json:"name" yaml:"name"`
	Value string `json:"value" yaml:"value"`
}

// ResponseJailbreakPluginConfig represents configuration for response-level jailbreak detection.
type ResponseJailbreakPluginConfig struct {
	Enabled   bool    `json:"enabled" yaml:"enabled"`
	Threshold float32 `json:"threshold,omitempty" yaml:"threshold,omitempty"`
	Action    string  `json:"action,omitempty" yaml:"action,omitempty"`
}

// HallucinationPluginConfig represents configuration for hallucination detection plugin.
type HallucinationPluginConfig struct {
	Enabled                     bool   `json:"enabled" yaml:"enabled"`
	UseNLI                      bool   `json:"use_nli,omitempty" yaml:"use_nli,omitempty"`
	HallucinationAction         string `json:"hallucination_action,omitempty" yaml:"hallucination_action,omitempty"`
	UnverifiedFactualAction     string `json:"unverified_factual_action,omitempty" yaml:"unverified_factual_action,omitempty"`
	IncludeHallucinationDetails bool   `json:"include_hallucination_details,omitempty" yaml:"include_hallucination_details,omitempty"`
}

// RouterReplayPluginConfig represents configuration for router_replay plugin.
type RouterReplayPluginConfig struct {
	Enabled             bool `json:"enabled" yaml:"enabled"`
	MaxRecords          int  `json:"max_records,omitempty" yaml:"max_records,omitempty"`
	CaptureRequestBody  bool `json:"capture_request_body,omitempty" yaml:"capture_request_body,omitempty"`
	CaptureResponseBody bool `json:"capture_response_body,omitempty" yaml:"capture_response_body,omitempty"`
	MaxBodyBytes        int  `json:"max_body_bytes,omitempty" yaml:"max_body_bytes,omitempty"`
}

// GetPlugin returns the plugin entry for a specific plugin type.
func (d *Decision) GetPlugin(pluginType string) *DecisionPlugin {
	for i := range d.Plugins {
		if d.Plugins[i].Type == pluginType {
			return &d.Plugins[i]
		}
	}
	return nil
}

// HasPlugin reports whether the decision includes a plugin of the given type.
func (d *Decision) HasPlugin(pluginType string) bool {
	return d.GetPlugin(pluginType) != nil
}

// UnmarshalPluginConfig converts a plugin payload into the given target struct.
func UnmarshalPluginConfig(config *StructuredPayload, target interface{}) error {
	if config == nil {
		return fmt.Errorf("plugin configuration is nil")
	}
	return config.DecodeInto(target)
}

// GetSemanticCacheConfig returns the semantic-cache plugin configuration.
func (d *Decision) GetSemanticCacheConfig() *SemanticCachePluginConfig {
	result := &SemanticCachePluginConfig{}
	return decodeDecisionPlugin(d, "semantic-cache", result)
}

// GetSystemPromptConfig returns the system_prompt plugin configuration.
func (d *Decision) GetSystemPromptConfig() *SystemPromptPluginConfig {
	result := &SystemPromptPluginConfig{}
	return decodeDecisionPlugin(d, "system_prompt", result)
}

// GetHeaderMutationConfig returns the header_mutation plugin configuration.
func (d *Decision) GetHeaderMutationConfig() *HeaderMutationPluginConfig {
	result := &HeaderMutationPluginConfig{}
	return decodeDecisionPlugin(d, "header_mutation", result)
}

// GetHallucinationConfig returns the hallucination plugin configuration.
func (d *Decision) GetHallucinationConfig() *HallucinationPluginConfig {
	result := &HallucinationPluginConfig{}
	return decodeDecisionPlugin(d, "hallucination", result)
}

// GetResponseJailbreakConfig returns the response_jailbreak plugin configuration.
func (d *Decision) GetResponseJailbreakConfig() *ResponseJailbreakPluginConfig {
	result := &ResponseJailbreakPluginConfig{}
	return decodeDecisionPlugin(d, "response_jailbreak", result)
}

// GetRouterReplayConfig returns the router_replay plugin configuration.
func (d *Decision) GetRouterReplayConfig() *RouterReplayPluginConfig {
	result := &RouterReplayPluginConfig{}
	return decodeDecisionPlugin(d, "router_replay", result)
}

// GetMemoryConfig returns the memory plugin config, or nil to use global config.
func (d *Decision) GetMemoryConfig() *MemoryPluginConfig {
	result := &MemoryPluginConfig{}
	return decodeDecisionPlugin(d, "memory", result)
}

// GetFastResponseConfig returns the fast_response plugin configuration.
func (d *Decision) GetFastResponseConfig() *FastResponsePluginConfig {
	result := &FastResponsePluginConfig{}
	return decodeDecisionPlugin(d, "fast_response", result)
}

// GetRequestParamsConfig returns the request_params plugin configuration.
func (d *Decision) GetRequestParamsConfig() *RequestParamsPluginConfig {
	result := &RequestParamsPluginConfig{}
	return decodeDecisionPlugin(d, "request_params", result)
}

// IsRAGEnabledForDecision returns whether RAG is enabled for a specific decision.
// Checks the per-decision RAG plugin config (enabled: true).
func (c *RouterConfig) IsRAGEnabledForDecision(decisionName string) bool {
	decision := c.GetDecisionByName(decisionName)
	if decision == nil {
		return false
	}
	ragConfig := decision.GetRAGConfig()
	return ragConfig != nil && ragConfig.Enabled
}

// IsMemoryEnabledForDecision returns whether memory is enabled for a specific decision.
// A decision has memory enabled if:
//   - It has a per-decision memory plugin with enabled: true, OR
//   - Global memory is enabled and the decision does NOT have a per-decision
//     memory plugin that explicitly disables it.
func (c *RouterConfig) IsMemoryEnabledForDecision(decisionName string) bool {
	decision := c.GetDecisionByName(decisionName)
	if decision == nil {
		return c.Memory.Enabled
	}
	memConfig := decision.GetMemoryConfig()
	if memConfig != nil {
		return memConfig.Enabled
	}
	return c.Memory.Enabled
}

// HasPersonalizationPlugins returns true if the decision has RAG or memory
// enabled, meaning cached responses would miss personalized context.
func (c *RouterConfig) HasPersonalizationPlugins(decisionName string) bool {
	return c.IsRAGEnabledForDecision(decisionName) || c.IsMemoryEnabledForDecision(decisionName)
}

func decodeDecisionPlugin[T any](d *Decision, pluginType string, result *T) *T {
	plugin := d.GetPlugin(pluginType)
	if plugin == nil || plugin.Configuration == nil {
		return nil
	}

	if err := UnmarshalPluginConfig(plugin.Configuration, result); err != nil {
		logging.Errorf("Failed to unmarshal %s config: %v", pluginType, err)
		return nil
	}
	return result
}
