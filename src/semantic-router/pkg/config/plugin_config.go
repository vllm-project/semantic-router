package config

import (
	"encoding/json"
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// DecisionPlugin represents a plugin configuration for a decision.
type DecisionPlugin struct {
	// Type specifies the plugin type. Permitted values: "semantic-cache", "jailbreak",
	// "pii", "system_prompt", "header_mutation", "hallucination",
	// "response_jailbreak", "router_replay", "memory", "fast_response".
	Type string `yaml:"type" json:"type"`

	// Configuration is the raw configuration for this plugin.
	// When loaded from YAML, this will be a map[string]interface{}.
	// When loaded from Kubernetes CRD, this will be []byte (from runtime.RawExtension).
	Configuration interface{} `yaml:"configuration,omitempty" json:"configuration,omitempty"`
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

// GetPluginConfig returns the configuration for a specific plugin type.
func (d *Decision) GetPluginConfig(pluginType string) interface{} {
	for _, plugin := range d.Plugins {
		if plugin.Type == pluginType {
			return plugin.Configuration
		}
	}
	return nil
}

// UnmarshalPluginConfig converts a plugin configuration into the given target struct.
func UnmarshalPluginConfig(config interface{}, target interface{}) error {
	return unmarshalPluginConfig(config, target)
}

func unmarshalPluginConfig(config interface{}, target interface{}) error {
	if config == nil {
		return fmt.Errorf("plugin configuration is nil")
	}

	switch v := config.(type) {
	case map[string]interface{}:
		data, err := json.Marshal(v)
		if err != nil {
			return fmt.Errorf("failed to marshal config: %w", err)
		}
		return json.Unmarshal(data, target)
	case map[interface{}]interface{}:
		converted := convertMapToStringKeys(v)
		data, err := json.Marshal(converted)
		if err != nil {
			return fmt.Errorf("failed to marshal config: %w", err)
		}
		return json.Unmarshal(data, target)
	case []byte:
		return json.Unmarshal(v, target)
	default:
		return fmt.Errorf("unsupported configuration type: %T", config)
	}
}

func convertMapToStringKeys(m map[interface{}]interface{}) map[string]interface{} {
	result := make(map[string]interface{})
	for k, v := range m {
		key, ok := k.(string)
		if !ok {
			key = fmt.Sprintf("%v", k)
		}

		switch val := v.(type) {
		case map[interface{}]interface{}:
			result[key] = convertMapToStringKeys(val)
		case []interface{}:
			result[key] = convertSliceValues(val)
		default:
			result[key] = v
		}
	}
	return result
}

func convertSliceValues(s []interface{}) []interface{} {
	result := make([]interface{}, len(s))
	for i, v := range s {
		switch val := v.(type) {
		case map[interface{}]interface{}:
			result[i] = convertMapToStringKeys(val)
		case []interface{}:
			result[i] = convertSliceValues(val)
		default:
			result[i] = v
		}
	}
	return result
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

func decodeDecisionPlugin[T any](d *Decision, pluginType string, result *T) *T {
	config := d.GetPluginConfig(pluginType)
	if config == nil {
		return nil
	}

	if err := unmarshalPluginConfig(config, result); err != nil {
		logging.Errorf("Failed to unmarshal %s config: %v", pluginType, err)
		return nil
	}
	return result
}
