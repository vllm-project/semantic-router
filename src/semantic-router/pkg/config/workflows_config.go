package config

import (
	"fmt"
	"strings"
	"time"
)

const (
	DefaultFlowModelName = "vllm-sr/flow"

	WorkflowModeStatic  = "static"
	WorkflowModeDynamic = "dynamic"

	WorkflowOnErrorSkip = "skip"
	WorkflowOnErrorFail = "fail"

	WorkflowStateBackendMemory = "memory"
	WorkflowStateBackendFile   = "file"
	WorkflowStateBackendRedis  = "redis"

	DefaultWorkflowStateTTLSeconds = 1800
)

// FlowRuntimeConfig registers direct Router Flow model slugs. Workflow policy
// lives on routing decisions, not in global runtime config.
type FlowRuntimeConfig struct {
	ModelNames []string                   `yaml:"model_names,omitempty" json:"model_names,omitempty"`
	State      WorkflowStateRuntimeConfig `yaml:"state,omitempty" json:"state,omitempty"`
}

// WorkflowsAlgorithmConfig configures Router Flow execution for
// decision.algorithm.type=workflows. modelRefs on the decision are the worker
// boundary; planner.model is a control-plane model that generates or
// synthesizes the workflow plan.
type WorkflowsAlgorithmConfig struct {
	Mode                         string                `yaml:"mode,omitempty" json:"mode,omitempty"`
	Template                     string                `yaml:"template,omitempty" json:"template,omitempty"`
	Roles                        []WorkflowRoleConfig  `yaml:"roles,omitempty" json:"roles,omitempty"`
	Final                        WorkflowFinalConfig   `yaml:"final,omitempty" json:"final,omitempty"`
	Planner                      WorkflowPlannerConfig `yaml:"planner,omitempty" json:"planner,omitempty"`
	MaxSteps                     int                   `yaml:"max_steps,omitempty" json:"max_steps,omitempty"`
	MaxParallel                  int                   `yaml:"max_parallel,omitempty" json:"max_parallel,omitempty"`
	MaxCompletionTokens          int                   `yaml:"max_completion_tokens,omitempty" json:"max_completion_tokens,omitempty"`
	RoundTimeoutSeconds          int                   `yaml:"round_timeout_seconds,omitempty" json:"round_timeout_seconds,omitempty"`
	MinSuccessfulResponses       int                   `yaml:"min_successful_responses,omitempty" json:"min_successful_responses,omitempty"`
	Temperature                  *float64              `yaml:"temperature,omitempty" json:"temperature,omitempty"`
	IncludeIntermediateResponses *bool                 `yaml:"include_intermediate_responses,omitempty" json:"include_intermediate_responses,omitempty"`
	OnError                      string                `yaml:"on_error,omitempty" json:"on_error,omitempty"`
}

type WorkflowRoleConfig struct {
	Name       string   `yaml:"name,omitempty" json:"name,omitempty"`
	Models     []string `yaml:"models,omitempty" json:"models,omitempty"`
	Prompt     string   `yaml:"prompt,omitempty" json:"prompt,omitempty"`
	AccessList []string `yaml:"access_list,omitempty" json:"access_list,omitempty"`
}

type WorkflowFinalConfig struct {
	Model  string `yaml:"model,omitempty" json:"model,omitempty"`
	Prompt string `yaml:"prompt,omitempty" json:"prompt,omitempty"`
}

type WorkflowPlannerConfig struct {
	Model               string `yaml:"model,omitempty" json:"model,omitempty"`
	MaxCompletionTokens int    `yaml:"max_completion_tokens,omitempty" json:"max_completion_tokens,omitempty"`
}

type WorkflowStateRuntimeConfig struct {
	StoreBackend string                   `yaml:"store_backend,omitempty" json:"store_backend,omitempty"`
	TTLSeconds   int                      `yaml:"ttl_seconds,omitempty" json:"ttl_seconds,omitempty"`
	File         WorkflowStateFileConfig  `yaml:"file,omitempty" json:"file,omitempty"`
	Redis        WorkflowStateRedisConfig `yaml:"redis,omitempty" json:"redis,omitempty"`
}

type WorkflowStateFileConfig struct {
	Directory string `yaml:"directory,omitempty" json:"directory,omitempty"`
}

type WorkflowStateRedisConfig struct {
	Address       string `yaml:"address,omitempty" json:"address,omitempty"`
	DB            int    `yaml:"db,omitempty" json:"db,omitempty"`
	Password      string `yaml:"password,omitempty" json:"password,omitempty"`
	UseTLS        bool   `yaml:"use_tls,omitempty" json:"use_tls,omitempty"`
	TLSSkipVerify bool   `yaml:"tls_skip_verify,omitempty" json:"tls_skip_verify,omitempty"`
	MaxRetries    int    `yaml:"max_retries,omitempty" json:"max_retries,omitempty"`
	PoolSize      int    `yaml:"pool_size,omitempty" json:"pool_size,omitempty"`
	KeyPrefix     string `yaml:"key_prefix,omitempty" json:"key_prefix,omitempty"`
}

func (c WorkflowStateRuntimeConfig) WithDefaults() WorkflowStateRuntimeConfig {
	if strings.TrimSpace(c.StoreBackend) == "" {
		c.StoreBackend = WorkflowStateBackendFile
	}
	if c.TTLSeconds <= 0 {
		c.TTLSeconds = DefaultWorkflowStateTTLSeconds
	}
	return c
}

func (c WorkflowStateRuntimeConfig) TTL() time.Duration {
	return time.Duration(c.WithDefaults().TTLSeconds) * time.Second
}

func DefaultFlowModelNames() []string {
	return []string{DefaultFlowModelName}
}

func (c FlowRuntimeConfig) EffectiveModelNames() []string {
	if len(c.ModelNames) > 0 {
		return normalizeFlowModelNames(c.ModelNames)
	}
	return DefaultFlowModelNames()
}

func (c *RouterConfig) ExposedFlowModelNames() []string {
	if c == nil || !c.Looper.IsEnabled() {
		return nil
	}
	if !c.HasFlowDecision() {
		return nil
	}
	return c.Looper.Flow.EffectiveModelNames()
}

func normalizeFlowModelNames(names []string) []string {
	seen := make(map[string]bool, len(names))
	result := make([]string, 0, len(names))
	for _, name := range names {
		normalized := strings.TrimSpace(name)
		if normalized == "" || seen[normalized] {
			continue
		}
		seen[normalized] = true
		result = append(result, normalized)
	}
	return result
}

func (c *RouterConfig) IsFlowModelName(modelName string) bool {
	if c == nil {
		return false
	}
	normalized := strings.TrimSpace(modelName)
	if normalized == "" {
		return false
	}
	for _, candidate := range c.Looper.Flow.EffectiveModelNames() {
		if normalized == candidate {
			return true
		}
	}
	return false
}

func (c *RouterConfig) HasFlowDecision() bool {
	if c == nil {
		return false
	}
	for _, decision := range c.Decisions {
		if decision.Algorithm != nil && decision.Algorithm.Type == "workflows" {
			return true
		}
	}
	return false
}

func ValidateFlowRuntimeConfig(cfg FlowRuntimeConfig) error {
	for i, name := range cfg.ModelNames {
		if strings.TrimSpace(name) == "" {
			return fmt.Errorf("model_names[%d] cannot be empty", i)
		}
	}
	return ValidateWorkflowStateRuntimeConfig(cfg.State)
}

func ValidateWorkflowStateRuntimeConfig(cfg WorkflowStateRuntimeConfig) error {
	backend := strings.TrimSpace(cfg.StoreBackend)
	switch backend {
	case "", WorkflowStateBackendMemory, WorkflowStateBackendFile, WorkflowStateBackendRedis:
	default:
		return fmt.Errorf("state.store_backend must be one of %q, %q, or %q, got %q", WorkflowStateBackendMemory, WorkflowStateBackendFile, WorkflowStateBackendRedis, cfg.StoreBackend)
	}
	if cfg.TTLSeconds < 0 {
		return fmt.Errorf("state.ttl_seconds must be >= 1 when set")
	}
	if cfg.Redis.DB < 0 {
		return fmt.Errorf("state.redis.db must be >= 0")
	}
	if cfg.Redis.PoolSize < 0 {
		return fmt.Errorf("state.redis.pool_size must be >= 1 when set")
	}
	if cfg.Redis.MaxRetries < 0 {
		return fmt.Errorf("state.redis.max_retries must be >= 1 when set")
	}
	return nil
}

func ValidateWorkflowsAlgorithmConfig(cfg *WorkflowsAlgorithmConfig) error {
	if cfg == nil {
		return nil
	}
	if err := validateWorkflowModeAndPlan(cfg); err != nil {
		return err
	}
	if err := validateWorkflowPositiveControls(cfg); err != nil {
		return err
	}
	if cfg.Temperature != nil && *cfg.Temperature < 0 {
		return fmt.Errorf("temperature must be >= 0 when set")
	}
	return validateWorkflowOnError(cfg.OnError)
}

func validateWorkflowModeAndPlan(cfg *WorkflowsAlgorithmConfig) error {
	mode := strings.TrimSpace(cfg.Mode)
	switch mode {
	case "", WorkflowModeStatic, WorkflowModeDynamic:
	default:
		return fmt.Errorf("mode must be %q or %q, got %q", WorkflowModeStatic, WorkflowModeDynamic, cfg.Mode)
	}
	if mode == WorkflowModeDynamic && strings.TrimSpace(cfg.Planner.Model) == "" {
		return fmt.Errorf("planner.model is required when mode=dynamic")
	}
	return validateWorkflowStaticPlanConfig(mode, cfg)
}

func validateWorkflowPositiveControls(cfg *WorkflowsAlgorithmConfig) error {
	if cfg.MaxSteps < 0 {
		return fmt.Errorf("max_steps must be >= 1 when set")
	}
	if cfg.MaxParallel < 0 {
		return fmt.Errorf("max_parallel must be >= 1 when set")
	}
	if cfg.MaxCompletionTokens < 0 {
		return fmt.Errorf("max_completion_tokens must be >= 1 when set")
	}
	if cfg.RoundTimeoutSeconds < 0 {
		return fmt.Errorf("round_timeout_seconds must be >= 1 when set")
	}
	if cfg.MinSuccessfulResponses < 0 {
		return fmt.Errorf("min_successful_responses must be >= 1 when set")
	}
	if cfg.Planner.MaxCompletionTokens < 0 {
		return fmt.Errorf("planner.max_completion_tokens must be >= 1 when set")
	}
	return nil
}

func validateWorkflowStaticPlanConfig(mode string, cfg *WorkflowsAlgorithmConfig) error {
	if mode == "" {
		mode = WorkflowModeStatic
	}
	if mode == WorkflowModeDynamic {
		if len(cfg.Roles) > 0 {
			return fmt.Errorf("roles can only be set when mode=static")
		}
		return nil
	}
	if len(cfg.Roles) == 0 {
		return fmt.Errorf("roles is required when mode=static")
	}
	for i, role := range cfg.Roles {
		if err := validateWorkflowRoleConfig(i, role); err != nil {
			return err
		}
	}
	return nil
}

func validateWorkflowRoleConfig(index int, role WorkflowRoleConfig) error {
	context := fmt.Sprintf("roles[%d]", index)
	if strings.TrimSpace(role.Name) == "" {
		return fmt.Errorf("%s.name cannot be empty", context)
	}
	if len(role.Models) == 0 {
		return fmt.Errorf("%s.models must include at least one model", context)
	}
	seen := map[string]bool{}
	for i, model := range role.Models {
		normalized := strings.TrimSpace(model)
		if normalized == "" {
			return fmt.Errorf("%s.models[%d] cannot be empty", context, i)
		}
		if seen[normalized] {
			return fmt.Errorf("%s.models contains duplicate model %q", context, normalized)
		}
		seen[normalized] = true
	}
	return nil
}

func (c WorkflowFinalConfig) IsZero() bool {
	return strings.TrimSpace(c.Model) == "" && strings.TrimSpace(c.Prompt) == ""
}

func validateWorkflowOnError(onError string) error {
	if strings.TrimSpace(onError) == "" {
		return nil
	}
	switch onError {
	case WorkflowOnErrorSkip, WorkflowOnErrorFail:
		return nil
	default:
		return fmt.Errorf("on_error must be one of %q or %q, got %q", WorkflowOnErrorSkip, WorkflowOnErrorFail, onError)
	}
}
