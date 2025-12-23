package config

// ToolVerifierConfig represents the configuration for two-stage tool call verification
// Stage 1: FunctionCallSentinel - prompt classification for injection detection
// Stage 2: ToolCallVerifier - token-level verification of generated tool calls
type ToolVerifierConfig struct {
	// Enable two-stage tool verification
	Enabled bool `yaml:"enabled"`

	// Stage 1: FunctionCallSentinel configuration
	// Classifies incoming prompts for injection risk
	Stage1 FunctionCallSentinelConfig `yaml:"stage1"`

	// Stage 2: ToolCallVerifier configuration
	// Verifies generated tool calls against user intent
	Stage2 ToolCallVerifierModelConfig `yaml:"stage2"`

	// Confidence-based routing configuration
	Routing RoutingConfig `yaml:"routing"`

	// Enforcement configuration - actions to take when issues detected
	Enforcement EnforcementConfig `yaml:"enforcement"`
}

// FunctionCallSentinelConfig represents Stage 1 configuration
type FunctionCallSentinelConfig struct {
	// Enable Stage 1 (prompt classification)
	Enabled bool `yaml:"enabled"`

	// Model ID or HuggingFace path (e.g., "rootfs/function-call-sentinel")
	ModelID string `yaml:"model_id"`

	// Threshold for blocking (high confidence INJECTION_RISK blocks immediately)
	BlockThreshold float32 `yaml:"block_threshold"`

	// Use CPU for inference (default: false, uses GPU if available)
	UseCPU bool `yaml:"use_cpu"`

	// Use vLLM REST API instead of local inference
	UseVLLM bool `yaml:"use_vllm,omitempty"`

	// vLLM endpoint for Stage 1 inference
	VLLMEndpoint ClassifierVLLMEndpoint `yaml:"vllm_endpoint,omitempty"`
}

// ToolCallVerifierModelConfig represents Stage 2 configuration
type ToolCallVerifierModelConfig struct {
	// Enable Stage 2 (tool-call verification)
	Enabled bool `yaml:"enabled"`

	// Model ID or HuggingFace path (e.g., "rootfs/tool-call-verifier")
	ModelID string `yaml:"model_id"`

	// Threshold for flagging unauthorized arguments
	UnauthorizedThreshold float32 `yaml:"unauthorized_threshold"`

	// Use CPU for inference (default: false, uses GPU if available)
	UseCPU bool `yaml:"use_cpu"`

	// Use vLLM REST API instead of local inference
	UseVLLM bool `yaml:"use_vllm,omitempty"`

	// vLLM endpoint for Stage 2 inference
	VLLMEndpoint ClassifierVLLMEndpoint `yaml:"vllm_endpoint,omitempty"`
}

// RoutingConfig defines confidence-based routing thresholds
type RoutingConfig struct {
	// Block threshold: if P(INJECTION) > BlockThreshold, block immediately
	// Default: 0.95
	BlockThreshold float32 `yaml:"block_threshold"`

	// Verify threshold: if P(INJECTION) > VerifyThreshold, run Stage 2
	// Default: 0.30
	VerifyThreshold float32 `yaml:"verify_threshold"`

	// High-risk tools that always trigger Stage 2 regardless of confidence
	// e.g., ["send_email", "delete_file", "transfer_money", "execute_command"]
	HighRiskTools []string `yaml:"high_risk_tools,omitempty"`

	// Skip Stage 2 for these low-risk tools even if confidence is borderline
	// e.g., ["get_weather", "search", "get_time"]
	LowRiskTools []string `yaml:"low_risk_tools,omitempty"`
}

// EnforcementConfig defines actions to take when tool verification detects issues
type EnforcementConfig struct {
	// InjectionAction specifies action when Stage 1 detects injection risk
	// "block" - return HTTP 403 (default for high confidence)
	// "header" - add warning headers, allow request through
	// "body" - prepend warning to response body
	// "none" - only log and metrics, no action
	InjectionAction string `yaml:"injection_action,omitempty"`

	// UnauthorizedToolCallAction specifies action when Stage 2 detects unauthorized tool calls
	// "block" - return HTTP 403 (default)
	// "header" - add warning headers, allow response through
	// "body" - prepend warning to response body
	// "none" - only log and metrics, no action
	UnauthorizedToolCallAction string `yaml:"unauthorized_tool_call_action,omitempty"`

	// IncludeDetails includes confidence scores and details in warnings
	// Only effective when action is "header" or "body"
	IncludeDetails bool `yaml:"include_details,omitempty"`
}

// GetInjectionAction returns the action for injection detection, defaulting to "block"
func (e *EnforcementConfig) GetInjectionAction() string {
	if e == nil || e.InjectionAction == "" {
		return "block"
	}
	switch e.InjectionAction {
	case "block", "header", "body", "none":
		return e.InjectionAction
	default:
		return "block"
	}
}

// GetUnauthorizedToolCallAction returns the action for unauthorized tool calls, defaulting to "block"
func (e *EnforcementConfig) GetUnauthorizedToolCallAction() string {
	if e == nil || e.UnauthorizedToolCallAction == "" {
		return "block"
	}
	switch e.UnauthorizedToolCallAction {
	case "block", "header", "body", "none":
		return e.UnauthorizedToolCallAction
	default:
		return "block"
	}
}

// GetDefaults returns default configuration values
func (c *ToolVerifierConfig) GetDefaults() ToolVerifierConfig {
	return ToolVerifierConfig{
		Enabled: false,
		Stage1: FunctionCallSentinelConfig{
			Enabled:        true,
			ModelID:        "rootfs/function-call-sentinel",
			BlockThreshold: 0.95,
			UseCPU:         false,
		},
		Stage2: ToolCallVerifierModelConfig{
			Enabled:               true,
			ModelID:               "rootfs/tool-call-verifier",
			UnauthorizedThreshold: 0.5,
			UseCPU:                false,
		},
		Routing: RoutingConfig{
			BlockThreshold:  0.95,
			VerifyThreshold: 0.30,
			HighRiskTools: []string{
				"send_email", "forward_email",
				"delete_file", "write_file",
				"transfer_money", "make_payment",
				"execute_command", "run_code",
				"create_user", "delete_user",
			},
			LowRiskTools: []string{
				"get_weather", "search", "get_time",
				"read_file", "list_files",
				"get_balance", "check_status",
			},
		},
	}
}

// IsHighRiskTool checks if a tool name is in the high-risk list
func (c *RoutingConfig) IsHighRiskTool(toolName string) bool {
	for _, t := range c.HighRiskTools {
		if t == toolName {
			return true
		}
	}
	return false
}

// IsLowRiskTool checks if a tool name is in the low-risk list
func (c *RoutingConfig) IsLowRiskTool(toolName string) bool {
	for _, t := range c.LowRiskTools {
		if t == toolName {
			return true
		}
	}
	return false
}

// ShouldRunStage2 determines if Stage 2 should run based on Stage 1 confidence and tool risk
func (c *RoutingConfig) ShouldRunStage2(injectionConfidence float32, toolNames []string) bool {
	// Check if any tool is high-risk
	for _, name := range toolNames {
		if c.IsHighRiskTool(name) {
			return true
		}
	}

	// Check if all tools are low-risk
	allLowRisk := true
	for _, name := range toolNames {
		if !c.IsLowRiskTool(name) {
			allLowRisk = false
			break
		}
	}

	// Skip Stage 2 for all low-risk tools with low injection confidence
	if allLowRisk && injectionConfidence < c.VerifyThreshold {
		return false
	}

	// Run Stage 2 if confidence is above verify threshold
	return injectionConfidence >= c.VerifyThreshold
}
