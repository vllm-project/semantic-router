package classification

import (
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"time"

	candle "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

// ToolVerificationResult represents the result of the two-stage verification pipeline
type ToolVerificationResult struct {
	// Stage 1 results
	Stage1Ran           bool    `json:"stage1_ran"`
	InjectionRisk       bool    `json:"injection_risk"`
	InjectionConfidence float32 `json:"injection_confidence"`
	Stage1Blocked       bool    `json:"stage1_blocked"`
	Stage1LatencyMs     int64   `json:"stage1_latency_ms"`

	// Stage 2 results (only if ran)
	Stage2Ran             bool                   `json:"stage2_ran"`
	Stage2SkipReason      string                 `json:"stage2_skip_reason,omitempty"`
	UnauthorizedToolCalls []UnauthorizedToolCall `json:"unauthorized_tool_calls,omitempty"`
	Stage2Blocked         bool                   `json:"stage2_blocked"`
	Stage2LatencyMs       int64                  `json:"stage2_latency_ms,omitempty"`

	// Overall decision
	Blocked        bool   `json:"blocked"`
	BlockReason    string `json:"block_reason,omitempty"`
	TotalLatencyMs int64  `json:"total_latency_ms"`
}

// UnauthorizedToolCall represents an unauthorized tool call detected by Stage 2
type UnauthorizedToolCall struct {
	ToolName         string            `json:"tool_name"`
	UnauthorizedArgs []UnauthorizedArg `json:"unauthorized_args"`
	Confidence       float32           `json:"confidence"`
}

// UnauthorizedArg represents a specific unauthorized argument in a tool call
type UnauthorizedArg struct {
	ArgName    string  `json:"arg_name"`
	ArgValue   string  `json:"arg_value"`
	Reason     string  `json:"reason"`
	Confidence float32 `json:"confidence"`
}

// Stage1Result represents the output of FunctionCallSentinel (prompt classification)
type Stage1Result struct {
	Label      string  `json:"label"` // "SAFE" or "INJECTION_RISK"
	Confidence float32 `json:"confidence"`
	LatencyMs  int64   `json:"latency_ms"`
}

// Stage2Result represents the output of ToolCallVerifier (token classification)
type Stage2Result struct {
	ToolCalls       []ToolCallVerification `json:"tool_calls"`
	HasUnauthorized bool                   `json:"has_unauthorized"`
	LatencyMs       int64                  `json:"latency_ms"`
}

// ToolCallVerification represents verification of a single tool call
type ToolCallVerification struct {
	ToolName           string       `json:"tool_name"`
	IsAuthorized       bool         `json:"is_authorized"`
	Confidence         float32      `json:"confidence"`
	UnauthorizedTokens []TokenLabel `json:"unauthorized_tokens,omitempty"`
}

// TokenLabel represents a token with its classification label
type TokenLabel struct {
	Token      string  `json:"token"`
	Start      int     `json:"start"`
	End        int     `json:"end"`
	Label      string  `json:"label"` // "AUTHORIZED", "SUSPICIOUS", "UNAUTHORIZED"
	Confidence float32 `json:"confidence"`
}

// ToolCallInfo represents a tool call to verify
type ToolCallInfo struct {
	Name      string                 `json:"name"`
	Arguments map[string]interface{} `json:"arguments"`
}

// ToolVerifier handles the two-stage verification pipeline using Candle bindings
type ToolVerifier struct {
	config *config.ToolVerifierConfig

	// Stage 1 state
	stage1Initialized bool
	stage1Mu          sync.RWMutex

	// Stage 2 state
	stage2Initialized bool
	stage2Mu          sync.RWMutex
	stage2ConfigPath  string // Path to config.json with id2label mapping

	// Label mappings loaded from model config
	stage1Id2Label map[int]string
	stage2Id2Label map[int]string
}

// NewToolVerifier creates a new ToolVerifier with the given configuration
func NewToolVerifier(cfg *config.ToolVerifierConfig) (*ToolVerifier, error) {
	if cfg == nil || !cfg.Enabled {
		return nil, nil
	}

	tv := &ToolVerifier{
		config:         cfg,
		stage1Id2Label: make(map[int]string),
		stage2Id2Label: make(map[int]string),
	}

	return tv, nil
}

// Initialize initializes both Stage 1 and Stage 2 models via Candle bindings
func (tv *ToolVerifier) Initialize() error {
	var errs []string

	if tv.config.Stage1.Enabled {
		if err := tv.initializeStage1(); err != nil {
			errs = append(errs, fmt.Sprintf("Stage 1: %v", err))
		}
	}

	if tv.config.Stage2.Enabled {
		if err := tv.initializeStage2(); err != nil {
			errs = append(errs, fmt.Sprintf("Stage 2: %v", err))
		}
	}

	if len(errs) > 0 {
		return fmt.Errorf("initialization errors: %s", strings.Join(errs, "; "))
	}

	return nil
}

// initializeStage1 initializes the Stage 1 (FunctionCallSentinel) model
// Uses ModernBERT sequence classification via Candle
func (tv *ToolVerifier) initializeStage1() error {
	tv.stage1Mu.Lock()
	defer tv.stage1Mu.Unlock()

	if tv.stage1Initialized {
		return nil
	}

	modelPath := tv.config.Stage1.ModelID
	logging.Infof("Initializing Tool Verifier Stage 1 (FunctionCallSentinel) from: %s", modelPath)

	// Use the ModernBERT jailbreak classifier initialization
	// Our FunctionCallSentinel model has the same architecture
	err := candle.InitModernBertJailbreakClassifier(modelPath, tv.config.Stage1.UseCPU)
	if err != nil {
		return fmt.Errorf("failed to initialize Stage 1 model: %w", err)
	}

	// Set default label mapping for FunctionCallSentinel
	// Class 0 = SAFE, Class 1 = INJECTION_RISK
	tv.stage1Id2Label[0] = "SAFE"
	tv.stage1Id2Label[1] = "INJECTION_RISK"

	tv.stage1Initialized = true
	logging.Infof("Tool Verifier Stage 1 initialized successfully")

	return nil
}

// initializeStage2 initializes the Stage 2 (ToolCallVerifier) model
// Uses ModernBERT token classification via Candle
func (tv *ToolVerifier) initializeStage2() error {
	tv.stage2Mu.Lock()
	defer tv.stage2Mu.Unlock()

	if tv.stage2Initialized {
		return nil
	}

	modelPath := tv.config.Stage2.ModelID
	logging.Infof("Initializing Tool Verifier Stage 2 (ToolCallVerifier) from: %s", modelPath)

	// Initialize as a ModernBERT token classifier (similar to PII detection)
	// The Stage 2 model uses token classification for AUTHORIZED/SUSPICIOUS/UNAUTHORIZED
	err := candle.InitModernBertPIITokenClassifier(modelPath, tv.config.Stage2.UseCPU)
	if err != nil {
		return fmt.Errorf("failed to initialize Stage 2 model: %w", err)
	}

	// Store config path for token classification (contains id2label mapping)
	tv.stage2ConfigPath = modelPath + "/config.json"

	// Set default label mapping for ToolCallVerifier
	// Class 0 = AUTHORIZED, Class 1 = SUSPICIOUS, Class 2 = UNAUTHORIZED
	tv.stage2Id2Label[0] = "AUTHORIZED"
	tv.stage2Id2Label[1] = "SUSPICIOUS"
	tv.stage2Id2Label[2] = "UNAUTHORIZED"

	tv.stage2Initialized = true
	logging.Infof("Tool Verifier Stage 2 initialized successfully")

	return nil
}

// IsStage1Initialized returns whether Stage 1 is initialized
func (tv *ToolVerifier) IsStage1Initialized() bool {
	tv.stage1Mu.RLock()
	defer tv.stage1Mu.RUnlock()
	return tv.stage1Initialized
}

// IsStage2Initialized returns whether Stage 2 is initialized
func (tv *ToolVerifier) IsStage2Initialized() bool {
	tv.stage2Mu.RLock()
	defer tv.stage2Mu.RUnlock()
	return tv.stage2Initialized
}

// ClassifyPrompt runs Stage 1 classification on a user prompt
// Returns injection risk assessment using Candle bindings
func (tv *ToolVerifier) ClassifyPrompt(prompt string) (Stage1Result, error) {
	tv.stage1Mu.RLock()
	defer tv.stage1Mu.RUnlock()

	if !tv.stage1Initialized {
		return Stage1Result{}, fmt.Errorf("Stage 1 model not initialized")
	}

	if prompt == "" {
		return Stage1Result{
			Label:      "SAFE",
			Confidence: 1.0,
		}, nil
	}

	start := time.Now()

	// Call Candle binding for ModernBERT jailbreak classification
	candleResult, err := candle.ClassifyModernBertJailbreakText(prompt)
	if err != nil {
		return Stage1Result{}, fmt.Errorf("Stage 1 classification failed: %w", err)
	}

	latencyMs := time.Since(start).Milliseconds()
	metrics.RecordClassifierLatency("tool_verifier_stage1", time.Since(start).Seconds())

	// Map class index to label
	label, ok := tv.stage1Id2Label[candleResult.Class]
	if !ok {
		label = "UNKNOWN"
	}

	result := Stage1Result{
		Label:      label,
		Confidence: candleResult.Confidence,
		LatencyMs:  latencyMs,
	}

	logging.Infof("Stage 1 classification: label=%s, confidence=%.3f, latency=%dms",
		result.Label, result.Confidence, result.LatencyMs)

	return result, nil
}

// VerifyToolCall runs Stage 2 verification on a tool call
// Uses token classification to identify unauthorized arguments
func (tv *ToolVerifier) VerifyToolCall(userIntent string, toolCallJSON string) (Stage2Result, error) {
	tv.stage2Mu.RLock()
	defer tv.stage2Mu.RUnlock()

	if !tv.stage2Initialized {
		return Stage2Result{}, fmt.Errorf("Stage 2 model not initialized")
	}

	// Format input: [USER] intent [TOOL_CALL] json
	input := fmt.Sprintf("[USER] %s [TOOL_CALL] %s", userIntent, toolCallJSON)

	start := time.Now()

	// Call Candle binding for token classification
	// Pass the config.json path which contains the id2label mapping
	candleResult, err := candle.ClassifyModernBertPIITokens(input, tv.stage2ConfigPath)
	if err != nil {
		return Stage2Result{}, fmt.Errorf("Stage 2 verification failed: %w", err)
	}

	latencyMs := time.Since(start).Milliseconds()
	metrics.RecordClassifierLatency("tool_verifier_stage2", time.Since(start).Seconds())

	// Parse tool name from JSON
	var toolCall struct {
		Name string `json:"name"`
	}
	json.Unmarshal([]byte(toolCallJSON), &toolCall)

	// Build result from token classifications
	// candleResult.Entities contains tokens classified as non-O labels
	hasUnauthorized := false
	var unauthorizedTokens []TokenLabel
	var maxConfidence float32 = 0.5 // Default confidence

	for _, entity := range candleResult.Entities {
		if entity.EntityType == "UNAUTHORIZED" || entity.EntityType == "2" {
			hasUnauthorized = true
			unauthorizedTokens = append(unauthorizedTokens, TokenLabel{
				Token:      entity.Text,
				Start:      entity.Start,
				End:        entity.End,
				Label:      "UNAUTHORIZED",
				Confidence: entity.Confidence,
			})
			if entity.Confidence > maxConfidence {
				maxConfidence = entity.Confidence
			}
		}
	}

	result := Stage2Result{
		ToolCalls: []ToolCallVerification{
			{
				ToolName:           toolCall.Name,
				IsAuthorized:       !hasUnauthorized,
				Confidence:         maxConfidence,
				UnauthorizedTokens: unauthorizedTokens,
			},
		},
		HasUnauthorized: hasUnauthorized,
		LatencyMs:       latencyMs,
	}

	logging.Infof("Stage 2 verification: tool=%s, authorized=%v, unauthorized_tokens=%d, latency=%dms",
		toolCall.Name, !hasUnauthorized, len(unauthorizedTokens), result.LatencyMs)

	return result, nil
}

// VerifyRequest runs the full two-stage verification pipeline
func (tv *ToolVerifier) VerifyRequest(
	userPrompt string,
	toolCalls []ToolCallInfo,
) (*ToolVerificationResult, error) {
	startTime := time.Now()
	result := &ToolVerificationResult{}

	// Stage 1: Prompt Classification
	if tv.config.Stage1.Enabled && tv.IsStage1Initialized() {
		s1Result, err := tv.ClassifyPrompt(userPrompt)
		result.Stage1LatencyMs = s1Result.LatencyMs
		result.Stage1Ran = true

		if err != nil {
			logging.Errorf("Stage 1 classification failed: %v", err)
			metrics.RecordRequestError("tool_verifier", "stage1_error")
			// Continue without blocking on error
		} else {
			result.InjectionRisk = s1Result.Label == "INJECTION_RISK"
			result.InjectionConfidence = s1Result.Confidence

			// Check if should block immediately
			if result.InjectionRisk && result.InjectionConfidence >= tv.config.Routing.BlockThreshold {
				result.Stage1Blocked = true
				result.Blocked = true
				result.BlockReason = fmt.Sprintf("High confidence injection detected (%.2f%%)", result.InjectionConfidence*100)
				result.TotalLatencyMs = time.Since(startTime).Milliseconds()

				logging.Warnf("TOOL VERIFIER STAGE 1 BLOCKED: %s", result.BlockReason)
				metrics.RecordRequestError("tool_verifier", "stage1_blocked")
				return result, nil
			}
		}
	}

	// Determine if Stage 2 should run
	if !tv.config.Stage2.Enabled || !tv.IsStage2Initialized() || len(toolCalls) == 0 {
		if len(toolCalls) == 0 {
			result.Stage2SkipReason = "no_tool_calls"
		} else {
			result.Stage2SkipReason = "stage2_disabled"
		}
		result.TotalLatencyMs = time.Since(startTime).Milliseconds()
		return result, nil
	}

	// Check routing decision
	toolNames := make([]string, len(toolCalls))
	for i, tc := range toolCalls {
		toolNames[i] = tc.Name
	}

	if !tv.config.Routing.ShouldRunStage2(result.InjectionConfidence, toolNames) {
		result.Stage2SkipReason = "confidence_based_skip"
		result.TotalLatencyMs = time.Since(startTime).Milliseconds()
		logging.Infof("Stage 2 skipped: confidence=%.2f, tools=%v", result.InjectionConfidence, toolNames)
		return result, nil
	}

	// Stage 2: Tool Call Verification
	stage2Start := time.Now()
	result.Stage2Ran = true

	for _, tc := range toolCalls {
		toolCallJSON, err := json.Marshal(tc)
		if err != nil {
			logging.Errorf("Failed to marshal tool call: %v", err)
			continue
		}

		s2Result, err := tv.VerifyToolCall(userPrompt, string(toolCallJSON))
		if err != nil {
			logging.Errorf("Stage 2 verification failed for %s: %v", tc.Name, err)
			continue
		}

		// Check for unauthorized tool calls
		for _, tcv := range s2Result.ToolCalls {
			if !tcv.IsAuthorized {
				unauthorized := UnauthorizedToolCall{
					ToolName:   tcv.ToolName,
					Confidence: tcv.Confidence,
				}

				// Extract unauthorized arguments
				for _, token := range tcv.UnauthorizedTokens {
					unauthorized.UnauthorizedArgs = append(unauthorized.UnauthorizedArgs, UnauthorizedArg{
						ArgName:    extractArgName(token.Token),
						ArgValue:   token.Token,
						Reason:     "Token classified as UNAUTHORIZED",
						Confidence: token.Confidence,
					})
				}

				result.UnauthorizedToolCalls = append(result.UnauthorizedToolCalls, unauthorized)
			}
		}
	}

	result.Stage2LatencyMs = time.Since(stage2Start).Milliseconds()

	// Check if should block based on Stage 2
	if len(result.UnauthorizedToolCalls) > 0 {
		result.Stage2Blocked = true
		result.Blocked = true

		// Build block reason
		var reasons []string
		for _, utc := range result.UnauthorizedToolCalls {
			reasons = append(reasons, fmt.Sprintf("%s (confidence: %.2f%%)", utc.ToolName, utc.Confidence*100))
		}
		result.BlockReason = fmt.Sprintf("Unauthorized tool calls detected: %s", strings.Join(reasons, ", "))

		logging.Warnf("TOOL VERIFIER STAGE 2 BLOCKED: %s", result.BlockReason)
		metrics.RecordRequestError("tool_verifier", "stage2_blocked")
	}

	result.TotalLatencyMs = time.Since(startTime).Milliseconds()
	return result, nil
}

// Helper to extract argument name from token
func extractArgName(token string) string {
	if strings.Contains(token, ":") {
		parts := strings.SplitN(token, ":", 2)
		return strings.TrimSpace(parts[0])
	}
	return "unknown"
}
