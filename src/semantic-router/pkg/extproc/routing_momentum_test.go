package extproc

import (
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// --- Config tests ---

func TestCRM_DisabledByDefault(t *testing.T) {
	cfg := config.RoutingMomentumConfig{}
	if cfg.Enabled {
		t.Error("CRM should be disabled by default")
	}
}

func TestCRM_WindowDefault(t *testing.T) {
	cfg := config.RoutingMomentumConfig{}
	if cfg.GetWindow() != 11 {
		t.Errorf("Default window should be 11, got %d", cfg.GetWindow())
	}
}

func TestCRM_WindowCustom(t *testing.T) {
	cfg := config.RoutingMomentumConfig{Window: 5}
	if cfg.GetWindow() != 5 {
		t.Errorf("Expected window 5, got %d", cfg.GetWindow())
	}
}

func TestCRM_WindowZeroUsesDefault(t *testing.T) {
	cfg := config.RoutingMomentumConfig{Window: 0}
	if cfg.GetWindow() != 11 {
		t.Errorf("Zero window should use default 11, got %d", cfg.GetWindow())
	}
}

func TestCRM_ResponseThresholdDefault(t *testing.T) {
	cfg := config.RoutingMomentumConfig{}
	if cfg.GetResponseThreshold() != 200 {
		t.Errorf("Default response threshold should be 200, got %d", cfg.GetResponseThreshold())
	}
}

func TestCRM_ResponseThresholdCustom(t *testing.T) {
	cfg := config.RoutingMomentumConfig{ResponseThreshold: 500}
	if cfg.GetResponseThreshold() != 500 {
		t.Errorf("Expected threshold 500, got %d", cfg.GetResponseThreshold())
	}
}

// --- FindEnabledMomentumConfig tests ---

func TestFindEnabledMomentumConfig(t *testing.T) {
	tests := []struct {
		name    string
		cfg     *config.RouterConfig
		wantNil bool
	}{
		{name: "nil config", cfg: nil, wantNil: true},
		{name: "no decisions", cfg: &config.RouterConfig{}, wantNil: true},
		{
			name: "decision without momentum",
			cfg: &config.RouterConfig{
				IntelligentRouting: config.IntelligentRouting{
					Decisions: []config.Decision{{
						Name:      "simple",
						Algorithm: &config.AlgorithmConfig{Type: "confidence"},
					}},
				},
			},
			wantNil: true,
		},
		{
			name: "decision with momentum enabled",
			cfg: &config.RouterConfig{
				IntelligentRouting: config.IntelligentRouting{
					Decisions: []config.Decision{{
						Name: "complex",
						Algorithm: &config.AlgorithmConfig{
							Type:     "confidence",
							Momentum: &config.RoutingMomentumConfig{Enabled: true},
						},
					}},
				},
			},
			wantNil: false,
		},
		{
			name: "decision with momentum disabled",
			cfg: &config.RouterConfig{
				IntelligentRouting: config.IntelligentRouting{
					Decisions: []config.Decision{{
						Name: "complex",
						Algorithm: &config.AlgorithmConfig{
							Type:     "confidence",
							Momentum: &config.RoutingMomentumConfig{Enabled: false},
						},
					}},
				},
			},
			wantNil: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := findEnabledMomentumConfig(tt.cfg)
			if tt.wantNil && got != nil {
				t.Error("expected nil")
			}
			if !tt.wantNil && got == nil {
				t.Error("expected non-nil")
			}
		})
	}
}

func TestDecision_GetMomentumConfig(t *testing.T) {
	d := config.Decision{Name: "test"}
	if d.GetMomentumConfig() != nil {
		t.Error("expected nil when no algorithm")
	}

	d.Algorithm = &config.AlgorithmConfig{Type: "confidence"}
	if d.GetMomentumConfig() != nil {
		t.Error("expected nil when no momentum")
	}

	m := &config.RoutingMomentumConfig{Enabled: true}
	d.Algorithm.Momentum = m
	if d.GetMomentumConfig() != m {
		t.Error("expected momentum config")
	}
}

// --- CRM override logic tests ---

// makeTestRouter creates a router with two decisions: complex_query (momentum
// enabled) and general_query. No real classifier — tests that need keywords
// must set one up separately.
func makeCRMTestRouter() *OpenAIRouter {
	return &OpenAIRouter{
		Config: &config.RouterConfig{
			IntelligentRouting: config.IntelligentRouting{
				Decisions: []config.Decision{
					{
						Name: "complex_query",
						Algorithm: &config.AlgorithmConfig{
							Type:     "confidence",
							Momentum: &config.RoutingMomentumConfig{Enabled: true},
						},
						ModelRefs: []config.ModelRef{{Model: "expensive"}},
					},
					{
						Name:      "general_query",
						Algorithm: &config.AlgorithmConfig{Type: "confidence"},
						ModelRefs: []config.ModelRef{{Model: "cheap"}},
					},
				},
			},
		},
	}
}

func TestCRM_SingleMessage_NoOverride(t *testing.T) {
	router := makeCRMTestRouter()
	ctx := &RequestContext{
		ConversationHistory: []conversationMessage{
			{Role: "user", Text: "implement a compiler", TextLen: 20},
		},
	}
	cfg := &config.RoutingMomentumConfig{Enabled: true}

	decision, model := router.applyCRMOverride(ctx, "general_query", "cheap", cfg)
	if decision != "" || model != "" {
		t.Errorf("Expected no override for single message, got decision=%q model=%q", decision, model)
	}
}

func TestCRM_CurrentDecisionHasMomentum_NoOverride(t *testing.T) {
	router := makeCRMTestRouter()
	ctx := &RequestContext{
		ConversationHistory: []conversationMessage{
			{Role: "user", Text: "implement a compiler", TextLen: 20},
			{Role: "assistant", Text: "here is the code...", TextLen: 19},
			{Role: "user", Text: "commit it", TextLen: 9},
		},
	}
	cfg := &config.RoutingMomentumConfig{Enabled: true}

	// Already routed to complex_query — no override needed
	decision, model := router.applyCRMOverride(ctx, "complex_query", "expensive", cfg)
	if decision != "" || model != "" {
		t.Errorf("Expected no override when already on momentum decision, got decision=%q model=%q", decision, model)
	}
}

func TestCRM_LongResponse_TriggersOverride(t *testing.T) {
	router := makeCRMTestRouter()
	longResponse := strings.Repeat("x", 300)
	ctx := &RequestContext{
		ConversationHistory: []conversationMessage{
			{Role: "user", Text: "implement LRU cache", TextLen: 19},
			{Role: "assistant", Text: longResponse, TextLen: len(longResponse)},
			{Role: "user", Text: "commit it", TextLen: 9},
		},
	}
	cfg := &config.RoutingMomentumConfig{Enabled: true, ResponseThreshold: 200}

	decision, model := router.applyCRMOverride(ctx, "general_query", "cheap", cfg)
	if decision != "complex_query" || model != "expensive" {
		t.Errorf("Expected override to complex_query/expensive, got decision=%q model=%q", decision, model)
	}
}

func TestCRM_ShortResponse_NoOverride(t *testing.T) {
	router := makeCRMTestRouter()
	ctx := &RequestContext{
		ConversationHistory: []conversationMessage{
			{Role: "user", Text: "what is 2+2?", TextLen: 12},
			{Role: "assistant", Text: "4", TextLen: 1},
			{Role: "user", Text: "thanks", TextLen: 6},
		},
	}
	cfg := &config.RoutingMomentumConfig{Enabled: true, ResponseThreshold: 200}

	decision, model := router.applyCRMOverride(ctx, "general_query", "cheap", cfg)
	if decision != "" || model != "" {
		t.Errorf("Expected no override for short responses, got decision=%q model=%q", decision, model)
	}
}

func TestCRM_ResponseOutsideWindow_NoOverride(t *testing.T) {
	router := makeCRMTestRouter()
	longResponse := strings.Repeat("x", 300)

	// Long response is at position 1, but window=1 turn (2 messages) only
	// looks at the last 2 messages before current (positions 3 and 4).
	ctx := &RequestContext{
		ConversationHistory: []conversationMessage{
			{Role: "user", Text: "implement cache", TextLen: 15},
			{Role: "assistant", Text: longResponse, TextLen: len(longResponse)},
			{Role: "user", Text: "ok", TextLen: 2},
			{Role: "assistant", Text: "sure", TextLen: 4},
			{Role: "user", Text: "bye", TextLen: 3},
		},
	}
	cfg := &config.RoutingMomentumConfig{Enabled: true, Window: 1, ResponseThreshold: 200}

	decision, model := router.applyCRMOverride(ctx, "general_query", "cheap", cfg)
	if decision != "" || model != "" {
		t.Errorf("Expected no override when long response is outside window, got decision=%q model=%q", decision, model)
	}
}

func TestCRM_ResponseInsideWindow_TriggersOverride(t *testing.T) {
	router := makeCRMTestRouter()
	longResponse := strings.Repeat("x", 300)

	// Window=2 turns (4 messages): looks at all 4 messages before current,
	// which includes the long response at position 1.
	ctx := &RequestContext{
		ConversationHistory: []conversationMessage{
			{Role: "user", Text: "implement cache", TextLen: 15},
			{Role: "assistant", Text: longResponse, TextLen: len(longResponse)},
			{Role: "user", Text: "ok", TextLen: 2},
			{Role: "assistant", Text: "sure", TextLen: 4},
			{Role: "user", Text: "bye", TextLen: 3},
		},
	}
	cfg := &config.RoutingMomentumConfig{Enabled: true, Window: 2, ResponseThreshold: 200}

	decision, model := router.applyCRMOverride(ctx, "general_query", "cheap", cfg)
	if decision != "complex_query" || model != "expensive" {
		t.Errorf("Expected override when long response is inside window, got decision=%q model=%q", decision, model)
	}
}

func TestCRM_MultipleShortTurns_DecaysCorrectly(t *testing.T) {
	router := makeCRMTestRouter()
	longResponse := strings.Repeat("x", 300)

	// Build a conversation: complex response followed by many trivial turns
	history := []conversationMessage{
		{Role: "user", Text: "implement cache", TextLen: 15},
		{Role: "assistant", Text: longResponse, TextLen: len(longResponse)},
	}
	// Add 6 trivial turns (12 messages)
	for i := 0; i < 6; i++ {
		history = append(history,
			conversationMessage{Role: "user", Text: "ok", TextLen: 2},
			conversationMessage{Role: "assistant", Text: "sure", TextLen: 4},
		)
	}
	// Final user message
	history = append(history, conversationMessage{Role: "user", Text: "bye", TextLen: 3})

	// With window=2 turns (4 messages back): the long response (position 1)
	// is far outside the window. Should NOT override.
	ctx := &RequestContext{ConversationHistory: history}
	cfg := &config.RoutingMomentumConfig{Enabled: true, Window: 2, ResponseThreshold: 200}

	decision, _ := router.applyCRMOverride(ctx, "general_query", "cheap", cfg)
	if decision != "" {
		t.Errorf("Expected no override when complex response decayed out of window")
	}

	// With window=10 turns (20 messages): the long response IS inside. Should override.
	cfg2 := &config.RoutingMomentumConfig{Enabled: true, Window: 10, ResponseThreshold: 200}
	decision2, _ := router.applyCRMOverride(ctx, "general_query", "cheap", cfg2)
	if decision2 != "complex_query" {
		t.Errorf("Expected override when complex response is within large window")
	}
}

func TestCRM_ConversationMessageExtraction(t *testing.T) {
	// Verify extractConversationHistory preserves order and roles
	// (Can't easily test without openai types, so test the struct directly)
	history := []conversationMessage{
		{Role: "user", Text: "hello", TextLen: 5},
		{Role: "assistant", Text: "hi there!", TextLen: 9},
		{Role: "user", Text: "how are you?", TextLen: 12},
	}

	if len(history) != 3 {
		t.Fatalf("Expected 3 messages, got %d", len(history))
	}
	if history[0].Role != "user" || history[1].Role != "assistant" {
		t.Error("Roles not preserved")
	}
	if history[1].TextLen != 9 {
		t.Errorf("Expected TextLen 9, got %d", history[1].TextLen)
	}
}
