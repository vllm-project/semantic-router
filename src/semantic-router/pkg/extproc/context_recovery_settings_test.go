package extproc

import (
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/historyreset"
)

func recoveryDecision(t *testing.T, compression map[string]interface{}, reset map[string]interface{}) *config.Decision {
	t.Helper()
	decision := &config.Decision{Name: "context"}
	if compression != nil {
		decision.Plugins = append(decision.Plugins, config.DecisionPlugin{
			Type:          config.DecisionPluginContextCompression,
			Configuration: config.MustStructuredPayload(compression),
		})
	}
	if reset != nil {
		decision.Plugins = append(decision.Plugins, config.DecisionPlugin{
			Type:          config.DecisionPluginHistoryReset,
			Configuration: config.MustStructuredPayload(reset),
		})
	}
	return decision
}

func enabledResetRecovery(recovery map[string]interface{}) map[string]interface{} {
	return map[string]interface{}{
		"enabled": true,
		"trigger": map[string]interface{}{
			"signal":         "topic_boundary",
			"min_confidence": 0.9,
		},
		"recovery": recovery,
	}
}

func TestContextRecoverySettingsResolveASingleConfiguredAction(t *testing.T) {
	decision := recoveryDecision(t, nil, enabledResetRecovery(map[string]interface{}{
		"enabled":     true,
		"store":       "redis",
		"ttl_seconds": 600,
	}))
	settings, err := resolveContextRecoverySettings(decision)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if settings == nil || settings.Store != "redis" || settings.TTLSeconds != 600 {
		t.Fatalf("unexpected settings %+v", settings)
	}
}

func TestContextRecoverySettingsAreAbsentWhenNoActionNeedsThem(t *testing.T) {
	decision := recoveryDecision(t,
		map[string]interface{}{"enabled": true},
		map[string]interface{}{"enabled": false},
	)
	settings, err := resolveContextRecoverySettings(decision)
	if err != nil || settings != nil {
		t.Fatalf("unexpected settings %+v (err=%v)", settings, err)
	}
	if settings, err = resolveContextRecoverySettings(nil); err != nil || settings != nil {
		t.Fatalf("a missing decision must resolve to no settings, got %+v (err=%v)", settings, err)
	}
}

// Two actions share one store, so every bound resolves to the stricter value
// and neither plugin can widen the other's budget.
func TestContextRecoverySettingsMergeToTheStricterBound(t *testing.T) {
	decision := recoveryDecision(t,
		map[string]interface{}{
			"enabled": true,
			"recovery": map[string]interface{}{
				"enabled":               true,
				"store":                 "redis",
				"ttl_seconds":           900,
				"max_bytes_per_request": 4096,
				"max_retrievals":        8,
			},
		},
		enabledResetRecovery(map[string]interface{}{
			"enabled":               true,
			"store":                 "redis",
			"ttl_seconds":           300,
			"max_bytes_per_request": 8192,
			"max_total_bytes":       16384,
		}),
	)
	settings, err := resolveContextRecoverySettings(decision)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if settings.TTLSeconds != 300 {
		t.Fatalf("ttl_seconds = %d, want the stricter 300", settings.TTLSeconds)
	}
	if settings.MaxBytesPerRequest != 4096 {
		t.Fatalf("max_bytes_per_request = %d, want the stricter 4096", settings.MaxBytesPerRequest)
	}
	if settings.MaxTotalBytes != 16384 {
		t.Fatalf("max_total_bytes = %d, want the only configured bound", settings.MaxTotalBytes)
	}
	if settings.MaxRetrievals != 8 {
		t.Fatalf("max_retrievals = %d, want the only configured bound", settings.MaxRetrievals)
	}
}

func TestContextRecoverySettingsRejectDisagreeingStores(t *testing.T) {
	decision := recoveryDecision(t,
		map[string]interface{}{
			"enabled":  true,
			"recovery": map[string]interface{}{"enabled": true, "store": "redis"},
		},
		enabledResetRecovery(map[string]interface{}{"enabled": true, "store": "valkey"}),
	)
	settings, err := resolveContextRecoverySettings(decision)
	if err == nil {
		t.Fatalf("disagreeing stores were accepted: %+v", settings)
	}
	if !strings.Contains(err.Error(), "disagree") {
		t.Fatalf("unexpected error %v", err)
	}
}

// Retrieval no longer depends on the compression plugin: a key issued by any
// context action stays retrievable.
func TestActiveContextRecoveryServesResetOnlyRequests(t *testing.T) {
	ctx := &RequestContext{
		VSRSelectedDecision: recoveryDecision(t, nil, enabledResetRecovery(map[string]interface{}{
			"enabled": true,
			"store":   "redis",
		})),
		ContextCompressionRecoveryKeys: []string{"issued-key"},
	}
	if recovery := activeContextRecovery(ctx); recovery == nil || recovery.Store != "redis" {
		t.Fatalf("reset-only recovery was not active: %+v", recovery)
	}

	ctx.ExpectStreamingResponse = true
	if activeContextRecovery(ctx) != nil {
		t.Fatal("streaming responses cannot run the retrieval follow-up")
	}

	ctx.ExpectStreamingResponse = false
	ctx.ContextCompressionRecoveryKeys = nil
	if activeContextRecovery(ctx) != nil {
		t.Fatal("a request that issued no key has nothing to retrieve")
	}
}

func TestRemovedHistoryIsRetrievableThroughTheReservedTool(t *testing.T) {
	router := recoverableResetRouter()
	request := resetConversation()
	ctx := recoverableResetContext(t, recoverableResetDecision(t, nil), request)

	router.prepareContextHistorySteps(ctx, request)
	if err := router.applyContextTransformationPlan(ctx, request); err != nil {
		t.Fatalf("expected the plan to succeed: %v", err)
	}
	key := ctx.ContextCompressionRecoveryKeys[0]

	recovery := activeContextRecovery(ctx)
	if recovery == nil {
		t.Fatal("recovery must be active after a recoverable removal")
	}
	messages, err := router.loadContextRecoveryToolMessages(
		t.Context(),
		ctx,
		recovery,
		[]contextRecoveryCall{{ID: "call-1", Key: key}},
	)
	if err != nil {
		t.Fatalf("retrieval failed: %v", err)
	}
	if len(messages) != 1 {
		t.Fatalf("expected one tool message, got %d", len(messages))
	}
	envelope, err := historyreset.DecodeEnvelope(messages[0].Content[0].ToolResult.Content[0].Text)
	if err != nil {
		t.Fatalf("retrieved payload is not a valid envelope: %v", err)
	}
	if envelope.Messages != 2 || envelope.Removed[0].Message.Content[0].Text != "old question" {
		t.Fatalf("unexpected retrieved envelope %+v", envelope)
	}

	// A key this request never issued cannot be retrieved.
	if _, err = router.loadContextRecoveryToolMessages(
		t.Context(),
		ctx,
		recovery,
		[]contextRecoveryCall{{ID: "call-1", Key: "forged-key"}},
	); err == nil {
		t.Fatal("a forged recovery key was accepted")
	}
}
