package extproc

import (
	"strings"
	"testing"
	"time"

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

const mib = 1 << 20

func compressionRecovery(recovery map[string]interface{}) map[string]interface{} {
	return map[string]interface{}{"enabled": true, "recovery": recovery}
}

// Each action's omitted bound is its own effective default, never "unbounded",
// so an omitted reset bound cannot be widened by a configured compression one.
func TestRecoveryContractNormalizesEachActionBeforeMerging(t *testing.T) {
	cases := []struct {
		name        string
		compression map[string]interface{}
		reset       map[string]interface{}
		want        contextRecoveryContract
	}{
		{
			name:        "compression only keeps its own defaults",
			compression: compressionRecovery(map[string]interface{}{"enabled": true, "store": "redis"}),
			want: contextRecoveryContract{
				Store: "redis", TTL: 15 * time.Minute, MaxBytesPerRequest: 10 * mib,
				MaxTotalBytes: 256 * mib, MaxRetrievals: 8,
			},
		},
		{
			name:  "reset only keeps its own defaults",
			reset: enabledResetRecovery(map[string]interface{}{"enabled": true, "store": "redis"}),
			want: contextRecoveryContract{
				Store: "redis", TTL: 15 * time.Minute, MaxBytesPerRequest: 1 * mib,
				MaxTotalBytes: 256 * mib, MaxRetrievals: 8,
			},
		},
		{
			name: "an omitted reset bound is its 1 MiB default, not unbounded",
			compression: compressionRecovery(map[string]interface{}{
				"enabled": true, "store": "redis", "max_bytes_per_request": 4 * mib,
			}),
			reset: enabledResetRecovery(map[string]interface{}{"enabled": true, "store": "redis"}),
			want: contextRecoveryContract{
				Store: "redis", TTL: 15 * time.Minute, MaxBytesPerRequest: 1 * mib,
				MaxTotalBytes: 256 * mib, MaxRetrievals: 8,
			},
		},
		{
			name: "every field resolves to the stricter effective value",
			compression: compressionRecovery(map[string]interface{}{
				"enabled": true, "store": "redis", "ttl_seconds": 900,
				"max_bytes_per_request": 4096, "max_retrievals": 3,
			}),
			reset: enabledResetRecovery(map[string]interface{}{
				"enabled": true, "store": "redis", "ttl_seconds": 300,
				"max_bytes_per_request": 8192, "max_total_bytes": 16384,
			}),
			want: contextRecoveryContract{
				Store: "redis", TTL: 300 * time.Second, MaxBytesPerRequest: 4096,
				MaxTotalBytes: 16384, MaxRetrievals: 3,
			},
		},
	}
	for _, test := range cases {
		t.Run(test.name, func(t *testing.T) {
			contract, err := resolveContextRecoveryContract(recoveryDecision(t, test.compression, test.reset))
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if contract == nil || *contract != test.want {
				t.Fatalf("contract = %+v, want %+v", contract, test.want)
			}
		})
	}
}

func TestRecoveryContractIsAbsentWhenNoActionNeedsIt(t *testing.T) {
	decision := recoveryDecision(t,
		map[string]interface{}{"enabled": true},
		map[string]interface{}{"enabled": false},
	)
	if contract, err := resolveContextRecoveryContract(decision); err != nil || contract != nil {
		t.Fatalf("unexpected contract %+v (err=%v)", contract, err)
	}
}

func TestRecoveryContractRejectsDisagreeingStores(t *testing.T) {
	decision := recoveryDecision(t,
		compressionRecovery(map[string]interface{}{"enabled": true, "store": "redis"}),
		enabledResetRecovery(map[string]interface{}{"enabled": true, "store": "valkey"}),
	)
	contract, err := resolveContextRecoveryContract(decision)
	if err == nil || !strings.Contains(err.Error(), "disagree") {
		t.Fatalf("disagreeing stores were accepted: %+v (err=%v)", contract, err)
	}
}

// Router construction compares effective values, so an omitted total and the
// same total written explicitly agree, and a genuinely different total does not.
func TestRecoveryAgreementComparesEffectiveTotals(t *testing.T) {
	omitted := *recoveryDecision(t, compressionRecovery(map[string]interface{}{"enabled": true, "store": "redis"}), nil)
	omitted.Name = "omitted"
	explicit := *recoveryDecision(t, compressionRecovery(map[string]interface{}{
		"enabled": true, "store": "redis", "max_total_bytes": 256 * mib,
	}), nil)
	explicit.Name = "explicit"
	cfg := &config.RouterConfig{}
	cfg.Decisions = []config.Decision{omitted, explicit}
	if err := verifyContextRecoveryAgreement(cfg); err != nil {
		t.Fatalf("an omitted default and the same explicit total disagreed: %v", err)
	}

	smaller := *recoveryDecision(t, compressionRecovery(map[string]interface{}{
		"enabled": true, "store": "redis", "max_total_bytes": 1024,
	}), nil)
	smaller.Name = "smaller"
	cfg.Decisions = append(cfg.Decisions, smaller)
	if err := verifyContextRecoveryAgreement(cfg); err == nil || !strings.Contains(err.Error(), "max_total_bytes") {
		t.Fatalf("different effective totals were accepted: %v", err)
	}
}

// Retrieval reads the contract the request was bound with, whichever context
// action issued the key, including its merged retrieval limit.
func TestActiveContextRecoveryServesResetOnlyRequests(t *testing.T) {
	ctx := &RequestContext{
		VSRSelectedDecision: recoveryDecision(t, nil, enabledResetRecovery(map[string]interface{}{
			"enabled":        true,
			"store":          "redis",
			"max_retrievals": 2,
		})),
		ContextCompressionRecoveryKeys: []string{"issued-key"},
	}
	recoverableResetRouter().bindContextPolicy(ctx)
	if recovery := activeContextRecovery(ctx); recovery == nil || recovery.Store != "redis" || recovery.MaxRetrievals != 2 {
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
	ctx := recoverableResetContext(t, router, recoverableResetDecision(t, nil), request)

	router.prepareContextHistorySteps(ctx, request)
	if err := router.applyContextTransformationPlan(ctx, request); err != nil {
		t.Fatalf("expected the plan to succeed: %v", err)
	}
	key := ctx.ContextCompressionRecoveryKeys[0]

	if activeContextRecovery(ctx) == nil {
		t.Fatal("recovery must be active after a recoverable removal")
	}
	messages, err := router.loadContextRecoveryToolMessages(
		t.Context(),
		ctx,
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
		[]contextRecoveryCall{{ID: "call-1", Key: "forged-key"}},
	); err == nil {
		t.Fatal("a forged recovery key was accepted")
	}
}
