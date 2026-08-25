package agentmanagement

import (
	"bytes"
	"encoding/json"
	"errors"
	"strings"
	"testing"
)

func TestScrubToolSecretsRemovesEveryExactNestedOccurrence(t *testing.T) {
	secret := []byte("credential-canary-42")
	raw := json.RawMessage(`{
  "credential-canary-42-key":"credential-canary-42",
  "start":"credential-canary-42 suffix",
  "end":"prefix credential-canary-42",
  "nested":{"items":["before-credential-canary-42-after",{"again":"credential-canary-42credential-canary-42"}]},
  "safe":"keep"
}`)
	redacted, err := ScrubToolSecrets(raw, secret)
	if err != nil {
		t.Fatalf("ScrubToolSecrets() error = %v", err)
	}
	if bytes.Contains(redacted, secret) {
		t.Fatalf("ScrubToolSecrets() retained exact credential: %s", redacted)
	}
	var value map[string]any
	if err := json.Unmarshal(redacted, &value); err != nil {
		t.Fatal(err)
	}
	if value["safe"] != "keep" || value["[redacted]-key"] != "[redacted]" ||
		value["start"] != "[redacted] suffix" || value["end"] != "prefix [redacted]" {
		t.Fatalf("exact credential redaction changed the wrong fields: %#v", value)
	}
	nested := value["nested"].(map[string]any)
	items := nested["items"].([]any)
	if items[0] != "before-[redacted]-after" ||
		items[1].(map[string]any)["again"] != "[redacted][redacted]" {
		t.Fatalf("nested exact credential occurrences were not all redacted: %#v", nested)
	}
}

func TestScrubToolSecretsHandlesOverlappingValuesLongestFirst(t *testing.T) {
	redacted, err := ScrubToolSecrets(
		json.RawMessage(`{"value":"token-value-long/token-value"}`),
		[]byte("token-value"), []byte("token-value-long"),
	)
	if err != nil {
		t.Fatalf("ScrubToolSecrets() error = %v", err)
	}
	if string(redacted) != `{"value":"[redacted]/[redacted]"}` {
		t.Fatalf("overlapping exact secret redaction = %s", redacted)
	}
}

func TestScrubToolSecretsNeverReintroducesSecretThroughItsMarker(t *testing.T) {
	for _, secret := range [][]byte{[]byte("redacted"), []byte("[redacted]"), []byte("credential removed")} {
		redacted, err := ScrubToolSecrets(
			json.RawMessage(`{"value":"prefix-`+string(secret)+`-suffix","password":"safe"}`),
			secret,
		)
		if err != nil {
			t.Fatalf("secret %q: ScrubToolSecrets() error = %v", secret, err)
		}
		if bytes.Contains(redacted, secret) {
			t.Fatalf("secret %q was reintroduced by a redaction marker: %s", secret, redacted)
		}
	}
}

func TestScrubToolSecretsFailsClosedForInvalidUTF8OrKeyCollision(t *testing.T) {
	if _, err := ScrubToolSecrets(
		json.RawMessage(`{"value":"safe"}`), []byte{0xff, 'a', 'b', 'c', 'd', 'e', 'f', 'g'},
	); !errors.Is(err, ErrInvalid) {
		t.Fatalf("invalid UTF-8 secret error = %v, want ErrInvalid", err)
	}
	if _, err := ScrubToolSecrets(json.RawMessage(`{"value":"safe"}`), []byte("short")); !errors.Is(err, ErrInvalid) {
		t.Fatalf("short secret error = %v, want ErrInvalid", err)
	}
	if _, err := ScrubToolSecrets(
		json.RawMessage(`{"secret-value":"first","[redacted]":"second"}`), []byte("secret-value"),
	); !errors.Is(err, ErrInvalid) {
		t.Fatalf("redacted key collision error = %v, want ErrInvalid", err)
	}
}

func TestNormalizeEventAppendRedactsCredentialShapedToolContent(t *testing.T) {
	payload, err := json.Marshal(ToolRequestEvent{
		InvocationID: "11111111-1111-4111-8111-111111111111",
		ToolName:     "router.recipe.get",
		Class:        ToolRead,
		Arguments: json.RawMessage(`{
  "recipeId":"balance",
  "authorization":"Bearer private-value",
  "nested":{"api_key":"sk-private","label":"keep"}
}`),
	})
	if err != nil {
		t.Fatal(err)
	}
	normalized, err := NormalizeEventAppend(EventAppend{
		Origin: "worker", Fence: int64Pointer(1), Type: EventToolRequest, Payload: payload,
	})
	if err != nil {
		t.Fatalf("NormalizeEventAppend() error = %v", err)
	}
	var event ToolRequestEvent
	if err := json.Unmarshal(normalized.Payload, &event); err != nil {
		t.Fatal(err)
	}
	var arguments map[string]any
	if err := json.Unmarshal(event.Arguments, &arguments); err != nil {
		t.Fatal(err)
	}
	if arguments["authorization"] != "[redacted]" {
		t.Fatalf("authorization = %#v, want redacted", arguments["authorization"])
	}
	nested, ok := arguments["nested"].(map[string]any)
	if !ok || nested["api_key"] != "[redacted]" || nested["label"] != "keep" {
		t.Fatalf("nested arguments = %#v", arguments["nested"])
	}
}

func TestNormalizeEventAppendRejectsOpenOrUnboundedPayloads(t *testing.T) {
	for name, request := range map[string]EventAppend{
		"unknown event field": {
			Origin: "worker", Fence: int64Pointer(1), Type: EventProgress,
			Payload: json.RawMessage(`{"phase":"probe","message":"running","secretExtension":true}`),
		},
		"oversized inline result": {
			Origin: "worker", Fence: int64Pointer(1), Type: EventToolResult,
			Payload: mustJSON(t, ToolResultEvent{
				InvocationID: "11111111-1111-4111-8111-111111111111",
				ToolName:     "router.recipe.get", Status: "completed",
				Result: json.RawMessage(`{"value":"` + strings.Repeat("x", maximumInlineToolResultBytes) + `"}`),
			}),
		},
	} {
		t.Run(name, func(t *testing.T) {
			if _, err := NormalizeEventAppend(request); !errors.Is(err, ErrInvalid) {
				t.Fatalf("NormalizeEventAppend() error = %v, want ErrInvalid", err)
			}
		})
	}
}

func TestNormalizeEventAppendAcceptsClosedModelStepSummary(t *testing.T) {
	ttft := int64(84)
	inputUncached := int64(90)
	inputCacheRead := int64(30)
	outputReasoning := int64(12)
	outputOther := int64(36)
	payload := mustJSON(t, ModelStepSummaryEvent{
		ModelStepID:         "11111111-1111-4111-8111-111111111111",
		RequestID:           "request-42",
		SelectedRecipe:      "balance",
		SelectedDecision:    "Complex",
		SelectedModel:       "remote/frontier",
		SelectedAlgorithm:   "static",
		ResponsePath:        "upstream",
		LatencyMilliseconds: 420,
		TTFTMilliseconds:    &ttft,
		Usage: &ModelStepUsage{
			InputTokens: 120, OutputTokens: 48, TotalTokens: 168,
			InputUncachedTokens: &inputUncached, InputCacheReadTokens: &inputCacheRead,
			OutputReasoningTokens: &outputReasoning, OutputOtherTokens: &outputOther,
		},
	})
	normalized, err := NormalizeEventAppend(EventAppend{
		Origin: "worker", Fence: int64Pointer(1), Type: EventModelStepSummary, Payload: payload,
	})
	if err != nil {
		t.Fatalf("NormalizeEventAppend() error = %v", err)
	}
	var summary ModelStepSummaryEvent
	if err := json.Unmarshal(normalized.Payload, &summary); err != nil {
		t.Fatal(err)
	}
	if summary.ModelStepID != "11111111-1111-4111-8111-111111111111" ||
		summary.RequestID != "request-42" || summary.Usage == nil || summary.Usage.TotalTokens != 168 {
		t.Fatalf("normalized model-step summary = %#v", summary)
	}
}

func TestNormalizeEventAppendRejectsUnsafeModelStepSummary(t *testing.T) {
	base := `{
  "modelStepId":"11111111-1111-4111-8111-111111111111",
  "requestId":"request-42",
  "responsePath":"upstream",
  "latencyMilliseconds":420
}`
	for name, payload := range map[string]json.RawMessage{
		"unknown provider field": json.RawMessage(`{
  "modelStepId":"11111111-1111-4111-8111-111111111111",
  "requestId":"request-42",
  "responsePath":"upstream",
  "latencyMilliseconds":420,
  "providerOpaque":{"secret":"must-not-persist"}
}`),
		"non-success response path": json.RawMessage(strings.Replace(base, `"upstream"`, `"error"`, 1)),
		"ttft exceeds latency":      json.RawMessage(strings.Replace(base, `"latencyMilliseconds":420`, `"latencyMilliseconds":420,"ttftMilliseconds":421`, 1)),
		"inconsistent usage total":  json.RawMessage(strings.Replace(base, `"latencyMilliseconds":420`, `"latencyMilliseconds":420,"usage":{"inputTokens":4,"outputTokens":5,"totalTokens":10}`, 1)),
	} {
		t.Run(name, func(t *testing.T) {
			_, err := NormalizeEventAppend(EventAppend{
				Origin: "worker", Fence: int64Pointer(1), Type: EventModelStepSummary, Payload: payload,
			})
			if !errors.Is(err, ErrInvalid) {
				t.Fatalf("NormalizeEventAppend() error = %v, want ErrInvalid", err)
			}
		})
	}
}

func TestNormalizeEventAppendRequiresExactlyOneTerminalOutcomeShape(t *testing.T) {
	for _, status := range []TurnStatus{TurnCompleted, TurnCancelled} {
		request := EventAppend{
			Origin: "control", Type: EventTerminal,
			Payload: mustJSON(t, TerminalEvent{Status: status}),
		}
		if _, err := NormalizeEventAppend(request); err != nil {
			t.Fatalf("status %s: %v", status, err)
		}
	}
	invalid := EventAppend{
		Origin: "control", Type: EventTerminal,
		Payload: mustJSON(t, TerminalEvent{Status: TurnWaitingApproval}),
	}
	if _, err := NormalizeEventAppend(invalid); !errors.Is(err, ErrInvalid) {
		t.Fatalf("waiting approval terminal error = %v, want ErrInvalid", err)
	}
}

func int64Pointer(value int64) *int64 { return &value }

func mustJSON(t *testing.T, value any) json.RawMessage {
	t.Helper()
	encoded, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	return encoded
}
