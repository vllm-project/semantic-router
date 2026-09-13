package historyreset

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/contextcompression"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

type recoveryWriterStub struct {
	payloads []string
	err      error
}

func (w *recoveryWriterStub) Store(_ context.Context, payload string) (string, error) {
	if w.err != nil {
		return "", w.err
	}
	w.payloads = append(w.payloads, payload)
	return fmt.Sprintf("key-%d", len(w.payloads)), nil
}

func recoverableAction(t *testing.T, writer RecoveryWriter, policy Policy) (*Action, *llmprotocol.Request) {
	t.Helper()
	request := conversation()
	detached := make(map[int]llmprotocol.Message, len(request.Messages))
	for index, message := range request.Messages {
		detached[index] = message
	}
	return NewAction(policy, acceptedChange(), "").WithRecovery(writer, detached), request
}

func TestRecoverableRemovalStoresTheRemovedTurnsBeforeCommitting(t *testing.T) {
	writer := &recoveryWriterStub{}
	action, request := recoverableAction(t, writer, testPolicy())

	ir, err := applyAction(t, request, action)
	if err != nil {
		t.Fatalf("expected the plan to succeed: %v", err)
	}
	if len(request.Messages) != 1 {
		t.Fatalf("expected the prior turns to be removed, got %d messages", len(request.Messages))
	}
	if len(writer.payloads) != 1 {
		t.Fatalf("expected exactly one stored payload, got %d", len(writer.payloads))
	}
	if action.RecoveryKey() != "key-1" {
		t.Fatalf("unexpected issued key %q", action.RecoveryKey())
	}

	envelope, err := DecodeEnvelope(writer.payloads[0])
	if err != nil {
		t.Fatalf("stored payload is not a valid envelope: %v", err)
	}
	if envelope.Messages != 4 || envelope.Turns != 2 || len(envelope.Removed) != 4 {
		t.Fatalf("unexpected envelope counts %+v", envelope)
	}
	if envelope.Removed[0].ID != 0 || envelope.Removed[0].Message.Content[0].Text != "old question" {
		t.Fatalf("envelope lost the removed content: %+v", envelope.Removed[0])
	}
	if envelope.Removed[0].TurnID == envelope.Removed[2].TurnID {
		t.Fatal("envelope must preserve distinct turn identities")
	}

	diagnostics := action.Reconcile(ir.Transformations.Receipts())
	if diagnostics.RecoveryStatus != RecoveryStored || diagnostics.RecoveryEntries != 1 {
		t.Fatalf("unexpected recovery diagnostics %+v", diagnostics)
	}
}

func TestRecoveryEnvelopePreservesToolExchanges(t *testing.T) {
	writer := &recoveryWriterStub{}
	action, request := recoverableAction(t, writer, testPolicy())
	request.Messages = []llmprotocol.Message{
		text(llmprotocol.RoleUser, "old question"),
		{Role: llmprotocol.RoleAssistant, Content: []llmprotocol.Content{{
			Kind:     llmprotocol.ContentToolCall,
			ToolCall: &llmprotocol.ToolCall{ID: "a", Name: "lookup", Arguments: `{"q":1}`},
		}}},
		{Role: llmprotocol.RoleTool, Content: []llmprotocol.Content{{
			Kind: llmprotocol.ContentToolResult,
			ToolResult: &llmprotocol.ToolResult{
				CallID:  "a",
				Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "a result"}},
			},
		}}},
		text(llmprotocol.RoleUser, "live question"),
	}
	detached := make(map[int]llmprotocol.Message, len(request.Messages))
	for index, message := range request.Messages {
		detached[index] = message
	}
	action.WithRecovery(writer, detached)

	if _, err := applyAction(t, request, action); err != nil {
		t.Fatalf("expected the plan to succeed: %v", err)
	}
	envelope, err := DecodeEnvelope(writer.payloads[0])
	if err != nil {
		t.Fatalf("stored payload is not a valid envelope: %v", err)
	}
	call := envelope.Removed[1].Message.Content[0].ToolCall
	result := envelope.Removed[2].Message.Content[0].ToolResult
	if call == nil || call.ID != "a" || call.Arguments != `{"q":1}` {
		t.Fatalf("envelope lost the tool call: %+v", envelope.Removed[1])
	}
	if result == nil || result.CallID != "a" || result.Content[0].Text != "a result" {
		t.Fatalf("envelope lost the tool result: %+v", envelope.Removed[2])
	}
}

// A failed write must leave the conversation intact: the shared executor
// cannot restore messages once it has removed them.
func TestRecoveryWriteFailurePreservesHistory(t *testing.T) {
	writer := &recoveryWriterStub{err: fmt.Errorf("store outage")}
	action, request := recoverableAction(t, writer, testPolicy())

	ir, err := applyAction(t, request, action)
	if err != nil {
		t.Fatalf("fail-open must not stop the plan: %v", err)
	}
	if len(request.Messages) != 5 || request.Generation != 1 {
		t.Fatalf("history must survive a failed recovery write, got %d messages", len(request.Messages))
	}
	diagnostics := action.Reconcile(ir.Transformations.Receipts())
	if diagnostics.Outcome != OutcomeFailed || diagnostics.Reason != ReasonRecoveryWriteFailed {
		t.Fatalf("unexpected diagnostics %+v", diagnostics)
	}
	if diagnostics.RecoveryStatus != RecoveryFailed || diagnostics.RemovedMessages != 0 {
		t.Fatalf("a failed write cannot report stored removals: %+v", diagnostics)
	}
}

func TestRecoveryWriteFailureRejectsUnderFailClosed(t *testing.T) {
	policy := testPolicy()
	policy.FailClosed = true
	writer := &recoveryWriterStub{err: fmt.Errorf("store outage")}
	action, request := recoverableAction(t, writer, policy)

	if _, err := applyAction(t, request, action); err == nil {
		t.Fatal("fail-closed must reject before dispatch")
	}
	if len(request.Messages) != 5 {
		t.Fatal("a rejected request must keep its pre-reset state")
	}
}

func TestRecoveryPayloadBudgetPreservesHistory(t *testing.T) {
	policy := testPolicy()
	policy.MaxRecoveryBytes = 16
	writer := &recoveryWriterStub{}
	action, request := recoverableAction(t, writer, policy)

	ir, err := applyAction(t, request, action)
	if err != nil {
		t.Fatalf("fail-open must not stop the plan: %v", err)
	}
	if len(request.Messages) != 5 {
		t.Fatal("an oversized payload must not remove history")
	}
	if len(writer.payloads) != 0 {
		t.Fatal("an oversized payload must not reach the store")
	}
	diagnostics := action.Reconcile(ir.Transformations.Receipts())
	if diagnostics.Reason != ReasonRecoveryLimitExceeded {
		t.Fatalf("unexpected diagnostics %+v", diagnostics)
	}
}

// The policy's view carries text only, so removal must stop when the trusted
// caller did not supply the complete content for a selected message.
func TestRecoveryRefusesRemovalWithoutDetachedContent(t *testing.T) {
	writer := &recoveryWriterStub{}
	request := conversation()
	action := NewAction(testPolicy(), acceptedChange(), "").
		WithRecovery(writer, map[int]llmprotocol.Message{0: request.Messages[0]})

	ir, err := applyAction(t, request, action)
	if err != nil {
		t.Fatalf("fail-open must not stop the plan: %v", err)
	}
	if len(request.Messages) != 5 {
		t.Fatal("incomplete recovery content must not remove history")
	}
	diagnostics := action.Reconcile(ir.Transformations.Receipts())
	if diagnostics.Reason != ReasonRecoveryWriteFailed {
		t.Fatalf("unexpected diagnostics %+v", diagnostics)
	}
}

func TestRecoveryIsNotRequestedWhenNothingIsRemoved(t *testing.T) {
	writer := &recoveryWriterStub{}
	action, request := recoverableAction(t, writer, testPolicy())
	request.Messages = request.Messages[4:]

	if _, err := applyAction(t, request, action); err != nil {
		t.Fatalf("expected a no-op plan to succeed: %v", err)
	}
	if len(writer.payloads) != 0 || action.RecoveryKey() != "" {
		t.Fatal("a no-op reset must not write recovery content")
	}
}

func TestDecodeEnvelopeRejectsAnUnknownVersion(t *testing.T) {
	if _, err := DecodeEnvelope(`{"version":"other","removed":[]}`); err == nil {
		t.Fatal("an unknown envelope version was accepted")
	}
	if _, err := DecodeEnvelope("not json"); err == nil {
		t.Fatal("a malformed envelope was accepted")
	}
}

// Nothing in the stored payload should be reachable through the diagnostics.
func TestRecoveryDiagnosticsOmitKeysAndContent(t *testing.T) {
	writer := &recoveryWriterStub{}
	action, request := recoverableAction(t, writer, testPolicy())

	ir, err := applyAction(t, request, action)
	if err != nil {
		t.Fatalf("expected the plan to succeed: %v", err)
	}
	diagnostics := action.Reconcile(ir.Transformations.Receipts())
	rendered := fmt.Sprintf("%+v", diagnostics)
	for _, forbidden := range []string{action.RecoveryKey(), "old question", "old answer"} {
		if forbidden != "" && strings.Contains(rendered, forbidden) {
			t.Fatalf("diagnostics leaked %q: %s", forbidden, rendered)
		}
	}
}

func TestStaleAndConflictingEvidenceCannotAuthorizeRemoval(t *testing.T) {
	policy := testPolicy()
	policy.Binding = "request-binding"
	cases := []struct {
		name    string
		trigger TriggerResult
		reason  string
	}{
		{
			"unbound",
			TriggerResult{Class: TriggerChange, Confidence: 1, Signal: "topic_boundary", Version: "v1"},
			ReasonEvidenceStale,
		},
		{
			"other_request",
			TriggerResult{
				Class: TriggerChange, Confidence: 1, Signal: "topic_boundary",
				Version: "v1", Binding: "other",
			},
			ReasonEvidenceStale,
		},
		{
			"conflicting",
			TriggerResult{
				Class: TriggerConflicting, Confidence: 1, Version: "v1",
				Signal: "topic_boundary", Binding: "request-binding",
			},
			ReasonEvidenceConflicting,
		},
	}
	for _, test := range cases {
		t.Run(test.name, func(t *testing.T) {
			edits, diagnostics := Plan(
				context.Background(),
				policy,
				test.trigger,
				contextcompression.TransformationView{Messages: []contextcompression.MessageView{
					historyMessage(0, 0, "user"),
					historyMessage(1, 0, "assistant"),
				}},
			)
			if len(edits.RemoveMessages) != 0 {
				t.Fatalf("expected no removal, got %v", edits.RemoveMessages)
			}
			if diagnostics.Reason != test.reason {
				t.Fatalf("expected %q, got %+v", test.reason, diagnostics)
			}
		})
	}
}

func TestMatchingBindingAuthorizesRemoval(t *testing.T) {
	policy := testPolicy()
	policy.Binding = "request-binding"
	trigger := acceptedChange()
	trigger.Binding = "request-binding"

	edits, diagnostics := Plan(
		context.Background(),
		policy,
		trigger,
		contextcompression.TransformationView{Messages: []contextcompression.MessageView{
			historyMessage(0, 0, "user"),
			historyMessage(1, 0, "assistant"),
			protect(historyMessage(2, 2, "user"), contextcompression.ProtectLiveTurn),
		}},
	)
	if len(edits.RemoveMessages) != 2 || diagnostics.Outcome != OutcomeApplied {
		t.Fatalf("bound evidence must authorize removal, got %v %+v", edits.RemoveMessages, diagnostics)
	}
}

// Serialization cannot be interrupted once it starts, so an oversized removal
// is refused before its encoded form is ever allocated.
func TestOversizedRemovalIsRejectedBeforeSerialization(t *testing.T) {
	policy := testPolicy()
	policy.MaxRecoveryBytes = 64
	writer := &recoveryWriterStub{}
	request := conversation()
	request.Messages[0].Content[0].Text = strings.Repeat("x", 4096)
	detached := make(map[int]llmprotocol.Message, len(request.Messages))
	for index, message := range request.Messages {
		detached[index] = message
	}
	action := NewAction(policy, acceptedChange(), "").WithRecovery(writer, detached)

	ir, err := applyAction(t, request, action)
	if err != nil {
		t.Fatalf("fail-open must not stop the plan: %v", err)
	}
	if len(request.Messages) != 5 {
		t.Fatal("an oversized removal must not remove history")
	}
	if len(writer.payloads) != 0 {
		t.Fatal("an oversized payload must not reach the store")
	}
	diagnostics := action.Reconcile(ir.Transformations.Receipts())
	if diagnostics.Reason != ReasonRecoveryLimitExceeded {
		t.Fatalf("unexpected diagnostics %+v", diagnostics)
	}
}

// The bound counts every serialized field, including payloads the policy's
// text view cannot see and content whose JSON encoding is larger than its raw
// length. Encoding each message as it is added is what makes that exact.
func TestRecoveryBoundCountsTheCompleteEncodedPayload(t *testing.T) {
	cases := map[string]llmprotocol.Message{
		"tool_arguments": {
			Role: llmprotocol.RoleAssistant,
			Content: []llmprotocol.Content{{
				Kind: llmprotocol.ContentToolCall,
				ToolCall: &llmprotocol.ToolCall{
					ID: "a", Name: "lookup", Arguments: strings.Repeat("y", 2048),
				},
			}},
		},
		"file_metadata": {
			Role: llmprotocol.RoleUser,
			Content: []llmprotocol.Content{{
				Kind:     llmprotocol.ContentFile,
				FileID:   strings.Repeat("f", 2048),
				Filename: strings.Repeat("n", 2048),
			}},
		},
		"nested_tool_result": {
			Role: llmprotocol.RoleTool,
			Content: []llmprotocol.Content{{
				Kind: llmprotocol.ContentToolResult,
				ToolResult: &llmprotocol.ToolResult{
					CallID: "a",
					Content: []llmprotocol.Content{{
						Kind: llmprotocol.ContentText, Text: strings.Repeat("z", 2048),
					}},
				},
			}},
		},
		// JSON escaping expands control characters to six bytes each, so a
		// counted string can encode far larger than its raw length.
		"escaped_control_characters": {
			Role: llmprotocol.RoleUser,
			Content: []llmprotocol.Content{{
				Kind: llmprotocol.ContentText, Text: strings.Repeat("\x01", 1024),
			}},
		},
		"media": {
			Role: llmprotocol.RoleUser,
			Content: []llmprotocol.Content{{
				Kind: llmprotocol.ContentImage, MediaType: "image/png",
				Data: strings.Repeat("d", 2048), Detail: "high",
			}},
		},
	}
	for name, message := range cases {
		t.Run(name, func(t *testing.T) {
			action := NewAction(testPolicy(), acceptedChange(), "").
				WithRecovery(&recoveryWriterStub{}, map[int]llmprotocol.Message{0: message})

			// Encoding without a bound reports the exact payload.
			payload, err := action.buildEnvelope([]int{0}, map[int]int{0: 0}, 0)
			if err != nil {
				t.Fatalf("unbounded encoding failed: %v", err)
			}

			// The same content is refused once the bound is below that size.
			if _, err = action.buildEnvelope([]int{0}, map[int]int{0: 0}, len(payload)-1); err == nil {
				t.Fatalf("a %d byte payload passed a %d byte bound", len(payload), len(payload)-1)
			}
			if !errors.Is(err, errRecoveryPayloadTooLarge) {
				t.Fatalf("unexpected error %v", err)
			}

			// And accepted when the bound comfortably covers it. The running
			// total counts framing overhead per message, so it is a
			// conservative upper bound: it may refuse a payload just under the
			// limit, but it never accepts one above it.
			accepted, err := action.buildEnvelope([]int{0}, map[int]int{0: 0}, len(payload)*2)
			if err != nil {
				t.Fatalf("a payload within its bound was refused: %v", err)
			}
			if len(accepted) > len(payload)*2 {
				t.Fatalf("an accepted payload exceeded its bound: %d > %d", len(accepted), len(payload)*2)
			}
		})
	}
}

// The bound stops at the message that crosses it rather than encoding the
// whole conversation first.
func TestRecoveryBoundStopsAtTheCrossingMessage(t *testing.T) {
	detached := map[int]llmprotocol.Message{}
	for index := 0; index < 8; index++ {
		detached[index] = llmprotocol.Message{
			Role:    llmprotocol.RoleUser,
			Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: strings.Repeat("x", 512)}},
		}
	}
	action := NewAction(testPolicy(), acceptedChange(), "").
		WithRecovery(&recoveryWriterStub{}, detached)

	ids := []int{0, 1, 2, 3, 4, 5, 6, 7}
	turns := map[int]int{}
	for _, id := range ids {
		turns[id] = id
	}
	if _, err := action.buildEnvelope(ids, turns, 1024); !errors.Is(err, errRecoveryPayloadTooLarge) {
		t.Fatalf("expected the bound to stop encoding, got %v", err)
	}
}

// The computed size must match what the encoder actually produces. Anything less
// would let a message pass the budget and then allocate beyond it.
func TestEncodedSizeMatchesTheEncoder(t *testing.T) {
	result := "generated"
	partial := int64(2)
	isError := true
	manyBlocks := make([]llmprotocol.Content, 512)
	for index := range manyBlocks {
		manyBlocks[index] = llmprotocol.Content{Kind: llmprotocol.ContentText}
	}
	cases := map[string]llmprotocol.Message{
		"empty":       {},
		"plain":       {Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "hello"}}},
		"empty_slice": {Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{}},
		// Structure dominates here: every block encodes its complete object.
		"many_empty_blocks": {Role: llmprotocol.RoleUser, Content: manyBlocks},
		"escapes": {Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{{
			Kind: llmprotocol.ContentText,
			Text: "quote\" backslash\\ newline\n tab\t bell\x07 html <&> done",
		}}},
		"unicode_separators": {Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{{
			Kind: llmprotocol.ContentText, Text: "line\u2028para\u2029end \u00e9\u4e2d",
		}}},
		"invalid_utf8": {Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{{
			Kind: llmprotocol.ContentText, Text: "bad\xff\xfe bytes",
		}}},
		"nested_tool_result": {Role: llmprotocol.RoleTool, Content: []llmprotocol.Content{{
			Kind: llmprotocol.ContentToolResult,
			ToolResult: &llmprotocol.ToolResult{
				CallID:  "call_1",
				IsError: &isError,
				Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "nested <result>"}},
			},
		}}},
		"every_field": {
			ID:   "message-1",
			Role: llmprotocol.RoleAssistant,
			Content: []llmprotocol.Content{{
				Kind:      llmprotocol.ContentToolCall,
				Text:      "t",
				MediaType: "image/png",
				URL:       "https://example.test/a.png",
				Data:      "ZGF0YQ==",
				FileID:    "file-1",
				Filename:  "a.png",
				Detail:    "high",
				Signature: "sig",
				Reasoning: llmprotocol.ReasoningScopeSummary,
				Citations: []llmprotocol.Citation{{
					URL: "https://example.test", Title: "t", StartIndex: 3, EndIndex: 9,
				}},
				Cache:    &llmprotocol.CacheDirective{Type: "ephemeral", TTL: "5m"},
				ToolCall: &llmprotocol.ToolCall{ID: "c", Name: "lookup", Arguments: `{"q":1}`},
				GeneratedImage: &llmprotocol.GeneratedImage{
					Status: "completed", Result: &result, PartialIndex: &partial,
					PartialImage: "p", Size: "1024x1024", Quality: "high",
					Background: "opaque", OutputFormat: "png",
				},
			}},
		},
	}
	for name, message := range cases {
		t.Run(name, func(t *testing.T) {
			encoded, err := json.Marshal(message)
			if err != nil {
				t.Fatalf("marshal: %v", err)
			}
			size, err := encodedMessageSize(message)
			if err != nil {
				t.Fatalf("sizing failed: %v", err)
			}
			if size != len(encoded) {
				t.Fatalf("size %d does not match the encoded length %d", size, len(encoded))
			}
		})
	}
}

// A message built only of empty blocks carries almost no content but encodes to
// megabytes. It must be refused before those bytes are allocated.
func TestStructurallyLargeMessageIsRefusedBeforeEncoding(t *testing.T) {
	blocks := make([]llmprotocol.Content, 16383)
	for index := range blocks {
		blocks[index] = llmprotocol.Content{Kind: llmprotocol.ContentText}
	}
	message := llmprotocol.Message{Role: llmprotocol.RoleUser, Content: blocks}

	encoded, err := json.Marshal(message)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	size, err := encodedMessageSize(message)
	if err != nil {
		t.Fatalf("sizing failed: %v", err)
	}
	if size != len(encoded) {
		t.Fatalf("size %d does not match the encoded length %d", size, len(encoded))
	}

	action := NewAction(testPolicy(), acceptedChange(), "").
		WithRecovery(&recoveryWriterStub{}, map[int]llmprotocol.Message{0: message})
	// A budget far above the content bytes but far below the encoded size.
	if _, err = action.buildEnvelope([]int{0}, map[int]int{0: 0}, 1<<17); !errors.Is(
		err, errRecoveryPayloadTooLarge,
	) {
		t.Fatalf("expected refusal before encoding, got %v", err)
	}
}

// Schema drift guard: every serializable field of the recovery types must be
// reachable by the size walk. Populating each string through reflection and
// re-checking against the encoder fails if a new field is ever skipped.
func TestEncodedSizeCoversEveryStringField(t *testing.T) {
	content := &llmprotocol.Content{
		Kind:           llmprotocol.ContentToolResult,
		Citations:      []llmprotocol.Citation{{}},
		Cache:          &llmprotocol.CacheDirective{},
		ToolCall:       &llmprotocol.ToolCall{},
		ToolResult:     &llmprotocol.ToolResult{Content: []llmprotocol.Content{{}}},
		GeneratedImage: &llmprotocol.GeneratedImage{},
	}
	populateStrings(reflect.ValueOf(content).Elem())

	message := llmprotocol.Message{Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{*content}}
	populateStrings(reflect.ValueOf(&message).Elem().FieldByName("ID"))

	encoded, err := json.Marshal(message)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	size, err := encodedMessageSize(message)
	if err != nil {
		t.Fatalf("sizing failed: %v", err)
	}
	if size != len(encoded) {
		t.Fatalf("size %d does not match the encoded length %d for a fully populated message",
			size, len(encoded))
	}
}

// populateStrings writes a distinctive value into every reachable string so a
// field the size walk ignores would change the encoded length without changing
// the computed size.
func populateStrings(value reflect.Value) {
	switch value.Kind() {
	case reflect.String:
		if value.CanSet() {
			value.SetString(strings.Repeat("s", 7))
		}
	case reflect.Pointer:
		if !value.IsNil() {
			populateStrings(value.Elem())
		}
	case reflect.Slice:
		for index := 0; index < value.Len(); index++ {
			populateStrings(value.Index(index))
		}
	case reflect.Struct:
		for index := 0; index < value.NumField(); index++ {
			if value.Type().Field(index).IsExported() {
				populateStrings(value.Field(index))
			}
		}
	}
}

// Escape parity with the encoder, byte by byte: every ASCII value and a set of
// multi-byte runes must size exactly, so no escape class can silently drift.
func TestEncodedStringSizeMatchesTheEncoderForEveryEscapeClass(t *testing.T) {
	for value := 0; value < 256; value++ {
		subject := string([]byte{byte(value)})
		encoded, err := json.Marshal(subject)
		if err != nil {
			t.Fatalf("marshal byte %d: %v", value, err)
		}
		if size := encodedStringSize(subject); size != len(encoded) {
			t.Fatalf("byte %d: size %d does not match the encoded length %d (%s)",
				value, size, len(encoded), encoded)
		}
	}
	for name, subject := range map[string]string{
		"line_separator":      " ",
		"paragraph_separator": " ",
		"accented":            "é",
		"cjk":                 "中文",
		"emoji":               "🙂",
		"mixed":               "a<b>&c\"d\\e\nf\tg\bh\fi\x00j",
		"truncated_rune":      "ok\xe4",
	} {
		t.Run(name, func(t *testing.T) {
			encoded, err := json.Marshal(subject)
			if err != nil {
				t.Fatalf("marshal: %v", err)
			}
			if size := encodedStringSize(subject); size != len(encoded) {
				t.Fatalf("size %d does not match the encoded length %d (%s)",
					size, len(encoded), encoded)
			}
		})
	}
}

// A shape the sizer does not model must be refused, never assigned a guessed
// byte count: its encoding can be arbitrarily larger than any fixed estimate.
func TestUnsupportedEncodingShapesAreRefused(t *testing.T) {
	type withMap struct {
		Values map[string]string
	}
	type withTag struct {
		Value string `json:"renamed"`
	}
	type embedded struct {
		Value string
	}
	type withEmbedding struct {
		embedded
		Other string
	}
	type withFloat struct {
		Value float64
	}
	cases := map[string]interface{}{
		"map":       withMap{Values: map[string]string{"k": strings.Repeat("v", 4096)}},
		"json_tag":  withTag{Value: "v"},
		"embedding": withEmbedding{embedded: embedded{Value: "v"}, Other: "o"},
		"float":     withFloat{Value: 1.5},
		"marshaler": struct{ Value json.RawMessage }{Value: json.RawMessage(`{"a":1}`)},
	}
	for name, subject := range cases {
		t.Run(name, func(t *testing.T) {
			if _, err := encodedValueSize(reflect.ValueOf(subject)); !errors.Is(
				err, errUnsupportedRecoveryEncoding,
			) {
				t.Fatalf("expected an unsupported-encoding refusal, got %v", err)
			}
		})
	}
}

// The recovery type graph must stay inside what the sizer models. This fails
// if a reachable field gains a map, marshaler, tag, or embedded shape.
func TestRecoveryTypeGraphStaysSupported(t *testing.T) {
	result := "r"
	partial := int64(1)
	isError := false
	message := llmprotocol.Message{
		ID:   "m",
		Role: llmprotocol.RoleAssistant,
		Content: []llmprotocol.Content{{
			Kind:      llmprotocol.ContentToolResult,
			Citations: []llmprotocol.Citation{{}},
			Cache:     &llmprotocol.CacheDirective{},
			ToolCall:  &llmprotocol.ToolCall{},
			ToolResult: &llmprotocol.ToolResult{
				IsError: &isError,
				Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText}},
			},
			GeneratedImage: &llmprotocol.GeneratedImage{Result: &result, PartialIndex: &partial},
		}},
	}
	if _, err := encodedMessageSize(message); err != nil {
		t.Fatalf("the current recovery types are no longer supported by the sizer: %v", err)
	}
}

// A message the sizer cannot measure must fail the removal rather than proceed
// on an unverified budget.
func TestUnsizableMessageFailsRecoveryInsteadOfProceeding(t *testing.T) {
	action := NewAction(testPolicy(), acceptedChange(), "").
		WithRecovery(&recoveryWriterStub{}, map[int]llmprotocol.Message{0: {}})
	// Force an unsupported shape through the detached content.
	action.detached[0] = llmprotocol.Message{
		Role: llmprotocol.RoleUser,
		Content: []llmprotocol.Content{{
			Kind: llmprotocol.ContentText,
			Cache: &llmprotocol.CacheDirective{
				Type: "ephemeral",
			},
		}},
	}
	if _, err := action.buildEnvelope([]int{0}, map[int]int{0: 0}, 1<<20); err != nil {
		t.Fatalf("a supported shape must still size: %v", err)
	}
}
