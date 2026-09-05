package contextcompression

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"strings"
	"testing"
	"testing/quick"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func historyFixture() *llmprotocol.Request {
	return &llmprotocol.Request{
		Generation: 1, Model: "model", Metadata: map[string]string{"keep": "unchanged"},
		Instructions: []llmprotocol.InstructionBlock{{Role: llmprotocol.RoleDeveloper, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "follow permissions"}}}},
		Messages: []llmprotocol.Message{
			textMessage(llmprotocol.RoleUser, "old question"), textMessage(llmprotocol.RoleAssistant, "old answer"),
			textMessage(llmprotocol.RoleUser, "another question"), textMessage(llmprotocol.RoleAssistant, "another answer"),
			textMessage(llmprotocol.RoleUser, "live question"),
		},
	}
}

func textMessage(role llmprotocol.Role, text string) llmprotocol.Message {
	return llmprotocol.Message{Role: role, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: text}}}
}

func removalStep(kind TransformationKind, ids ...int) TransformationStep {
	return TransformationStep{Kind: kind, Enabled: true, FailureMode: FailureClosed,
		Propose: func(context.Context, TransformationView) (TransformationEdits, error) {
			return TransformationEdits{RemoveMessages: ids}, nil
		},
	}
}

func TestTransformationCompleteTurnAndStableIdentity(t *testing.T) {
	request := historyFixture()
	ir := ParseSemanticRequest(request, Provenance{})
	steps := []TransformationStep{removalStep(TransformReset, 0, 1), {Kind: TransformDeduplicate}, {Kind: TransformSelectTurns}}
	if err := ir.ApplySteps(context.Background(), steps); err != nil {
		t.Fatal(err)
	}
	if len(request.Messages) != 3 || ir.Messages[0].Index != 2 || request.Generation != 2 {
		t.Fatalf("wrong survivors: %+v", ir.TransformationView())
	}
	before := encoded(t, request)
	if err := ir.ApplySteps(context.Background(), steps); err != nil {
		t.Fatal(err)
	}
	if encoded(t, request) != before || len(ir.Transformations.Receipts()) != 3 {
		t.Fatal("repeated plan changed state")
	}
	if request.Instructions[0].Content[0].Text != "follow permissions" || request.Metadata["keep"] != "unchanged" {
		t.Fatal("non-history metadata changed")
	}
}

func TestTransformationRejectsPartialAndProtectedEdits(t *testing.T) {
	for _, tc := range []struct {
		name       string
		ids        []int
		protection Protection
	}{
		{"partial_turn", []int{0}, 0}, {"live_turn", []int{4}, 0}, {"unknown_message", []int{99}, 0},
		{"duplicate_edit", []int{0, 0, 1}, 0}, {"authorization", []int{0, 1}, ProtectAuthorization}, {"safety", []int{0, 1}, ProtectSafety},
	} {
		t.Run(tc.name, func(t *testing.T) {
			request := historyFixture()
			before := encoded(t, request)
			ir := ParseSemanticRequest(request, Provenance{ProtectedMessages: map[int]Protection{0: tc.protection}})
			if err := ir.ApplySteps(context.Background(), []TransformationStep{removalStep(TransformReset, tc.ids...)}); err == nil {
				t.Fatal("expected rejection")
			}
			if encoded(t, request) != before {
				t.Fatal("rejected edit changed request")
			}
			if ir.Transformations.Receipts()[0].Reason != "invariant_violation" {
				t.Fatal("missing rejection receipt")
			}
		})
	}
}

func TestTransformationFailureModesAndDetachedView(t *testing.T) {
	for _, mode := range []FailureMode{FailureOpen, FailureClosed} {
		t.Run(string(mode), func(t *testing.T) {
			request := historyFixture()
			ir := ParseSemanticRequest(request, Provenance{})
			calls := 0
			steps := []TransformationStep{{Kind: TransformReset, Enabled: true, FailureMode: mode,
				Propose: func(_ context.Context, view TransformationView) (TransformationEdits, error) {
					view.Messages[0].Blocks[0].Text = "secret mutation"
					return TransformationEdits{RemoveMessages: []int{0, 1}}, errors.New("private customer text")
				}}, {Kind: TransformSelectTurns, Enabled: true, Propose: func(context.Context, TransformationView) (TransformationEdits, error) {
				calls++
				return TransformationEdits{}, nil
			}}}
			err := ir.ApplySteps(context.Background(), steps)
			if (err != nil) != (mode == FailureClosed) {
				t.Fatalf("unexpected failure: %v", err)
			}
			if (calls == 0) != (mode == FailureClosed) {
				t.Fatal("failure did not control remaining steps")
			}
			if request.Messages[0].Content[0].Text != "old question" {
				t.Fatal("policy mutated request")
			}
			receipt := encoded(t, ir.Transformations.Receipts())
			if strings.Contains(receipt, "private") || strings.Contains(receipt, "old question") {
				t.Fatal("receipt leaked content")
			}
		})
	}
}

func TestTransformationOrderValidatedBeforeMutation(t *testing.T) {
	for _, steps := range [][]TransformationStep{
		{removalStep(TransformSelectTurns, 0, 1), {Kind: TransformReset}},
		{{Kind: TransformReset}, {Kind: TransformReset}},
		{{Kind: TransformCompress}},
	} {
		request := historyFixture()
		before := encoded(t, request)
		ir := ParseSemanticRequest(request, Provenance{})
		if err := ir.ApplySteps(context.Background(), steps); err == nil {
			t.Fatal("invalid order accepted")
		}
		if encoded(t, request) != before || len(ir.Transformations.Receipts()) != 0 {
			t.Fatal("invalid plan partially executed")
		}
	}
}

func TestTransformationBaselineProperty(t *testing.T) {
	property := func(text string) bool {
		request := historyFixture()
		request.Messages[0].Content[0].Text = text
		before := encoded(t, request)
		ir := ParseSemanticRequest(request, Provenance{})
		err := ir.ApplySteps(context.Background(), []TransformationStep{{Kind: TransformReset}, {Kind: TransformDeduplicate}, {Kind: TransformSelectTurns}})
		return err == nil && before == encoded(t, request)
	}
	if err := quick.Check(property, &quick.Config{MaxCount: 100}); err != nil {
		t.Fatal(err)
	}
}

func TestHistorySnapshotAndProvenance(t *testing.T) {
	request := historyFixture()
	snapshot := CaptureHistory(request)
	request.Messages = append(request.Messages, textMessage(llmprotocol.RoleSystem, "memory fact"))
	provenance := Provenance{OriginalHistory: snapshot, MemoryMessageIndexes: map[int]struct{}{5: {}}}
	ir := ParseSemanticRequest(request, provenance)
	history := ir.OriginalHistory()
	history.Messages[0].Content[0].Text = "changed"
	history.Instructions[0].Content[0].Text = "changed"
	if len(ir.OriginalHistory().Messages) != 5 || ir.OriginalHistory().Messages[0].Content[0].Text != "old question" {
		t.Fatal("original snapshot is mutable or enriched")
	}
	if ir.OriginalHistory().Instructions[0].Content[0].Text != "follow permissions" {
		t.Fatal("instructions alias snapshot")
	}
	if ir.Messages[5].Source != SourceMemory || ir.Messages[5].Blocks[0].Source != TargetMemory {
		t.Fatal("memory origin lost")
	}
	if !reflect.DeepEqual(snapshot.Conversation(), ir.OriginalHistory()) {
		t.Fatal("snapshot changed")
	}
}

func encoded(t *testing.T, value interface{}) string {
	t.Helper()
	data, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	return string(data)
}

func FuzzTransformationRemovalInvariants(f *testing.F) {
	f.Add(uint8(3))
	f.Add(uint8(31))
	f.Fuzz(func(t *testing.T, mask uint8) {
		request := historyFixture()
		ir := ParseSemanticRequest(request, Provenance{})
		before := encoded(t, request)
		var ids []int
		for i := 0; i < 5; i++ {
			if mask&(1<<i) != 0 {
				ids = append(ids, i)
			}
		}
		err := ir.ApplySteps(context.Background(), []TransformationStep{removalStep(TransformSelectTurns, ids...)})
		if err != nil {
			if encoded(t, request) != before {
				t.Fatal("failed edit changed request")
			}
			return
		}
		// Only complete historical pairs can be removed in this fixture.
		if mask&31 != 0 && mask&31 != 3 && mask&31 != 12 && mask&31 != 15 {
			t.Fatalf("unsafe deletion accepted: %d", mask)
		}
		if request.Messages[len(request.Messages)-1].Content[0].Text != "live question" {
			t.Fatal("live turn lost")
		}
	})
}

func TestTransformationCancellationAndReceiptIsolation(t *testing.T) {
	request := historyFixture()
	ir := ParseSemanticRequest(request, Provenance{})
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := ir.ApplySteps(ctx, []TransformationStep{removalStep(TransformReset, 0, 1)}); err == nil {
		t.Fatal("canceled plan succeeded")
	}
	receipts := ir.Transformations.Receipts()
	receipts[0].Reason = "changed"
	if ir.Transformations.Receipts()[0].Reason == "changed" {
		t.Fatal("receipt accessor aliases internal state")
	}
	if len(request.Messages) != 5 {
		t.Fatal("canceled plan mutated request")
	}
}
