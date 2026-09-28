package contextcompression

import (
	"context"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func TestCompressionUsesPlanAfterHistorySelection(t *testing.T) {
	request := exchangeFixture()
	// Place the large result in a retained prior turn, followed by a live user.
	request.Messages[3].Content[0].ToolResult.Content[0].Text = strings.Repeat("unrelated log entry\n", 500) + "important answer\n"
	ir := ParseSemanticRequest(request, Provenance{})
	if err := ir.ApplySteps(context.Background(), []TransformationStep{{Kind: TransformReset}, {Kind: TransformDeduplicate}, {Kind: TransformSelectTurns}}); err != nil {
		t.Fatal(err)
	}
	policy := Policy{Mode: ModeAlways, Budget: Budget{TargetTokens: 500}, Targets: Targets{
		ToolOutputs: TargetPolicy{Mode: TargetExtractive, MinTokens: 10, TargetTokens: 100},
		History:     TargetPolicy{Mode: TargetPreserve}, RAG: TargetPolicy{Mode: TargetPreserve}, Memory: TargetPolicy{Mode: TargetPreserve},
	}}
	result := NewService().Apply(context.Background(), Request{Request: ir, Policy: policy})
	if result.Failure != nil || !result.Applied {
		t.Fatalf("compression failed: %+v", result)
	}
	receipts := ir.Transformations.Receipts()
	if len(receipts) != 4 || receipts[3].Kind != TransformCompress || receipts[3].Status != TransformationApplied {
		t.Fatalf("compression missing from shared plan: %+v", receipts)
	}
	if request.Messages[1].Content[0].ToolCall.ID != "a" || request.Messages[3].Content[0].ToolResult.CallID != "b" {
		t.Fatal("tool exchange changed")
	}
	before := encoded(t, request)
	NewService().Apply(context.Background(), Request{Request: ir, Policy: policy})
	if before != encoded(t, request) || len(ir.Transformations.Receipts()) != 4 {
		t.Fatal("compression repeated")
	}
}

func TestCompressionCannotChangeTrustedProtectedText(t *testing.T) {
	for _, protection := range []Protection{ProtectAuthorization, ProtectSafety} {
		request := historyFixture()
		request.Messages[0].Content[0].Text = strings.Repeat("must preserve permission ", 500)
		ir := ParseSemanticRequest(request, Provenance{ProtectedMessages: map[int]Protection{0: protection}})
		before := request.Messages[0].Content[0].Text
		NewService().Apply(context.Background(), Request{Request: ir, Policy: Policy{Mode: ModeAlways, Targets: Targets{History: TargetPolicy{Mode: TargetExtractive, TargetTokens: 10}}}})
		if request.Messages[0].Content[0].Text != before {
			t.Fatal("protected text compressed")
		}
	}
}

func TestCompressionPreservesMediaWhileEditingSiblingText(t *testing.T) {
	request := exchangeFixture()
	result := request.Messages[3].Content[0].ToolResult
	result.Content[0].Text = strings.Repeat("irrelevant log entry\n", 500)
	result.Content = append(result.Content, llmprotocol.Content{Kind: llmprotocol.ContentImage, URL: "https://example.com/image.png"})
	ir := ParseSemanticRequest(request, Provenance{})
	compressed := NewService().Apply(context.Background(), Request{Request: ir, Policy: Policy{Mode: ModeAlways, Budget: Budget{TargetTokens: 500}, Targets: Targets{
		ToolOutputs: TargetPolicy{Mode: TargetExtractive, TargetTokens: 100}, History: TargetPolicy{Mode: TargetPreserve},
	}}})
	if !compressed.Applied || result.Content[1].URL != "https://example.com/image.png" || result.Content[1].Kind != llmprotocol.ContentImage {
		t.Fatalf("multimodal preservation failed: %+v", compressed)
	}
}

func TestHistoryRemovalKeepsCompressionBoundToSurvivingContent(t *testing.T) {
	request := exchangeFixture()
	request.Messages[3].Content[0].ToolResult.Content[0].Text = strings.Repeat("irrelevant log entry\n", 500)
	request.Messages = append(historyFixture().Messages[:2], request.Messages...)
	ir := ParseSemanticRequest(request, Provenance{})
	if err := ir.ApplySteps(context.Background(), []TransformationStep{removalStep(TransformReset, 0, 1)}); err != nil {
		t.Fatal(err)
	}
	result := NewService().Apply(context.Background(), Request{Request: ir, Policy: Policy{Mode: ModeAlways, Budget: Budget{TargetTokens: 500}, Targets: Targets{
		ToolOutputs: TargetPolicy{Mode: TargetExtractive, TargetTokens: 100}, History: TargetPolicy{Mode: TargetPreserve},
	}}})
	if !result.Applied || result.Failure != nil {
		t.Fatalf("compression after removal failed: %+v", result)
	}
	if len(request.Messages) != 7 || len(request.Messages[3].Content[0].ToolResult.Content[0].Text) >= 10000 {
		t.Fatal("replacement did not reach surviving semantic message")
	}
	if ir.Messages[3].Index != 5 {
		t.Fatal("message identity was renumbered")
	}
}

func TestEnabledHistoryPolicyProtectsEntireLiveHistoryDuringCompression(t *testing.T) {
	request := exchangeFixture()
	// A user-role result is part of the existing turn, not a new user prompt.
	request.Messages = request.Messages[:4]
	request.Messages[3].Role = llmprotocol.RoleUser
	ir := ParseSemanticRequest(request, Provenance{})
	live := ir.Messages[0]
	block := live.Blocks[0]
	legacyEligibility := ir.compressionBlockAllowed(live, block)
	if err := ir.ApplySteps(context.Background(), []TransformationStep{{Kind: TransformReset}}); err != nil {
		t.Fatal(err)
	}
	if ir.compressionBlockAllowed(live, block) != legacyEligibility {
		t.Fatal("disabled step changed legacy compression eligibility")
	}
	if err := ir.ApplySteps(context.Background(), []TransformationStep{{Kind: TransformSelectTurns, Enabled: true, Propose: func(context.Context, TransformationView) (TransformationEdits, error) {
		return TransformationEdits{}, nil
	}}}); err != nil {
		t.Fatal(err)
	}
	if ir.compressionBlockAllowed(live, block) {
		t.Fatal("enabled plan permits compression of the live user prompt")
	}
}
