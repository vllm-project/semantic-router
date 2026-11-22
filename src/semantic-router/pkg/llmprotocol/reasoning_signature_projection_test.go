package llmprotocol

import (
	"errors"
	"reflect"
	"strings"
	"testing"
)

// signedHistory is a Messages conversation that an Anthropic backend answered
// with signed thinking, as decoding leaves it.
func signedHistory() Request {
	return Request{
		Generation: 3,
		Trusted:    TrustedMetadata{SourceFormat: AnthropicMessagesV1},
		Messages: []Message{
			{Role: RoleUser, Content: []Content{{Kind: ContentText, Text: "What is 2+2?"}}, SourcePosition: 1},
			{Role: RoleAssistant, Content: []Content{
				{Kind: ContentReasoning, Text: "simple arithmetic", Signature: "EqQBCkYIBxgCKkBexample", Reasoning: ReasoningScopeText},
				{Kind: ContentText, Text: "4"},
			}, SourcePosition: 2},
			{Role: RoleUser, Content: []Content{{Kind: ContentText, Text: "And 3+3?"}}, SourcePosition: 3},
		},
	}
}

// The projected request keeps every block's thinking text without its
// signature, inside tool results too, while the caller's request still holds
// the signatures.
func TestProjectReasoningSignaturesKeepsTheThinkingText(t *testing.T) {
	request := signedHistory()
	request.Messages = append(request.Messages, Message{Role: RoleTool, SourcePosition: 4, Content: []Content{{
		Kind: ContentToolResult, ToolResult: &ToolResult{CallID: "toolu_1", Content: []Content{
			{Kind: ContentText, Text: "ok"}, {Kind: ContentReasoning, Text: "nested", Signature: "nested-signature"},
		}},
	}}})
	for _, target := range []WireFormat{OpenAIChatV1, OpenAIResponsesV1} {
		projected, diagnostics := ProjectReasoningSignatures(request, target)
		if thinking := projected.Messages[1].Content; thinking[0].Signature != "" || thinking[0].Text != "simple arithmetic" ||
			thinking[0].Reasoning != ReasoningScopeText || thinking[1].Text != "4" {
			t.Fatalf("%s: assistant content = %+v", target, thinking)
		}
		if nested := projected.Messages[3].Content[0].ToolResult.Content; nested[1].Signature != "" || nested[1].Text != "nested" || nested[0].Text != "ok" {
			t.Fatalf("%s: tool result content = %+v", target, nested)
		}
		if projected.Generation != request.Generation+1 {
			t.Fatalf("%s: generation = %d, want %d", target, projected.Generation, request.Generation+1)
		}
		if request.Messages[1].Content[0].Signature != "EqQBCkYIBxgCKkBexample" ||
			request.Messages[3].Content[0].ToolResult.Content[1].Signature != "nested-signature" {
			t.Fatalf("%s: the source request was changed", target)
		}
		want := Diagnostic{
			Source: AnthropicMessagesV1, Target: target, Field: LossyFieldReasoningSignature, Action: DiagnosticApproximated,
			Reason: "the selected backend cannot accept a reasoning signature; the reasoning text is kept",
		}
		if len(diagnostics) != 1 || diagnostics[0] != want {
			t.Fatalf("%s: diagnostics = %+v", target, diagnostics)
		}
	}
}

// Thinking without a signature, an empty one included, reaches the backend as
// it is, and so does any request bound for a Messages backend.
func TestProjectReasoningSignaturesLeavesOtherRequestsAlone(t *testing.T) {
	unsigned := signedHistory()
	unsigned.Messages[1].Content[0].Signature = ""
	for name, test := range map[string]struct {
		request Request
		target  WireFormat
	}{
		"unsigned thinking":   {unsigned, OpenAIChatV1},
		"no thinking":         {Request{Messages: []Message{{Role: RoleUser, Content: []Content{{Kind: ContentText, Text: "hi"}}}}}, OpenAIResponsesV1},
		"to a Messages model": {signedHistory(), AnthropicMessagesV1},
	} {
		projected, diagnostics := ProjectReasoningSignatures(test.request, test.target)
		if diagnostics != nil || !reflect.DeepEqual(projected, test.request) || &projected.Messages[0] != &test.request.Messages[0] {
			t.Fatalf("%s: projected %+v with diagnostics %+v", name, projected, diagnostics)
		}
	}
	if err := RefuseReasoningSignatures(unsigned); err != nil {
		t.Fatalf("an empty signature was refused: %v", err)
	}
	if RequiredCapabilities(unsigned).Supports(CapabilityReasoningSignature) {
		t.Fatal("an empty signature requires the signature capability")
	}
}

// The refusal names the signed block as the client sent it and the setting
// that accepts the loss, never a wire format or capability identifier.
// Decoding splits a client message at its tool results, so a block's index
// counts every part of that message.
func TestRefuseReasoningSignaturesNamesTheClientField(t *testing.T) {
	split := Request{Messages: []Message{
		{Role: RoleUser, Content: []Content{{Kind: ContentText, Text: "run it"}}, SourcePosition: 1},
		{Role: RoleAssistant, Content: []Content{{Kind: ContentToolCall, ToolCall: &ToolCall{ID: "toolu_1", Name: "Bash", Arguments: "{}"}}}, SourcePosition: 2},
		{Role: RoleUser, Content: []Content{{Kind: ContentText, Text: "output follows"}}, WithToolResults: true, SourcePosition: 3},
		{Role: RoleTool, Content: []Content{{Kind: ContentToolResult, ToolResult: &ToolResult{CallID: "toolu_1", Content: []Content{
			{Kind: ContentText, Text: "ok"}, {Kind: ContentReasoning, Text: "nested", Signature: "nested-signature"},
		}}}}, SourcePosition: 3},
	}}
	added := Request{Messages: []Message{{Role: RoleAssistant, Content: []Content{{Kind: ContentReasoning, Text: "plan", Signature: "signature"}}}}}
	for name, test := range map[string]struct {
		request Request
		want    string
	}{
		"assistant thinking":     {signedHistory(), "messages[1].content[0].signature: the selected model cannot accept signed thinking"},
		"inside a split message": {split, "messages[2].content[1].content[1].signature: the selected model cannot accept signed thinking"},
		"position unknown":       {added, "the selected model cannot accept signed thinking in the conversation"},
	} {
		var refusal *ProtocolError
		if !errors.As(RefuseReasoningSignatures(test.request), &refusal) {
			t.Fatalf("%s: no refusal", name)
		}
		if refusal.Category != ErrorUnsupportedFeature || refusal.Code != "unsupported_capability" ||
			refusal.Message != test.want+OptInHint(LossyFieldReasoningSignature) {
			t.Fatalf("%s: refusal = %+v", name, refusal)
		}
		if strings.Contains(refusal.Message, "reasoning_signature") || strings.Contains(refusal.Message, string(OpenAIChatV1)) {
			t.Fatalf("%s: the refusal names an internal identifier: %q", name, refusal.Message)
		}
	}
}
