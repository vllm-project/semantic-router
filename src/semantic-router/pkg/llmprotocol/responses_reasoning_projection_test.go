package llmprotocol

import (
	"reflect"
	"testing"
)

func TestResponsesReasoningProjectionCopiesHistory(t *testing.T) {
	blob := Content{Kind: ContentReasoning, ReasoningDetail: &ReasoningDetail{
		Type: ReasoningDetailEncrypted, Format: OptionalString{Present: true, Value: ReasoningFormatResponses},
		ID: OptionalString{Present: true, Value: "rs"}, Data: OptionalString{Present: true, Value: "cipher"},
	}}
	plain := Content{Kind: ContentReasoning, Text: "private thought", Reasoning: ReasoningScopeText}
	summary := Content{Kind: ContentReasoning, Text: "summary", Reasoning: ReasoningScopeSummary}
	original := Request{Generation: 7, IncludeEncryptedReasoning: true, Messages: []Message{
		{ID: "rs", Role: RoleAssistant, Content: []Content{summary, plain, blob}},
		{ID: "plain", Role: RoleAssistant, Content: []Content{plain}},
		{Role: RoleUser, Content: []Content{{Kind: ContentText, Text: "next"}}},
	}}
	hosted, diagnostics := ProjectResponsesReasoning(original, OpenAIResponsesV1, true)
	if len(diagnostics) != 2 || hosted.Generation != 8 || len(hosted.Messages) != 2 || len(hosted.Messages[0].Content) != 2 || !hosted.IncludeEncryptedReasoning {
		t.Fatalf("hosted projection: %+v, %+v", hosted, diagnostics)
	}
	for _, target := range []WireFormat{OpenAIResponsesV1, OpenAIChatV1, AnthropicMessagesV1} {
		projected, targetDiagnostics := ProjectResponsesReasoning(original, target, false)
		if len(targetDiagnostics) != 2 || projected.Generation != 8 || projected.IncludeEncryptedReasoning || len(projected.Messages) != 3 || !reflect.DeepEqual(projected.Messages[0].Content, []Content{summary, plain}) {
			t.Fatalf("%s projection: %+v, %+v", target, projected, targetDiagnostics)
		}
	}
	if original.Generation != 7 || !original.IncludeEncryptedReasoning || len(original.Messages[0].Content) != 3 || original.Messages[0].Content[1].Text != "private thought" {
		t.Fatalf("projection mutated original: %+v", original)
	}
	unchanged, diagnostics := ProjectResponsesReasoning(hosted, OpenAIResponsesV1, true)
	if len(diagnostics) != 0 || unchanged.Generation != hosted.Generation {
		t.Fatal("projection is not idempotent")
	}
}

func TestResponsesProjectionPreservesSignedThinking(t *testing.T) {
	signed := Content{Kind: ContentReasoning, Text: "thought", Signature: "signature"}
	request := Request{Messages: []Message{{Role: RoleAssistant, Content: []Content{signed}}}}
	projected, diagnostics := ProjectResponsesReasoning(request, OpenAIResponsesV1, true)
	if len(diagnostics) != 0 || !reflect.DeepEqual(request, projected) || !RequiredCapabilities(projected).Supports(CapabilityReasoningSignature) {
		t.Fatalf("signed thinking was dropped before capability checking: %+v, %+v", projected, diagnostics)
	}
}

func TestResponsesBlobCapabilities(t *testing.T) {
	blob := Content{Kind: ContentReasoning, ReasoningDetail: &ReasoningDetail{
		Type: ReasoningDetailEncrypted, Format: OptionalString{Present: true, Value: ReasoningFormatResponses},
		Data: OptionalString{Present: true, Value: "cipher"},
	}}
	for name, capabilities := range map[string]CapabilitySet{
		"request":  RequiredCapabilities(Request{Messages: []Message{{Content: []Content{blob}}}}),
		"response": RequiredResponseCapabilities(Response{Output: []OutputItem{{Content: []Content{blob}}}}),
		"event":    RequiredEventCapabilities(Event{Content: &blob}),
	} {
		if capabilities != Capabilities(CapabilityReasoning) {
			t.Fatalf("%s blob capabilities = %v", name, capabilities.Names())
		}
	}
}
