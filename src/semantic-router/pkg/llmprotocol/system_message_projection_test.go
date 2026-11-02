package llmprotocol

import (
	"reflect"
	"testing"
)

func systemProjectionText(text string) []Content {
	return []Content{{Kind: ContentText, Text: text}}
}

// Claude Code sends its environment section after the first user turn and
// later notices (a file changed on disk, a truncated read) after the tool
// results they refer to. Chat templates that accept a system message only
// first must receive that text as user text in the same place.
func TestProjectAnthropicSystemMessagesKeepsTextInPlaceAsUserText(t *testing.T) {
	request := Request{
		Trusted:      TrustedMetadata{SourceFormat: AnthropicMessagesV1},
		Instructions: []InstructionBlock{{Role: RoleSystem, Content: systemProjectionText("system prompt")}},
		Messages: []Message{
			{Role: RoleUser, Content: systemProjectionText("first")},
			{Role: RoleSystem, Content: systemProjectionText("environment")},
			{Role: RoleAssistant, Content: systemProjectionText("reading")},
			{Role: RoleTool, Content: systemProjectionText("file body")},
			{Role: RoleSystem, Content: systemProjectionText("file changed")},
			{Role: RoleAssistant, Content: systemProjectionText("answer")},
			{Role: RoleSystem, Content: systemProjectionText("date"), ReasoningEffort: "low"},
			{Role: RoleSystem, ReasoningEffort: "medium"},
			{Role: RoleUser, Content: systemProjectionText("second")},
		},
	}
	want := []Message{
		{Role: RoleUser, Content: append(systemProjectionText("first"), systemProjectionText("environment")...)},
		{Role: RoleAssistant, Content: systemProjectionText("reading")},
		{Role: RoleTool, Content: systemProjectionText("file body")},
		{Role: RoleUser, Content: systemProjectionText("file changed")},
		{Role: RoleAssistant, Content: systemProjectionText("answer")},
		{Role: RoleUser, Content: systemProjectionText("date")},
		{Role: RoleSystem, ReasoningEffort: "low"},
		{Role: RoleSystem, ReasoningEffort: "medium"},
		{Role: RoleUser, Content: systemProjectionText("second")},
	}
	for _, target := range []WireFormat{OpenAIChatV1, OpenAIResponsesV1} {
		projected, diagnostics := ProjectAnthropicSystemMessages(request, target)
		if !reflect.DeepEqual(projected.Messages, want) {
			t.Fatalf("%s conversation = %+v", target, projected.Messages)
		}
		if !reflect.DeepEqual(projected.Instructions, request.Instructions) {
			t.Fatalf("%s rewrote the leading instruction: %+v", target, projected.Instructions)
		}
		if len(diagnostics) != 1 || diagnostics[0].Action != DiagnosticApproximated ||
			diagnostics[0].Field != "messages[].role" || diagnostics[0].Target != target {
			t.Fatalf("%s diagnostics = %+v", target, diagnostics)
		}
		if projected.Generation != request.Generation+1 {
			t.Fatalf("%s did not retire the source envelope", target)
		}
	}
	if len(request.Messages) != 9 || len(request.Messages[0].Content) != 1 || request.Messages[1].Role != RoleSystem {
		t.Fatalf("projection mutated the public request: %+v", request)
	}
}

func TestProjectAnthropicSystemMessagesLeavesOtherTargetsAndSources(t *testing.T) {
	messages := []Message{
		{Role: RoleUser, Content: systemProjectionText("first")},
		{Role: RoleSystem, Content: systemProjectionText("environment")},
	}
	for _, check := range []struct {
		source, target WireFormat
	}{
		{AnthropicMessagesV1, AnthropicMessagesV1},
		{OpenAIChatV1, OpenAIChatV1},
	} {
		request := Request{Trusted: TrustedMetadata{SourceFormat: check.source}, Messages: messages}
		projected, diagnostics := ProjectAnthropicSystemMessages(request, check.target)
		if diagnostics != nil || !reflect.DeepEqual(projected, request) {
			t.Fatalf("%s to %s changed the request: %+v %+v", check.source, check.target, projected, diagnostics)
		}
	}
}
