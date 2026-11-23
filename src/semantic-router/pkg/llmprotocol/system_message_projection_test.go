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
func TestProjectSystemMessagesKeepsAnthropicTextInPlaceAsUserText(t *testing.T) {
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
		projected, diagnostics := ProjectSystemMessages(request, target)
		if !reflect.DeepEqual(projected.Messages, want) {
			t.Fatalf("%s conversation = %+v", target, projected.Messages)
		}
		if !reflect.DeepEqual(projected.Instructions, request.Instructions) {
			t.Fatalf("%s rewrote the leading instruction: %+v", target, projected.Instructions)
		}
		if len(diagnostics) != 1 || diagnostics[0].Action != DiagnosticApproximated ||
			diagnostics[0].Field != "messages[].role" || diagnostics[0].Target != target ||
			diagnostics[0].Source != AnthropicMessagesV1 || diagnostics[0].Reason != "3 system messages sent as user text in place" {
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

// A Chat client gives each instruction its own system message. Qwen3.5 and
// later templates refuse the second one, so a Chat backend receives one system
// message with both texts, in order.
func TestProjectSystemMessagesSendsAChatBackendOneLeadingSystemMessage(t *testing.T) {
	request := Request{
		Generation: 3,
		Trusted:    TrustedMetadata{SourceFormat: OpenAIChatV1},
		Instructions: []InstructionBlock{
			{Role: RoleSystem, Content: systemProjectionText("You are a helpful geography tutor.")},
			{Role: RoleSystem, Content: systemProjectionText("Answer every question in one word.")},
		},
		Messages: []Message{{Role: RoleUser, Content: systemProjectionText("What is the capital of Spain?")}},
	}
	projected, diagnostics := ProjectSystemMessages(request, OpenAIChatV1)
	want := []InstructionBlock{{Role: RoleSystem, Content: systemProjectionText(
		"You are a helpful geography tutor.\n\nAnswer every question in one word.")}}
	if !reflect.DeepEqual(projected.Instructions, want) {
		t.Fatalf("instructions = %+v", projected.Instructions)
	}
	if !reflect.DeepEqual(projected.Messages, request.Messages) {
		t.Fatalf("conversation = %+v", projected.Messages)
	}
	if len(diagnostics) != 1 || diagnostics[0].Field != LossyFieldSystemMessageRole ||
		diagnostics[0].Action != DiagnosticApproximated || diagnostics[0].Source != OpenAIChatV1 ||
		diagnostics[0].Reason != "2 system and developer messages sent as one system message" {
		t.Fatalf("diagnostics = %+v", diagnostics)
	}
	if projected.Generation != request.Generation+1 {
		t.Fatal("the merge did not retire the source envelope")
	}
	if len(request.Instructions) != 2 || request.Instructions[0].Content[0].Text != "You are a helpful geography tutor." {
		t.Fatalf("projection mutated the public request: %+v", request.Instructions)
	}
}

// A Responses request's instructions and developer and system items, or a
// lone developer message, reach a Chat backend as one system message.
func TestProjectSystemMessagesSendsDeveloperInstructionsAsTheSystemMessage(t *testing.T) {
	for name, test := range map[string]struct {
		instructions []InstructionBlock
		text, reason string
	}{
		"responses instructions and items": {
			[]InstructionBlock{
				{Role: RoleDeveloper, Content: systemProjectionText("instructions")},
				{Role: RoleDeveloper, Content: systemProjectionText("permissions")},
				{Role: RoleSystem, Content: systemProjectionText("environment")},
			},
			"instructions\n\npermissions\n\nenvironment", "3 system and developer messages sent as one system message",
		},
		"one developer message": {
			[]InstructionBlock{{Role: RoleDeveloper, Content: systemProjectionText("instructions")}},
			"instructions", "1 system and developer messages sent as one system message",
		},
	} {
		request := Request{
			Trusted: TrustedMetadata{SourceFormat: OpenAIResponsesV1}, Instructions: test.instructions,
			Messages: []Message{{Role: RoleUser, Content: systemProjectionText("hello")}},
		}
		projected, diagnostics := ProjectSystemMessages(request, OpenAIChatV1)
		want := []InstructionBlock{{Role: RoleSystem, Content: systemProjectionText(test.text)}}
		if !reflect.DeepEqual(projected.Instructions, want) {
			t.Fatalf("%s: instructions = %+v", name, projected.Instructions)
		}
		if len(diagnostics) != 1 || diagnostics[0].Reason != test.reason {
			t.Fatalf("%s: diagnostics = %+v", name, diagnostics)
		}
		if request.Instructions[0].Role != RoleDeveloper {
			t.Fatalf("%s: the public request lost its developer role", name)
		}
	}
}

// A system or developer message a Chat client sends inside the conversation
// becomes user text where the client put it: it joins the user message right
// before it, or is a user message of its own after an assistant turn or a tool
// result. One without text is left out.
func TestProjectSystemMessagesSendsLaterChatInstructionsAsUserTextInPlace(t *testing.T) {
	request := Request{
		Trusted: TrustedMetadata{SourceFormat: OpenAIChatV1},
		Instructions: []InstructionBlock{
			{Role: RoleSystem, Content: systemProjectionText("system prompt")},
			{Role: RoleSystem, Content: systemProjectionText("after the answer"), Position: 2},
			{Role: RoleDeveloper, Content: systemProjectionText("after the question"), Position: 3},
			{Role: RoleSystem, Content: systemProjectionText(""), Position: 3},
			{Role: RoleSystem, Content: systemProjectionText("after the tool result"), Position: 5},
		},
		Messages: []Message{
			{Role: RoleUser, Content: systemProjectionText("first")},
			{Role: RoleAssistant, Content: systemProjectionText("answer")},
			{Role: RoleUser, Content: systemProjectionText("second")},
			{Role: RoleAssistant, Content: []Content{{Kind: ContentToolCall, ToolCall: &ToolCall{ID: "call_1", Name: "read"}}}},
			{Role: RoleTool, Content: []Content{{Kind: ContentToolResult, ToolResult: &ToolResult{CallID: "call_1"}}}},
		},
	}
	projected, diagnostics := ProjectSystemMessages(request, OpenAIChatV1)
	want := []Message{
		request.Messages[0],
		request.Messages[1],
		{Role: RoleUser, Content: systemProjectionText("after the answer")},
		{Role: RoleUser, Content: append(systemProjectionText("second"), systemProjectionText("after the question")...)},
		request.Messages[3],
		request.Messages[4],
		{Role: RoleUser, Content: systemProjectionText("after the tool result")},
	}
	if !reflect.DeepEqual(projected.Messages, want) {
		t.Fatalf("conversation = %+v", projected.Messages)
	}
	if !reflect.DeepEqual(projected.Instructions, request.Instructions[:1]) {
		t.Fatalf("instructions = %+v", projected.Instructions)
	}
	if len(diagnostics) != 1 || diagnostics[0].Reason != "4 system messages sent as user text in place" {
		t.Fatalf("diagnostics = %+v", diagnostics)
	}
	if len(request.Messages) != 5 || len(request.Messages[2].Content) != 1 || len(request.Instructions) != 5 {
		t.Fatalf("projection mutated the public request: %+v", request)
	}
}

// Leading instructions merged and a later one sent in place are reported
// once, with both counts.
func TestProjectSystemMessagesReportsAMergeAndUserTextOnce(t *testing.T) {
	request := Request{
		Trusted: TrustedMetadata{SourceFormat: OpenAIChatV1},
		Instructions: []InstructionBlock{
			{Role: RoleSystem, Content: systemProjectionText("A")},
			{Role: RoleDeveloper, Content: systemProjectionText("B")},
			{Role: RoleSystem, Content: systemProjectionText("C"), Position: 1},
		},
		Messages: []Message{{Role: RoleUser, Content: systemProjectionText("question")}},
	}
	projected, diagnostics := ProjectSystemMessages(request, OpenAIChatV1)
	if !reflect.DeepEqual(projected.Instructions, []InstructionBlock{{Role: RoleSystem, Content: systemProjectionText("A\n\nB")}}) ||
		!reflect.DeepEqual(projected.Messages, []Message{{Role: RoleUser, Content: append(systemProjectionText("question"), systemProjectionText("C")...)}}) {
		t.Fatalf("projected instructions %+v, conversation %+v", projected.Instructions, projected.Messages)
	}
	if len(diagnostics) != 1 || diagnostics[0].Reason != "2 system and developer messages sent as one system message, 1 as user text in place" {
		t.Fatalf("diagnostics = %+v", diagnostics)
	}
}

// Text joins the plain text before it, so the merged message reads as one
// text, and a cache boundary stays at the end of the content that set it.
// Empty instructions add nothing, and a name survives only when they share it.
func TestProjectSystemMessagesJoinsContentAroundCacheBoundaries(t *testing.T) {
	cache := &CacheDirective{Type: "ephemeral", TTL: "5m"}
	cached := func(text string) []Content { return []Content{{Kind: ContentText, Text: text, Cache: cache}} }
	image := Content{Kind: ContentImage, URL: "https://example.com/logo.png"}
	for name, test := range map[string]struct {
		instructions []InstructionBlock
		want         InstructionBlock
	}{
		"boundary after the second": {
			[]InstructionBlock{{Role: RoleSystem, Content: systemProjectionText("A")}, {Role: RoleSystem, Content: cached("B")}},
			InstructionBlock{Role: RoleSystem, Content: cached("A\n\nB")},
		},
		"boundary after the first": {
			[]InstructionBlock{{Role: RoleSystem, Content: cached("A")}, {Role: RoleSystem, Content: systemProjectionText("B")}},
			InstructionBlock{Role: RoleSystem, Content: append(cached("A"), systemProjectionText("\n\nB")...)},
		},
		"blocks around an image": {
			[]InstructionBlock{{Role: RoleSystem, Content: systemProjectionText("A")}, {Role: RoleSystem, Content: []Content{image}}, {Role: RoleSystem, Content: systemProjectionText("C")}},
			InstructionBlock{Role: RoleSystem, Content: []Content{{Kind: ContentText, Text: "A\n\n"}, image, {Kind: ContentText, Text: "\n\nC"}}},
		},
		"empty instructions": {
			[]InstructionBlock{{Role: RoleSystem, Content: systemProjectionText("")}, {Role: RoleSystem}, {Role: RoleDeveloper, Content: systemProjectionText("B")}},
			InstructionBlock{Role: RoleSystem, Content: systemProjectionText("B")},
		},
		"only empty instructions": {
			[]InstructionBlock{{Role: RoleSystem, Content: systemProjectionText("")}, {Role: RoleDeveloper, Content: systemProjectionText("")}},
			InstructionBlock{Role: RoleSystem, Content: systemProjectionText("")},
		},
		"a shared name": {
			[]InstructionBlock{{Role: RoleSystem, Name: "tutor", Content: systemProjectionText("A")}, {Role: RoleSystem, Name: "tutor", Content: systemProjectionText("B")}},
			InstructionBlock{Role: RoleSystem, Name: "tutor", Content: systemProjectionText("A\n\nB")},
		},
		"different names": {
			[]InstructionBlock{{Role: RoleSystem, Name: "tutor", Content: systemProjectionText("A")}, {Role: RoleSystem, Name: "grader", Content: systemProjectionText("B")}},
			InstructionBlock{Role: RoleSystem, Content: systemProjectionText("A\n\nB")},
		},
	} {
		request := Request{Instructions: test.instructions, Messages: []Message{{Role: RoleUser, Content: systemProjectionText("hello")}}}
		projected, _ := ProjectSystemMessages(request, OpenAIChatV1)
		if !reflect.DeepEqual(projected.Instructions, []InstructionBlock{test.want}) {
			t.Fatalf("%s: instructions = %+v", name, projected.Instructions)
		}
	}
}

// One system instruction already suits every template, and other backend
// formats keep the instructions they can carry.
func TestProjectSystemMessagesLeavesOtherRequestsAndTargets(t *testing.T) {
	one := Request{
		Generation:   2,
		Trusted:      TrustedMetadata{SourceFormat: OpenAIChatV1},
		Instructions: []InstructionBlock{{Role: RoleSystem, Content: systemProjectionText("A")}},
		Messages:     []Message{{Role: RoleUser, Content: systemProjectionText("hello")}},
	}
	two := one
	two.Instructions = append(append([]InstructionBlock(nil), one.Instructions...),
		InstructionBlock{Role: RoleSystem, Content: systemProjectionText("B")},
		InstructionBlock{Role: RoleSystem, Content: systemProjectionText("C"), Position: 1})
	anthropic := Request{Trusted: TrustedMetadata{SourceFormat: AnthropicMessagesV1}, Messages: []Message{
		{Role: RoleUser, Content: systemProjectionText("first")},
		{Role: RoleSystem, Content: systemProjectionText("environment")},
	}}
	for _, check := range []struct {
		name    string
		request Request
		target  WireFormat
	}{
		{"one system instruction", one, OpenAIChatV1},
		{"chat instructions to responses", two, OpenAIResponsesV1},
		{"chat instructions to messages", two, AnthropicMessagesV1},
		{"anthropic system message to messages", anthropic, AnthropicMessagesV1},
	} {
		projected, diagnostics := ProjectSystemMessages(check.request, check.target)
		if diagnostics != nil || !reflect.DeepEqual(projected, check.request) {
			t.Fatalf("%s changed the request: %+v %+v", check.name, projected, diagnostics)
		}
	}
}

// The developer-role projection runs after this one and finds no developer
// message left, so the request reports one messages[].role diagnostic.
func TestProjectSystemMessagesLeavesNoDeveloperRoleToReport(t *testing.T) {
	request := Request{
		Trusted: TrustedMetadata{SourceFormat: OpenAIChatV1},
		Instructions: []InstructionBlock{
			{Role: RoleDeveloper, Content: systemProjectionText("The secret word is PAPAYA.")},
			{Role: RoleDeveloper, Content: systemProjectionText("Answer in one word."), Position: 1},
		},
		Messages: []Message{{Role: RoleUser, Content: systemProjectionText("What is the secret word?")}},
	}
	projected, diagnostics := ProjectSystemMessages(request, OpenAIChatV1)
	if len(diagnostics) != 1 {
		t.Fatalf("diagnostics = %+v", diagnostics)
	}
	if _, developerDiagnostics := ProjectUnsupportedDeveloperRole(projected, Capabilities(CapabilityDeveloperRole), OpenAIChatV1); developerDiagnostics != nil {
		t.Fatalf("the developer-role projection reported again: %+v", developerDiagnostics)
	}
}
