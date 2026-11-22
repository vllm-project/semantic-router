package llmprotocol

import "testing"

func TestProjectToolChoiceNoneLeavesToolsOutOnlyForChat(t *testing.T) {
	parallel := false
	request := Request{
		Generation: 3,
		Tools:      []Tool{{Name: "get_time", InputSchema: []byte(`{"type":"object"}`)}},
		ToolChoice: ToolChoice{Mode: ToolChoiceNone}, ParallelToolCalls: &parallel,
		Trusted: TrustedMetadata{SourceFormat: AnthropicMessagesV1},
	}
	projected, diagnostics := ProjectToolChoiceNone(request, OpenAIChatV1)
	if len(projected.Tools) != 0 || projected.ToolChoice.Mode != "" || projected.ParallelToolCalls != nil ||
		projected.Generation != request.Generation+1 {
		t.Fatalf("projected = %+v", projected)
	}
	if len(request.Tools) != 1 || request.ToolChoice.Mode != ToolChoiceNone || request.ParallelToolCalls == nil {
		t.Fatalf("source request changed: %+v", request)
	}
	if len(diagnostics) != 1 || diagnostics[0].Field != "tools" || diagnostics[0].Action != DiagnosticDropped ||
		diagnostics[0].Source != AnthropicMessagesV1 || diagnostics[0].Target != OpenAIChatV1 {
		t.Fatalf("diagnostics = %+v", diagnostics)
	}

	auto := request
	auto.ToolChoice = ToolChoice{Mode: ToolChoiceAuto}
	for name, test := range map[string]struct {
		request Request
		target  WireFormat
	}{
		"messages backend":  {request, AnthropicMessagesV1},
		"responses backend": {request, OpenAIResponsesV1},
		"automatic choice":  {auto, OpenAIChatV1},
		"no tools":          {Request{ToolChoice: ToolChoice{Mode: ToolChoiceNone}}, OpenAIChatV1},
	} {
		unchanged, diagnostics := ProjectToolChoiceNone(test.request, test.target)
		if len(unchanged.Tools) != len(test.request.Tools) || unchanged.ToolChoice != test.request.ToolChoice ||
			unchanged.Generation != test.request.Generation || len(diagnostics) != 0 {
			t.Fatalf("%s: projected = %+v, diagnostics = %+v", name, unchanged, diagnostics)
		}
	}
}
