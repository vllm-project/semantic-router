package llmprotocol

import "testing"

func TestStructuredOutputWithToolsNeedsJSONOutputAndACallableTool(t *testing.T) {
	tools := []Tool{{Name: "get_weather", InputSchema: []byte(`{"type":"object"}`)}}
	schema := OutputFormat{Kind: OutputJSONSchema, Name: "answer", Schema: []byte(`{"type":"object"}`)}
	for name, test := range map[string]struct {
		request Request
		want    bool
	}{
		"json_schema and tools":           {Request{Tools: tools, OutputFormat: schema}, true},
		"json_object and tools":           {Request{Tools: tools, OutputFormat: OutputFormat{Kind: OutputJSONObject}}, true},
		"tool_choice auto":                {Request{Tools: tools, OutputFormat: schema, ToolChoice: ToolChoice{Mode: ToolChoiceAuto}}, true},
		"tool_choice required":            {Request{Tools: tools, OutputFormat: schema, ToolChoice: ToolChoice{Mode: ToolChoiceRequired}}, true},
		"named tool_choice":               {Request{Tools: tools, OutputFormat: schema, ToolChoice: ToolChoice{Mode: ToolChoiceNamed, Name: "get_weather"}}, true},
		"tool_choice none":                {Request{Tools: tools, OutputFormat: schema, ToolChoice: ToolChoice{Mode: ToolChoiceNone}}, false},
		"no tools":                        {Request{OutputFormat: schema}, false},
		"text output":                     {Request{Tools: tools, OutputFormat: OutputFormat{Kind: OutputText}}, false},
		"no output format":                {Request{Tools: tools}, false},
		"image generation without a tool": {Request{ImageGeneration: &ImageGenerationOptions{}, OutputFormat: schema}, false},
	} {
		if got := StructuredOutputWithTools(test.request); got != test.want {
			t.Errorf("%s: StructuredOutputWithTools = %v, want %v", name, got, test.want)
		}
	}
}

func TestFinalAnswerEndsTheTurnWithoutCallingATool(t *testing.T) {
	text := func(stop StopReason, contents ...Content) Response {
		return Response{StopReason: stop, Output: []OutputItem{{Role: RoleAssistant, Content: contents}}}
	}
	answer := Content{Kind: ContentText, Text: `It is sunny.`}
	reasoning := Content{Kind: ContentReasoning, Text: "The tool said sunny."}
	call := Content{Kind: ContentToolCall, ToolCall: &ToolCall{ID: "call_1", Name: "get_weather", Arguments: `{"city":"Paris"}`}}
	for name, test := range map[string]struct {
		response Response
		want     bool
	}{
		"text at the end of the turn":   {text(StopEndTurn, reasoning, answer), true},
		"text at a stop sequence":       {text(StopSequence, answer), true},
		"text with no stated reason":    {text(StopUnknown, answer), true},
		"text with an empty reason":     {text("", answer), true},
		"only reasoning":                {text(StopEndTurn, reasoning), true},
		"a tool call":                   {text(StopToolCall, call), false},
		"text and a tool call":          {text(StopToolCall, answer, call), false},
		"a tool call that ended a turn": {text(StopEndTurn, call), false},
		"a refusal":                     {text(StopEndTurn, Content{Kind: ContentRefusal, Text: "I can't."}), false},
		"cut at the output limit":       {text(StopMaxTokens, answer), false},
		"filtered":                      {text(StopContentFilter, answer), false},
		"paused":                        {text(StopPaused, answer), false},
		"failed":                        {Response{StopReason: StopError, Error: NewError(ErrorUpstreamUnavailable, "x", "y", nil)}, false},
		"several candidates":            {Response{StopReason: StopEndTurn, Alternatives: [][]OutputItem{{}}}, false},
	} {
		if got := FinalAnswer(test.response); got != test.want {
			t.Errorf("%s: FinalAnswer = %v, want %v", name, got, test.want)
		}
	}
}

func TestConstrainedAnswerRequestMakesNoToolCallableForABufferedAnswer(t *testing.T) {
	parallel, include := true, true
	request := Request{
		Generation: 4, Stream: true, StreamOptions: StreamOptions{IncludeUsage: &include},
		Tools:             []Tool{{Name: "get_weather", InputSchema: []byte(`{"type":"object"}`)}},
		ToolChoice:        ToolChoice{Mode: ToolChoiceRequired},
		ParallelToolCalls: &parallel,
		OutputFormat:      OutputFormat{Kind: OutputJSONSchema, Name: "answer", Schema: []byte(`{"type":"object"}`)},
		Messages:          []Message{{Role: RoleUser, Content: []Content{{Kind: ContentText, Text: "Umbrella?"}}}},
	}
	constrained := ConstrainedAnswerRequest(request)
	if constrained.ToolChoice.Mode != ToolChoiceNone || constrained.ParallelToolCalls != nil ||
		constrained.Stream || constrained.StreamOptions.IncludeUsage != nil || constrained.Generation != 5 {
		t.Fatalf("constrained = %+v", constrained)
	}
	if len(constrained.Tools) != 1 || constrained.OutputFormat.Kind != OutputJSONSchema || len(constrained.Messages) != 1 {
		t.Fatalf("constrained request lost the turn: %+v", constrained)
	}
	if request.ToolChoice.Mode != ToolChoiceRequired || !request.Stream || request.ParallelToolCalls == nil || request.Generation != 4 {
		t.Fatalf("source request changed: %+v", request)
	}
}
