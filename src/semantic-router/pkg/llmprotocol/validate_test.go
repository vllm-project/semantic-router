package llmprotocol

import (
	"encoding/json"
	"errors"
	"math"
	"reflect"
	"strings"
	"testing"
)

func requireLLMProtocolErrorCode(t *testing.T, err error, code string) {
	t.Helper()
	var protocolError *ProtocolError
	if !errors.As(err, &protocolError) || protocolError.Code != code {
		t.Fatalf("returned %T %v, want protocol error %q", err, err, code)
	}
}

func TestParseCapabilitiesRejectsUnknownNames(t *testing.T) {
	if _, err := ParseCapabilities([]string{"text", "future_product_feature"}); err == nil {
		t.Fatal("unknown capability was silently ignored")
	}
}

func TestCapabilityNamesAndParserStayClosed(t *testing.T) {
	all := Capabilities(
		CapabilityText,
		CapabilityImageInput, CapabilityImageOutput,
		CapabilityAudioInput, CapabilityAudioOutput,
		CapabilityVideoInput, CapabilityVideoOutput,
		CapabilityFileInput, CapabilityFileOutput,
		CapabilityTools, CapabilityParallelTools,
		CapabilityReasoning, CapabilityStructuredJSON, CapabilityStrictJSONSchema, CapabilityStrictToolSchema,
		CapabilityStreaming, CapabilityCacheAccounting, CapabilityReasoningAccounting, CapabilityAuthoritativeUsage,
		CapabilityMultipleCandidates, CapabilityCacheDirectives,
		CapabilityReasoningDisable, CapabilityReasoningEffort, CapabilityReasoningBudget,
		CapabilitySamplingTopK, CapabilitySamplingSeed, CapabilitySamplingPenalties, CapabilityStopSequences,
		CapabilityRequestMetadata, CapabilityRequestStorage, CapabilityAutomaticStorage, CapabilityConversationState,
		CapabilityReasoningAdaptive, CapabilityReasoningSignature, CapabilityReasoningDisplay,
		CapabilityMatchedStopSequence,
		CapabilityImageGeneration,
	)
	names := all.Names()
	seen := make(map[string]struct{}, len(names))
	for _, name := range names {
		if _, duplicate := seen[name]; duplicate {
			t.Fatalf("duplicate capability name %q", name)
		}
		seen[name] = struct{}{}
	}
	parsed, err := ParseCapabilities(names)
	if err != nil {
		t.Fatal(err)
	}
	if got := parsed.Names(); !reflect.DeepEqual(got, names) {
		t.Fatalf("capability parser drifted\n got: %v\nwant: %v", got, names)
	}
}

func TestRequiredCapabilitiesDescribeSemanticContent(t *testing.T) {
	request := validSemanticRequest()
	request.Messages = []Message{{Role: RoleUser, Content: []Content{{Kind: ContentImage, URL: "https://example.com/image.png"}}}}
	request.Tools = nil
	request.ToolChoice = ToolChoice{}
	required := RequiredCapabilities(request)
	if !required.Supports(CapabilityImageInput) || required.Supports(CapabilityText) {
		t.Fatalf("image-only capabilities = %v", required.Names())
	}

	request.Messages = []Message{{
		Role: RoleTool,
		Content: []Content{{Kind: ContentToolResult, ToolResult: &ToolResult{
			CallID: "call_1", Content: []Content{{Kind: ContentFile, FileID: "file_1"}},
		}}},
	}}
	required = RequiredCapabilities(request)
	if !required.Supports(CapabilityTools) || !required.Supports(CapabilityFileInput) {
		t.Fatalf("nested tool result capabilities = %v", required.Names())
	}

	request.ReasoningEffort = "high"
	request.ReasoningDisplay = "summarized"
	request.Tools = []Tool{{Name: "lookup", Strict: Bool(true), InputSchema: json.RawMessage(`{"type":"object"}`)}}
	required = RequiredCapabilities(request)
	if !required.Supports(CapabilityReasoning) || !required.Supports(CapabilityReasoningEffort) ||
		!required.Supports(CapabilityReasoningDisplay) || !required.Supports(CapabilityStrictToolSchema) {
		t.Fatalf("reasoning/strict tool capabilities = %v", required.Names())
	}
}

func TestRequiredCapabilitiesSeparateRequestAndResponseMedia(t *testing.T) {
	request := validSemanticRequest()
	request.Messages[0].Content = []Content{{Kind: ContentAudio, Data: "YQ==", MediaType: "audio/wav"}}
	if required := RequiredCapabilities(request); !required.Supports(CapabilityAudioInput) || required.Supports(CapabilityAudioOutput) {
		t.Fatalf("request capabilities = %v", required.Names())
	}
	response := Response{
		Generation: 1, ID: "response", StopReason: StopEndTurn,
		Output: []OutputItem{{ID: "item", Role: RoleAssistant, Content: []Content{{Kind: ContentAudio, Data: "YQ==", MediaType: "audio/wav"}}}},
		Usage:  Usage{State: UsageUnavailable},
	}
	if required := RequiredResponseCapabilities(response); !required.Supports(CapabilityAudioOutput) || required.Supports(CapabilityAudioInput) {
		t.Fatalf("response capabilities = %v", required.Names())
	}
}

func TestValidateContentClosesReasoningScope(t *testing.T) {
	request := validSemanticRequest()
	request.Messages[0].Content = []Content{{
		Kind: ContentReasoning, Text: "thought", Reasoning: ReasoningScope("future"),
	}}
	if err := ValidateRequest(request, DefaultPolicy().Limits); err == nil {
		t.Fatal("unknown reasoning scope was accepted")
	}

	request.Messages[0].Content = []Content{{
		Kind: ContentText, Text: "answer", Reasoning: ReasoningScopeSummary,
	}}
	if err := ValidateRequest(request, DefaultPolicy().Limits); err == nil {
		t.Fatal("reasoning scope on text content was accepted")
	}
}

func TestMatchedStopSequenceIsResponseOnlyCapability(t *testing.T) {
	response := Response{MatchedStopSequence: "END"}
	if required := RequiredResponseCapabilities(response); !required.Supports(CapabilityMatchedStopSequence) {
		t.Fatalf("response capabilities = %v", required.Names())
	}
	event := Event{MatchedStopSequence: "END"}
	if required := RequiredEventCapabilities(event); !required.Supports(CapabilityMatchedStopSequence) {
		t.Fatalf("event capabilities = %v", required.Names())
	}
	request := validSemanticRequest()
	request.Sampling.Stop = []string{"END"}
	required := RequiredCapabilities(request)
	if !required.Supports(CapabilityStopSequences) || required.Supports(CapabilityMatchedStopSequence) {
		t.Fatalf("request capabilities = %v", required.Names())
	}
}

func TestResponseAlternativesRequireMultipleCandidateCapability(t *testing.T) {
	response := Response{Alternatives: [][]OutputItem{{{ID: "alternative"}}}}
	if required := RequiredResponseCapabilities(response); !required.Supports(CapabilityMultipleCandidates) {
		t.Fatalf("response capabilities = %v", required.Names())
	}
}

func TestValidateRequestCandidateCountAndRetainedToolLink(t *testing.T) {
	limits := DefaultPolicy().Limits
	request := validSemanticRequest()
	request.CandidateCount = Int64(2)
	if err := ValidateRequest(request, limits); err != nil {
		t.Fatalf("valid candidate count rejected: %v", err)
	}
	request.CandidateCount = Int64(int64(limits.Candidates + 1))
	if err := ValidateRequest(request, limits); err == nil {
		t.Fatal("oversized candidate count accepted")
	}

	continuation := validSemanticRequest()
	continuation.PreviousResponseID = "response_previous"
	continuation.Messages = []Message{{Role: RoleTool, Content: []Content{{Kind: ContentToolResult, ToolResult: &ToolResult{
		CallID: "retained_call", DeferredLink: true, Content: []Content{{Kind: ContentText, Text: "done"}},
	}}}}}
	if err := ValidateRequest(continuation, limits); err != nil {
		t.Fatalf("retained-history tool result rejected: %v", err)
	}
	continuation.PreviousResponseID = ""
	if err := ValidateRequest(continuation, limits); err == nil {
		t.Fatal("deferred tool result without retained response accepted")
	}
}

func TestValidateRequestReasoningModesAreUnambiguous(t *testing.T) {
	limits := DefaultPolicy().Limits
	for name, mutate := range map[string]func(*Request){
		"unknown mode":           func(request *Request) { request.ReasoningMode = "future" },
		"enabled without budget": func(request *Request) { request.ReasoningMode = ReasoningModeEnabled },
		"disabled with effort": func(request *Request) {
			request.ReasoningMode = ReasoningModeDisabled
			request.ReasoningEffort = "high"
		},
		"adaptive with budget": func(request *Request) {
			request.ReasoningMode = ReasoningModeAdaptive
			request.ReasoningBudgetTokens = Int64(512)
		},
	} {
		t.Run(name, func(t *testing.T) {
			request := validSemanticRequest()
			mutate(&request)
			if err := ValidateRequest(request, limits); err == nil {
				t.Fatalf("ambiguous reasoning controls were accepted: %+v", request)
			}
		})
	}
	request := validSemanticRequest()
	request.ReasoningMode = ReasoningModeEnabled
	request.ReasoningBudgetTokens = Int64(512)
	if err := ValidateRequest(request, limits); err != nil {
		t.Fatalf("valid enabled reasoning rejected: %v", err)
	}
	request.ReasoningMode = ReasoningModeAdaptive
	request.ReasoningBudgetTokens = nil
	if err := ValidateRequest(request, limits); err != nil {
		t.Fatalf("valid adaptive reasoning rejected: %v", err)
	}
}

func TestValidateRequestReasoningDisplayIsModeScoped(t *testing.T) {
	limits := DefaultPolicy().Limits
	for _, mode := range []ReasoningMode{ReasoningModeEnabled, ReasoningModeAdaptive} {
		for _, display := range []string{"summarized", "omitted"} {
			request := validSemanticRequest()
			request.ReasoningMode = mode
			request.ReasoningDisplay = display
			if mode == ReasoningModeEnabled {
				request.ReasoningBudgetTokens = Int64(1024)
			}
			if err := ValidateRequest(request, limits); err != nil {
				t.Fatalf("valid %s/%s reasoning display rejected: %v", mode, display, err)
			}
		}
	}
	for name, mutate := range map[string]func(*Request){
		"unknown display": func(request *Request) {
			request.ReasoningMode = ReasoningModeAdaptive
			request.ReasoningDisplay = "verbose"
		},
		"display without reasoning": func(request *Request) {
			request.ReasoningDisplay = "omitted"
		},
		"display with disabled reasoning": func(request *Request) {
			request.ReasoningMode = ReasoningModeDisabled
			request.ReasoningDisplay = "summarized"
		},
	} {
		t.Run(name, func(t *testing.T) {
			request := validSemanticRequest()
			mutate(&request)
			if err := ValidateRequest(request, limits); err == nil {
				t.Fatalf("invalid reasoning display accepted: %+v", request)
			}
		})
	}
}

func TestValidateRequestAcceptsPublishedReasoningEfforts(t *testing.T) {
	for _, effort := range []string{"none", "minimal", "low", "medium", "high", "xhigh", "max"} {
		t.Run(effort, func(t *testing.T) {
			request := validSemanticRequest()
			request.ReasoningEffort = effort
			if err := ValidateRequest(request, DefaultPolicy().Limits); err != nil {
				t.Fatalf("published reasoning effort %q was rejected: %v", effort, err)
			}
		})
	}
}

func TestValidateRequestRejectsNoncanonicalReasoningEfforts(t *testing.T) {
	for _, effort := range []string{"HIGH", "hiGh", " high", "high "} {
		t.Run(effort, func(t *testing.T) {
			request := validSemanticRequest()
			request.ReasoningEffort = effort
			requireLLMProtocolErrorCode(t, ValidateRequest(request, DefaultPolicy().Limits), "invalid_reasoning_effort")
		})
	}
}

func TestValidateRequestRejectsNonFiniteSamplingValues(t *testing.T) {
	tests := []struct {
		name   string
		mutate func(*Sampling)
	}{
		{name: "temperature NaN", mutate: func(sampling *Sampling) { sampling.Temperature = Float64(math.NaN()) }},
		{name: "temperature positive infinity", mutate: func(sampling *Sampling) { sampling.Temperature = Float64(math.Inf(1)) }},
		{name: "top p negative infinity", mutate: func(sampling *Sampling) { sampling.TopP = Float64(math.Inf(-1)) }},
		{name: "frequency penalty NaN", mutate: func(sampling *Sampling) { sampling.FrequencyPenalty = Float64(math.NaN()) }},
		{name: "presence penalty infinity", mutate: func(sampling *Sampling) { sampling.PresencePenalty = Float64(math.Inf(1)) }},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			request := validSemanticRequest()
			test.mutate(&request.Sampling)
			if err := ValidateRequest(request, DefaultPolicy().Limits); err == nil {
				t.Fatal("non-finite sampling value was accepted")
			}
		})
	}
}

func TestValidateRequestRejectsUnsafeMediaURLs(t *testing.T) {
	for name, mediaURL := range map[string]string{
		"relative":           "images/example.png",
		"credentials":        "https://user:secret@example.com/image.png",
		"unsupported scheme": "file:///tmp/image.png",
		"control":            "https://example.com/image.png\nnext",
		"data uri":           "data:image/png;base64,YQ==",
	} {
		t.Run(name, func(t *testing.T) {
			request := validSemanticRequest()
			request.Messages[0].Content = []Content{{Kind: ContentImage, URL: mediaURL}}
			if err := ValidateRequest(request, DefaultPolicy().Limits); err == nil {
				t.Fatalf("unsafe media URL accepted: %q", mediaURL)
			}
		})
	}
	request := validSemanticRequest()
	request.Messages[0].Content = []Content{{Kind: ContentImage, URL: "https://example.com/image.png"}}
	if err := ValidateRequest(request, DefaultPolicy().Limits); err != nil {
		t.Fatalf("valid media URL rejected: %v", err)
	}
}

func TestValidateRequestRejectsInvalidRoleToolAndBoundedFields(t *testing.T) {
	limits := DefaultPolicy().Limits
	for name, mutate := range map[string]func(*Request){
		"oversized model": func(request *Request) { request.Model = strings.Repeat("m", limits.ModelBytes+1) },
		"empty message":   func(request *Request) { request.Messages[0].Content = nil },
		"user tool call": func(request *Request) {
			request.Messages[0].Content = []Content{{Kind: ContentToolCall, ToolCall: &ToolCall{ID: "call", Name: "lookup", Arguments: `{}`}}}
		},
		"non-object arguments": func(request *Request) {
			request.Messages = []Message{
				{Role: RoleAssistant, Content: []Content{{Kind: ContentToolCall, ToolCall: &ToolCall{ID: "call", Name: "lookup", Arguments: `[]`}}}},
			}
		},
		"duplicate tool argument": func(request *Request) {
			request.Messages = []Message{
				{Role: RoleAssistant, Content: []Content{{Kind: ContentToolCall, ToolCall: &ToolCall{ID: "call", Name: "lookup", Arguments: `{"query":1,"query":2}`}}}},
			}
		},
		"unpaired surrogate tool argument": func(request *Request) {
			request.Messages = []Message{
				{Role: RoleAssistant, Content: []Content{{Kind: ContentToolCall, ToolCall: &ToolCall{ID: "call", Name: "lookup", Arguments: `{"query":"\ud800"}`}}}},
			}
		},
		"orphan result": func(request *Request) {
			request.Messages = []Message{{Role: RoleTool, Content: []Content{{Kind: ContentToolResult, ToolResult: &ToolResult{CallID: "missing", Content: []Content{{Kind: ContentText, Text: "result"}}}}}}}
		},
		"result before call": func(request *Request) {
			request.Messages = []Message{
				{Role: RoleTool, Content: []Content{{Kind: ContentToolResult, ToolResult: &ToolResult{CallID: "call", Content: []Content{{Kind: ContentText, Text: "result"}}}}}},
				{Role: RoleAssistant, Content: []Content{{Kind: ContentToolCall, ToolCall: &ToolCall{ID: "call", Name: "lookup", Arguments: `{}`}}}},
			}
		},
		"two results in one tool message": func(request *Request) {
			request.Messages = []Message{
				{Role: RoleAssistant, Content: []Content{
					{Kind: ContentToolCall, ToolCall: &ToolCall{ID: "one", Name: "lookup", Arguments: `{}`}},
					{Kind: ContentToolCall, ToolCall: &ToolCall{ID: "two", Name: "lookup", Arguments: `{}`}},
				}},
				{Role: RoleTool, Content: []Content{
					{Kind: ContentToolResult, ToolResult: &ToolResult{CallID: "one", Content: []Content{{Kind: ContentText, Text: "one"}}}},
					{Kind: ContentToolResult, ToolResult: &ToolResult{CallID: "two", Content: []Content{{Kind: ContentText, Text: "two"}}}},
				}},
			}
		},
		"schema array": func(request *Request) { request.Tools[0].InputSchema = json.RawMessage(`[]`) },
		"duplicate schema field": func(request *Request) {
			request.Tools[0].InputSchema = json.RawMessage(`{"type":"object","type":"array"}`)
		},
		"too many stops": func(request *Request) { request.Sampling.Stop = make([]string, limits.StopSequences+1) },
		"conflicting conversation state": func(request *Request) {
			request.PreviousResponseID, request.ConversationID = "response_1", "conversation_1"
		},
	} {
		t.Run(name, func(t *testing.T) {
			request := validSemanticRequest()
			mutate(&request)
			if err := ValidateRequest(request, limits); err == nil {
				t.Fatalf("invalid request was accepted: %+v", request)
			}
		})
	}
}

func TestValidateJSONObjectBoundsNestedValues(t *testing.T) {
	for name, body := range map[string]string{
		"nested object":  `{"outer":{"inner":{"leaf":1}}}`,
		"nested array":   `{"outer":[[[1]]]}`,
		"trailing value": `{} {}`,
		"invalid UTF-8":  string([]byte{'{', '"', 'x', '"', ':', '"', 0xff, '"', '}'}),
	} {
		t.Run(name, func(t *testing.T) {
			if err := ValidateJSONObject([]byte(body), 2); err == nil {
				t.Fatalf("invalid bounded JSON object was accepted: %q", body)
			}
		})
	}
	if err := ValidateJSONObject([]byte(`{"outer":[1]}`), 2); err != nil {
		t.Fatalf("valid bounded JSON object rejected: %v", err)
	}
}

func TestValidateUsageRequiresExplicitStateAndSafeTotals(t *testing.T) {
	valid := Usage{
		State:         UsageAvailable,
		InputUncached: authoritativeTestCount(2), InputCacheRead: authoritativeTestCount(1),
		InputCacheWrite: authoritativeTestCount(0), OutputReasoning: authoritativeTestCount(1),
		OutputOther: authoritativeTestCount(2), InputTotal: authoritativeTestCount(3),
		OutputTotal: authoritativeTestCount(3), Total: authoritativeTestCount(6),
	}
	if err := ValidateUsage(valid); err != nil {
		t.Fatalf("valid usage rejected: %v", err)
	}
	for name, usage := range map[string]Usage{
		"unknown with value": {State: UsageUnavailable, Total: authoritativeTestCount(1)},
		"available empty":    {State: UsageAvailable},
		"mismatch": {
			State: UsageAvailable, InputTotal: authoritativeTestCount(4),
			InputUncached: authoritativeTestCount(1), InputCacheRead: authoritativeTestCount(1),
		},
		"overflow": {
			State: UsageAvailable, Total: authoritativeTestCount(1),
			InputTotal: authoritativeTestCount(math.MaxInt64), OutputTotal: authoritativeTestCount(1),
		},
		"overflow without declared total": {
			State:      UsageAvailable,
			InputTotal: authoritativeTestCount(math.MaxInt64), OutputTotal: authoritativeTestCount(1),
		},
		"input breakdown overflow without declared total": {
			State:         UsageAvailable,
			InputUncached: authoritativeTestCount(math.MaxInt64), InputCacheRead: authoritativeTestCount(1),
		},
	} {
		t.Run(name, func(t *testing.T) {
			if err := ValidateUsage(usage); err == nil {
				t.Fatalf("invalid usage was accepted: %+v", usage)
			}
		})
	}
}

func TestValidateResponseBoundsErrorIdentityAndMessage(t *testing.T) {
	limits := DefaultPolicy().Limits
	valid := Response{
		Generation: 1,
		StopReason: StopError,
		Error:      NewError(ErrorAuthentication, "authentication_error", "API key is invalid.", nil),
	}
	if err := ValidateResponse(valid, limits); err != nil {
		t.Fatalf("valid error response rejected: %v", err)
	}
	for name, mutate := range map[string]func(*Response){
		"provider request ID": func(response *Response) {
			response.ProviderRequestID = strings.Repeat("r", limits.IdentifierBytes+1)
		},
		"error code": func(response *Response) {
			response.Error.Code = strings.Repeat("c", limits.IdentifierBytes+1)
		},
		"error parameter": func(response *Response) {
			response.Error.Parameter = strings.Repeat("p", limits.IdentifierBytes+1)
		},
		"error message": func(response *Response) {
			response.Error.Message = strings.Repeat("m", limits.TextBytes+1)
		},
	} {
		t.Run(name, func(t *testing.T) {
			response := valid
			copyError := *valid.Error
			response.Error = &copyError
			mutate(&response)
			if err := ValidateResponse(response, limits); err == nil {
				t.Fatalf("oversized error response accepted: %#v", response)
			}
		})
	}
}

func TestValidateProtocolErrorsRequireClosedCategoryAndMessage(t *testing.T) {
	limits := DefaultPolicy().Limits
	valid := NewError(ErrorRateLimited, "rate_limit_error", "Try again later.", nil)
	if err := ValidateTransportError(TransportError{Error: valid}, limits); err != nil {
		t.Fatalf("valid transport error rejected: %v", err)
	}

	tests := []struct {
		name  string
		error *ProtocolError
		code  string
	}{
		{
			name:  "missing category",
			error: NewError("", "provider_error", "request failed", nil),
			code:  "invalid_error_category",
		},
		{
			name:  "unknown category",
			error: NewError(ErrorCategory("future_category"), "provider_error", "request failed", nil),
			code:  "invalid_error_category",
		},
		{
			name:  "missing message",
			error: NewError(ErrorUpstreamUnavailable, "provider_error", "", nil),
			code:  "error_message_required",
		},
		{
			name:  "whitespace message",
			error: NewError(ErrorUpstreamUnavailable, "provider_error", " \t\n", nil),
			code:  "error_message_required",
		},
	}
	for _, test := range tests {
		t.Run(test.name+"/transport", func(t *testing.T) {
			err := ValidateTransportError(TransportError{Error: test.error}, limits)
			protocolError, ok := err.(*ProtocolError)
			if !ok || protocolError.Code != test.code {
				t.Fatalf("transport validation error = %T %v, want code %q", err, err, test.code)
			}
		})
		t.Run(test.name+"/model failure", func(t *testing.T) {
			err := ValidateResponse(Response{
				Generation: 1,
				StopReason: StopError,
				Error:      test.error,
			}, limits)
			protocolError, ok := err.(*ProtocolError)
			if !ok || protocolError.Code != test.code {
				t.Fatalf("response validation error = %T %v, want code %q", err, err, test.code)
			}
		})
	}
}

func TestValidateErrorResponseStillValidatesUsage(t *testing.T) {
	response := Response{
		Generation: 1,
		StopReason: StopError,
		Error:      NewError(ErrorUpstreamUnavailable, "provider_error", "failed", nil),
		Usage: Usage{
			State: UsageAvailable,
			Total: TokenCount{Value: Int64(-1), Provenance: UsageAuthoritative},
		},
	}
	if err := ValidateResponse(response, DefaultPolicy().Limits); err == nil {
		t.Fatal("error response with invalid usage was accepted")
	}
}

func TestValidateResponseRequiresConsistentMatchedStopSequence(t *testing.T) {
	response := Response{
		Generation: 1, ID: "response", StopReason: StopSequence, MatchedStopSequence: "END",
		Output: []OutputItem{{
			ID: "item", Role: RoleAssistant,
			Content: []Content{{Kind: ContentText, Text: "done"}},
		}},
		Usage: Usage{State: UsageUnavailable},
	}
	if err := ValidateResponse(response, DefaultPolicy().Limits); err != nil {
		t.Fatalf("valid matched stop sequence rejected: %v", err)
	}
	response.MatchedStopSequence = " "
	if err := ValidateResponse(response, DefaultPolicy().Limits); err != nil {
		t.Fatalf("whitespace matched stop sequence rejected: %v", err)
	}
	response.MatchedStopSequence = ""
	if err := ValidateResponse(response, DefaultPolicy().Limits); err == nil {
		t.Fatal("stop_sequence response without the matched value was accepted")
	}
	response.StopReason = StopEndTurn
	response.MatchedStopSequence = "END"
	if err := ValidateResponse(response, DefaultPolicy().Limits); err == nil {
		t.Fatal("matched stop sequence with the wrong terminal reason was accepted")
	}
}

func TestStableIDUsesUnambiguousLengths(t *testing.T) {
	if StableID("ab", "c") == StableID("a", "bc") {
		t.Fatal("stable ID framing collided")
	}
	large := strings.Repeat("x", 1<<16)
	if StableID(large, "y") == StableID(large+"y", "") {
		t.Fatal("stable ID length framing truncated a large field")
	}
}

func validSemanticRequest() Request {
	return Request{
		Generation: 1, Model: "model",
		Messages:   []Message{{Role: RoleUser, Content: []Content{{Kind: ContentText, Text: "hello"}}}},
		Tools:      []Tool{{Name: "lookup", InputSchema: json.RawMessage(`{"type":"object"}`)}},
		ToolChoice: ToolChoice{Mode: ToolChoiceAuto},
	}
}

func authoritativeTestCount(value int64) TokenCount {
	return TokenCount{Value: Int64(value), Provenance: UsageAuthoritative}
}
