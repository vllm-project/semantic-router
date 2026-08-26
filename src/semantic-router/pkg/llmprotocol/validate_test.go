package llmprotocol

import (
	"encoding/json"
	"math"
	"strings"
	"testing"
)

func TestParseCapabilitiesRejectsUnknownNames(t *testing.T) {
	if _, err := ParseCapabilities([]string{"text", "future_product_feature"}); err == nil {
		t.Fatal("unknown capability was silently ignored")
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
	request.Tools = []Tool{{Name: "lookup", Strict: Bool(true), InputSchema: json.RawMessage(`{"type":"object"}`)}}
	required = RequiredCapabilities(request)
	if !required.Supports(CapabilityReasoning) || !required.Supports(CapabilityStrictToolSchema) {
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
		"schema array":   func(request *Request) { request.Tools[0].InputSchema = json.RawMessage(`[]`) },
		"too many stops": func(request *Request) { request.Sampling.Stop = make([]string, limits.StopSequences+1) },
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
