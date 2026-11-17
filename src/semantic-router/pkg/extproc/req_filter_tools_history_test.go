package extproc

import (
	"encoding/json"
	"slices"
	"testing"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func TestStripSemanticToolPolicyRemovesControlsAndLinkedHistory(t *testing.T) {
	parallel := true
	request := &llmprotocol.Request{
		Generation:        1,
		Tools:             []llmprotocol.Tool{{Name: "lookup", InputSchema: json.RawMessage(`{"type":"object"}`)}},
		ToolChoice:        llmprotocol.ToolChoice{Mode: llmprotocol.ToolChoiceAuto},
		ParallelToolCalls: &parallel,
		Messages: []llmprotocol.Message{
			{Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "Inspect the service."}}},
			{Role: llmprotocol.RoleAssistant, Content: []llmprotocol.Content{
				{Kind: llmprotocol.ContentText, Text: "Keep this explanation."},
				{Kind: llmprotocol.ContentToolCall, ToolCall: &llmprotocol.ToolCall{ID: "call_1", Name: "lookup", Arguments: `{}`}},
			}},
			{Role: llmprotocol.RoleTool, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentToolResult, ToolResult: &llmprotocol.ToolResult{CallID: "call_1"}}}},
			{Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "Summarize safely."}}},
		},
	}

	changed, removed := stripSemanticToolPolicy(request, true)
	if !changed || removed != 2 {
		t.Fatalf("changed=%v removed=%d, want true/2", changed, removed)
	}
	if len(request.Tools) != 0 || request.ToolChoice.Mode != "" || request.ParallelToolCalls != nil {
		t.Fatalf("tool controls remain: %+v", request)
	}
	if len(request.Messages) != 3 || len(request.Messages[1].Content) != 1 ||
		request.Messages[1].Content[0].Text != "Keep this explanation." {
		t.Fatalf("non-tool history was not preserved: %+v", request.Messages)
	}
}

func TestStripSemanticToolPolicyCanKeepHistory(t *testing.T) {
	request := &llmprotocol.Request{
		Generation: 1,
		Tools:      []llmprotocol.Tool{{Name: "lookup", InputSchema: json.RawMessage(`{"type":"object"}`)}},
		Messages: []llmprotocol.Message{{
			Role: llmprotocol.RoleAssistant,
			Content: []llmprotocol.Content{{Kind: llmprotocol.ContentToolCall, ToolCall: &llmprotocol.ToolCall{
				ID: "call_1", Name: "lookup", Arguments: `{}`,
			}}},
		}},
	}
	changed, removed := stripSemanticToolPolicy(request, false)
	if !changed || removed != 0 || len(request.Messages) != 1 {
		t.Fatalf("changed=%v removed=%d messages=%+v", changed, removed, request.Messages)
	}
}

func TestApplyPreDispatchToolsPolicyMutatesNeutralRequestOnce(t *testing.T) {
	payload, err := config.NewStructuredPayload(config.ToolsPluginConfig{
		Enabled: true, Policy: &config.ToolsPolicyConfig{Mode: config.ToolsPolicyModeNone, StripToolHistory: true},
	})
	if err != nil {
		t.Fatal(err)
	}
	request := &llmprotocol.Request{
		Generation: 7,
		Tools:      []llmprotocol.Tool{{Name: "lookup", InputSchema: json.RawMessage(`{"type":"object"}`)}},
		Messages: []llmprotocol.Message{{Role: llmprotocol.RoleTool, Content: []llmprotocol.Content{{
			Kind: llmprotocol.ContentToolResult, ToolResult: &llmprotocol.ToolResult{CallID: "call_1"},
		}}}},
	}
	ctx := &RequestContext{
		SemanticRequest: request,
		VSRSelectedDecision: &config.Decision{Name: "private", Plugins: []config.DecisionPlugin{{
			Type: config.DecisionPluginTools, Configuration: payload,
		}}},
	}
	changed, err := (&OpenAIRouter{}).applyPreDispatchToolsPolicy(ctx)
	if err != nil || !changed {
		t.Fatalf("changed=%v err=%v", changed, err)
	}
	if request.Generation != 8 {
		t.Fatalf("generation=%d, want 8", request.Generation)
	}
	if len(request.Tools) != 0 || len(request.Messages) != 0 {
		t.Fatalf("neutral policy was not applied: %+v", request)
	}
}

// forwardResponsesUnderToolsPolicy routes a Responses request through the
// request headers and body phases of a decision with the given tools policy
// and returns the provider-bound body.
func forwardResponsesUnderToolsPolicy(t *testing.T, policy, body string) map[string]json.RawMessage {
	t.Helper()
	cfg, err := config.ParseYAMLBytes([]byte(`
version: v0.5
providers:
  defaults:
    model: m
  models:
    - name: m
      provider_model_id: m
      api_format: responses
      backend_refs: [{name: b, provider: openai, endpoint: "https://api.example.com/v1"}]
recipes:
  - name: default
    signals:
      keywords:
        - {name: kw, operator: OR, keywords: ["draw"]}
    decisions:
      - name: tools_route
        rules:
          operator: AND
          conditions: [{type: keyword, name: kw}]
        modelRefs: [{model: m}]
        plugins:
          - type: tools
            configuration: {enabled: true, policy: ` + policy + `}
`))
	if err != nil {
		t.Fatal(err)
	}
	router, err := buildOpenAIRouterFromConfig(cfg)
	if err != nil {
		t.Fatal(err)
	}
	router.CredentialResolver = newTestCredentialResolver(cfg)
	ctx := &RequestContext{Headers: map[string]string{}}
	if err := router.processRequestHeaders(NewMockStream(nil), newRequestHeaders("POST", "/v1/responses"), ctx); err != nil {
		t.Fatal(err)
	}
	stream := NewMockStream(nil)
	if err := router.processRequestBody(stream, &ext_proc.ProcessingRequest_RequestBody{
		RequestBody: &ext_proc.HttpBody{Body: []byte(body), EndOfStream: true},
	}, ctx); err != nil || len(stream.Responses) == 0 {
		t.Fatalf("err=%v responses=%d", err, len(stream.Responses))
	}
	forwarded := stream.Responses[len(stream.Responses)-1].GetRequestBody().GetResponse().GetBodyMutation().GetBody()
	var sent map[string]json.RawMessage
	if err := json.Unmarshal(forwarded, &sent); err != nil {
		t.Fatalf("decision=%q forwarded %q: %v", ctx.VSRSelectedDecisionName, forwarded, err)
	}
	return sent
}

// forwardedToolTypeNames names each forwarded tool: a function by its name, a
// hosted tool by its type.
func forwardedToolTypeNames(t *testing.T, sent map[string]json.RawMessage) []string {
	t.Helper()
	var tools []struct {
		Type string `json:"type"`
		Name string `json:"name"`
	}
	if raw, ok := sent["tools"]; ok {
		if err := json.Unmarshal(raw, &tools); err != nil {
			t.Fatal(err)
		}
	}
	names := []string{}
	for _, tool := range tools {
		if tool.Type == "function" {
			names = append(names, tool.Name)
		} else {
			names = append(names, tool.Type)
		}
	}
	return names
}

// The tools policy treats a Responses hosted tool by its type like any other
// tool: none removes it, filtered keeps it only when allowed, and a
// tool_choice that names a removed one goes with it.
func TestToolsPolicyAppliesToHostedTools(t *testing.T) {
	const tools = `"tools":[{"type":"image_generation"},{"type":"function","name":"lookup","parameters":{"type":"object","properties":{}}}]`
	const forced = `,"tool_choice":{"type":"image_generation"}`
	for _, test := range []struct {
		name, policy, choice string
		want                 []string
		keepsChoice          bool
	}{
		{name: "none", policy: `{mode: none}`, want: []string{}},
		{name: "none forced", policy: `{mode: none}`, choice: forced, want: []string{}},
		{name: "filtered allow", policy: `{mode: filtered, allow_tools: [lookup]}`, choice: forced, want: []string{"lookup"}},
		{name: "filtered block", policy: `{mode: filtered, block_tools: [image_generation]}`, choice: forced, want: []string{"lookup"}},
		{name: "filtered allows it", policy: `{mode: filtered, allow_tools: [image_generation]}`, choice: forced, want: []string{"image_generation"}, keepsChoice: true},
	} {
		t.Run(test.name, func(t *testing.T) {
			sent := forwardResponsesUnderToolsPolicy(t, test.policy,
				`{"model":"vllm-sr/auto","store":false,"input":"draw a cat",`+tools+test.choice+`}`)
			if got := forwardedToolTypeNames(t, sent); !slices.Equal(got, test.want) {
				t.Errorf("tools %v reached the provider, want %v", got, test.want)
			}
			if _, ok := sent["tool_choice"]; ok != test.keepsChoice {
				t.Errorf("tool_choice forwarded = %v, want %v: %s", ok, test.keepsChoice, sent["tool_choice"])
			}
		})
	}
}
