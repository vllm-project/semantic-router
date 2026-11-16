package extproc

import (
	"fmt"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing"
)

// cachedIDsConfig caches the answers of a decision whose model speaks
// Responses, so a hit is a Responses answer the Router makes itself.
const cachedIDsConfig = `
version: v0.5
listeners:
  - name: http-8899
    address: 0.0.0.0
    port: 8899
providers:
  defaults:
    model: responses-model
  models:
    - name: responses-model
      provider_model_id: responses-model-instruct
      api_format: responses
      backend_refs:
        - name: responses-primary
          endpoint: 127.0.0.1:18000
          protocol: http
          provider: vllm
recipes:
  - name: default
    signals:
      keywords:
        - name: cached
          operator: OR
          keywords: ["__cached__"]
    decisions:
      - name: cached
        priority: 10
        rules: {type: keyword, name: cached}
        modelRefs:
          - model: responses-model
            use_reasoning: false
        plugins:
          - type: response_cache
            configuration:
              enabled: true
              mode: exact
              scope: global
entrypoints:
  - model_names: [vllm-sr/auto]
    recipe: default
    default: true
global:
  stores:
    response_cache:
      backend_type: memory
      max_entries: 100
      ttl_seconds: 3600
`

// cachedIDsAnswer is the backend's answer: a reasoning item, a function
// call, a custom tool call as Codex's apply_patch is, and a message, each with
// the backend's own id.
const cachedIDsAnswer = `{"id":"resp_backend","object":"response","created_at":1,"model":"responses-model-instruct","status":"completed","output":[` +
	`{"type":"reasoning","id":"rs_backend","summary":[{"type":"summary_text","text":"List the files first."}]},` +
	`{"type":"function_call","id":"fc_backend","call_id":"call_backend","name":"ls","arguments":"{}","status":"completed"},` +
	`{"type":"custom_tool_call","id":"ctc_backend","call_id":"call_patch","name":"apply_patch","input":"*** Begin Patch\n*** End Patch","status":"completed"},` +
	`{"type":"message","id":"msg_backend","role":"assistant","status":"completed","content":[{"type":"output_text","text":"Listing the files.","annotations":[]}]}],` +
	`"usage":{"input_tokens":5,"input_tokens_details":{"cached_tokens":0},"output_tokens":3,"output_tokens_details":{"reasoning_tokens":0},"total_tokens":8}}`

// A response-cache hit is an answer the Router makes itself, so it names its
// items afresh, each with the prefix the Responses API gives its type, and
// keeps each call's call_id, which the client echoes with its result. A
// streamed hit is encoded from the cached answer, a custom tool call such as
// Codex's apply_patch included.
// A Responses client sends the items back with those ids, and a Responses
// backend receives them unchanged.
func TestACacheHitNamesEachItemAsTheResponsesAPIDoes(t *testing.T) {
	cfg, err := config.ParseYAMLBytes([]byte(cachedIDsConfig))
	if err != nil {
		t.Fatalf("parse config: %v", err)
	}
	engine := protocolcodec.NewBuiltinEngine()
	for mode, engineFor := range looperAdapters {
		for _, stream := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/stream=%t", mode, stream), func(t *testing.T) {
				router := engineFor(newParityRouter(t, cfg))
				prompt := fmt.Sprintf("__cached__ list the files (%s %t)", mode, stream)
				primeCachedIDs(t, router, engine, prompt, stream)
				hit := planCachedIDs(t, router, prompt, stream, "").Immediate
				if hit == nil || hit.Header.Get(headers.VSRResponsePath) != headers.ResponsePathCache {
					t.Fatalf("the repeated request was not answered from the cache: %+v", hit)
				}
				answer := decodeCachedIDs(t, engine, drainResponse(t, hit), stream)
				if !strings.HasPrefix(answer.ID, "resp_") || len(answer.Output) != 4 {
					t.Fatalf("cache hit = %s %+v, want a resp_ answer with four items", answer.ID, answer.Output)
				}
				for index, prefix := range []string{"rs_", "fc_", "ctc_", "msg_"} {
					if id := answer.Output[index].ID; !strings.HasPrefix(id, prefix) || strings.Contains(id, "backend") {
						t.Fatalf("item %d id = %q, want a new id starting with %s", index, id, prefix)
					}
				}
				if call := answer.Output[1].Content[0].ToolCall; call == nil || call.ID != "call_backend" {
					t.Fatalf("function call = %+v, want its call_id kept", call)
				}
				if call := answer.Output[2].Content[0].ToolCall; call == nil || call.ID != "call_patch" || call.Kind != llmprotocol.ToolKindCustom {
					t.Fatalf("custom tool call = %+v, want a custom call with its call_id kept", call)
				}
				followUp := planCachedIDs(t, router, "__cached__ go on", stream, fmt.Sprintf(
					`{"type":"reasoning","id":%q,"summary":[{"type":"summary_text","text":"List the files first."}]},`+
						`{"type":"function_call","id":%q,"call_id":"call_backend","name":"ls","arguments":"{}","status":"completed"},`+
						`{"type":"custom_tool_call","id":%q,"call_id":"call_patch","name":"apply_patch","input":"*** Begin Patch\n*** End Patch","status":"completed"},`+
						`{"type":"message","id":%q,"role":"assistant","status":"completed","content":[{"type":"output_text","text":"Listing the files.","annotations":[]}]},`+
						`{"type":"function_call_output","call_id":"call_backend","output":"a.txt"},`+
						`{"type":"custom_tool_call_output","call_id":"call_patch","output":"Done"},`,
					answer.Output[0].ID, answer.Output[1].ID, answer.Output[2].ID, answer.Output[3].ID))
				if followUp.Call == nil {
					t.Fatalf("the follow-up was answered by the Router: %+v", followUp.Immediate)
				}
				sent := string(followUp.Call.Request.Body)
				for _, id := range []string{answer.Output[0].ID, answer.Output[1].ID, answer.Output[2].ID, answer.Output[3].ID, "call_backend", "call_patch"} {
					if !strings.Contains(sent, fmt.Sprintf("%q", id)) {
						t.Fatalf("the backend receives %s, want %q unchanged", sent, id)
					}
				}
			})
		}
	}
}

// A cache hit's ids are as stable as before: the same request names the same
// items, and a Chat or Messages hit has its protocol's answer id. The
// Responses encoder names each item by its type.
func TestCacheHitIDsAreStableAndOfTheClientsProtocol(t *testing.T) {
	for format, prefix := range map[llmprotocol.WireFormat]string{
		llmprotocol.OpenAIChatV1: "chatcmpl-", llmprotocol.AnthropicMessagesV1: "msg_", llmprotocol.OpenAIResponsesV1: "resp_",
	} {
		ids := func() []string {
			response := &llmprotocol.Response{Output: []llmprotocol.OutputItem{
				{Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "hi"}}},
				{Content: []llmprotocol.Content{{Kind: llmprotocol.ContentToolCall, ToolCall: &llmprotocol.ToolCall{ID: "call_1", Name: "ls"}}}},
			}}
			refreshCachedSemanticResponse(response, &RequestContext{RequestID: "repeat", SourceFormat: format})
			return []string{response.ID, response.Output[0].ID, response.Output[1].ID}
		}
		first, again := ids(), ids()
		if strings.Join(first, " ") != strings.Join(again, " ") || !strings.HasPrefix(first[0], prefix) {
			t.Fatalf("%s: ids %v, then %v for the same request, want the same ids and a %s answer id", format, first, again, prefix)
		}
	}
}

// primeCachedIDs sends prompt once, answered by the backend, so the cache
// keeps the answer.
func primeCachedIDs(t *testing.T, router routing.Engine, engine *protocolcodec.Engine, prompt string, stream bool) {
	t.Helper()
	plan := planCachedIDs(t, router, prompt, stream, "")
	if plan.Call == nil {
		t.Fatalf("the first request was answered by the Router: %+v", plan.Immediate)
	}
	upstream := &routing.UpstreamResponse{Status: 200, Header: routing.Header{{Name: "content-type", Value: "application/json"}}, Body: strings.NewReader(cachedIDsAnswer)}
	if stream {
		decoded, _, _, err := engine.DecodeResponse(llmprotocol.OpenAIResponsesV1, []byte(cachedIDsAnswer))
		if err != nil {
			t.Fatalf("decode the backend's answer: %v", err)
		}
		body, _, err := engine.EncodeResponseStream(llmprotocol.OpenAIResponsesV1, decoded, llmprotocol.StreamContext{Context: t.Context()})
		if err != nil {
			t.Fatalf("encode the backend's stream: %v", err)
		}
		upstream = &routing.UpstreamResponse{Status: 200, Header: routing.Header{{Name: "content-type", Value: "text/event-stream"}}, Body: strings.NewReader(string(body))}
	}
	out, err := router.Respond(t.Context(), plan, upstream)
	if err != nil {
		t.Fatalf("respond: %v", err)
	}
	if body := drainResponse(t, out); out.Status != 200 || !strings.Contains(body, "Listing the files.") {
		t.Fatalf("first answer = %d %s", out.Status, body)
	}
}

// planCachedIDs plans a Responses request for prompt that declares the ls
// tool, with history items before the prompt.
func planCachedIDs(t *testing.T, router routing.Engine, prompt string, stream bool, history string) *routing.Plan {
	t.Helper()
	body := fmt.Sprintf(`{"model":"vllm-sr/auto","stream":%t,"store":false,`+
		`"tools":[{"type":"function","name":"ls","parameters":{"type":"object","properties":{}}},{"type":"custom","name":"apply_patch"}],`+
		`"input":[%s{"role":"user","content":%q}]}`, stream, history, prompt)
	plan, err := router.Plan(t.Context(), globalPluginsRequest("/v1/responses", body))
	if err != nil {
		t.Fatalf("plan: %v", err)
	}
	t.Cleanup(func() { plan.Finish(nil) })
	return plan
}

func decodeCachedIDs(t *testing.T, engine *protocolcodec.Engine, body string, stream bool) llmprotocol.Response {
	t.Helper()
	var answer llmprotocol.Response
	var err error
	if stream {
		answer, _, err = engine.DecodeResponseStream(llmprotocol.OpenAIResponsesV1, []byte(body), llmprotocol.StreamContext{Context: t.Context()})
	} else {
		answer, _, _, err = engine.DecodeResponse(llmprotocol.OpenAIResponsesV1, []byte(body))
	}
	if err != nil {
		t.Fatalf("decode the cache hit %s: %v", body, err)
	}
	return answer
}
