package extproc

import (
	"fmt"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing"
)

// globalPluginsConfig sets every plugin in global.plugins and none on its one
// decision. The guard blocks an answer to a __refuse__ prompt, and on_unknown
// any answer whose rule it cannot read.
const globalPluginsConfig = `
version: v0.5
listeners:
  - name: http-8899
    address: 0.0.0.0
    port: 8899
    timeout: 300s
providers:
  defaults:
    model: model-a
  models:
    - name: model-a
      provider_model_id: model-a-instruct
      api_format: openai
      backend_refs:
        - name: a-primary
          endpoint: 127.0.0.1:18000
          protocol: http
          provider: vllm
    - name: model-b
      provider_model_id: model-b-instruct
      api_format: openai
      backend_refs:
        - name: b-primary
          endpoint: 127.0.0.1:18001
          protocol: http
          provider: vllm
recipes:
  - name: default
    signals:
      keywords:
        - name: routed
          operator: OR
          keywords: ["__routed__"]
        - name: refuse_answer
          operator: OR
          keywords: ["__refuse__"]
    decisions:
      - name: routed
        priority: 10
        rules:
          operator: AND
          conditions:
            - type: keyword
              name: routed
        modelRefs:
          - model: model-a
            use_reasoning: false
entrypoints:
  - model_names: [vllm-sr/auto]
    recipe: default
    default: true
global:
  plugins:
    - type: system_prompt
      configuration:
        system_prompt: "GLOBAL_PROMPT_MARK"
    - type: header_mutation
      configuration:
        add:
          - name: x-global-plugin
            value: "on"
    - type: request_params
      configuration:
        max_tokens_limit: 7
    - type: response_guard
      configuration:
        on_unknown: block
        rules:
          - when: {type: keyword, name: refuse_answer}
            action: block
    - type: context
      configuration:
        enabled: true
        compress:
          match: {recall: {method: bm25}}
          targets:
            tool_outputs: {mode: extractive, min_tokens: 200, target_tokens: 160}
`

// global.plugins are the plugins of a request that matches no decision and of
// a request for a concrete model, as they are of a decision that inherits
// them: each one changes what reaches the model. The guard reads its rule on
// a recipe's request, and a concrete model, which bypasses the recipe's
// signals, runs no guard.
func TestGlobalPluginsRunWithoutAMatchedDecision(t *testing.T) {
	cfg, err := config.ParseYAMLBytes([]byte(globalPluginsConfig))
	if err != nil {
		t.Fatalf("parse config: %v", err)
	}
	toolOutput := strings.Join([]string{
		"source header", strings.Repeat("irrelevant inventory values ", 180),
		"authentication token validator failed", strings.Repeat("irrelevant billing values ", 180), "source footer",
	}, "\n")
	for mode, engineFor := range looperAdapters {
		engine := engineFor(newParityRouter(t, cfg))
		for _, tc := range []struct {
			name, model, prompt string
			refused             bool
		}{
			{"decision", "vllm-sr/auto", "__routed__ __refuse__ fix the authentication validator", true},
			{"no decision", "vllm-sr/auto", "__refuse__ fix the authentication validator", true},
			{"no decision passed", "vllm-sr/auto", "fix the authentication validator", false},
			{"concrete model", "model-b", "__refuse__ fix the authentication validator", false},
		} {
			for _, stream := range []bool{false, true} {
				t.Run(fmt.Sprintf("%s/%s/stream=%t", mode, tc.name, stream), func(t *testing.T) {
					body := fmt.Sprintf(`{"model":%q,"max_tokens":500,"stream":%t,"messages":[`+
						`{"role":"user","content":"collect diagnostic data"},`+
						`{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"diagnostics","arguments":"{}"}}]},`+
						`{"role":"tool","tool_call_id":"call_1","content":%q},`+
						`{"role":"user","content":%q}]}`, tc.model, stream, toolOutput, tc.prompt)
					plan, err := engine.Plan(t.Context(), globalPluginsRequest("/v1/chat/completions", body))
					if err != nil {
						t.Fatalf("plan: %v", err)
					}
					t.Cleanup(func() { plan.Finish(nil) })
					if plan.Call == nil {
						t.Fatalf("the Router answered the request itself: %+v", plan.Immediate)
					}
					sent := string(plan.Call.Request.Body)
					for plugin, ok := range map[string]bool{
						"system_prompt":   strings.Contains(sent, "GLOBAL_PROMPT_MARK"),
						"request_params":  strings.Contains(sent, `"max_completion_tokens":7`),
						"header_mutation": plan.Call.Request.Header.Get("x-global-plugin") == "on",
						"context compress": !strings.Contains(sent, strings.Repeat("irrelevant inventory values ", 180)) &&
							strings.Contains(sent, "authentication token validator"),
					} {
						if !ok {
							t.Errorf("%s did not run: %.200s", plugin, sent)
						}
					}
					out, err := engine.Respond(t.Context(), plan, chatUpstreamAnswer(benignAnswer, stream))
					if err != nil {
						t.Fatalf("respond: %v", err)
					}
					if tc.refused {
						assertGuardRefused(t, llmprotocol.OpenAIChatV1, stream, out)
					} else {
						assertGuardDelivered(t, out, benignAnswer)
					}
				})
			}
		}
	}
}

// globalPluginsRequest is a request for body with its own request ID.
func globalPluginsRequest(path, body string) *routing.Request {
	return &routing.Request{Header: routing.Header{
		{Name: ":method", Value: "POST"},
		{Name: ":path", Value: path},
		{Name: ":authority", Value: "router"},
		{Name: ":scheme", Value: "http"},
		{Name: "content-type", Value: "application/json"},
		{Name: "x-request-id", Value: fmt.Sprintf("global-plugins-%d", guardCacheRequestIDs.Add(1))},
	}, Body: []byte(body)}
}

// inheritedPluginConfig sets plugin with configuration in global.plugins,
// and the recipe's one decision turns it off with enabled: false.
func inheritedPluginConfig(plugin, configuration string) string {
	return fmt.Sprintf(`
version: v0.5
providers:
  defaults:
    model: model-a
  models:
    - name: model-a
      provider_model_id: model-a-instruct
      api_format: openai
      backend_refs:
        - name: a-primary
          endpoint: 127.0.0.1:18000
          protocol: http
          provider: vllm
recipes:
  - name: default
    signals:
      keywords:
        - name: routed
          operator: OR
          keywords: ["__routed__"]
        - name: refuse_answer
          operator: OR
          keywords: ["__refuse__"]
    decisions:
      - name: routed
        priority: 10
        rules:
          operator: AND
          conditions:
            - type: keyword
              name: routed
        modelRefs:
          - model: model-a
            use_reasoning: false
        plugins:
          - type: %[1]s
            configuration:
              enabled: false
entrypoints:
  - model_names: [vllm-sr/auto]
    recipe: default
    default: true
global:
  plugins:
    - type: %[1]s
      configuration: %[2]s
`, plugin, configuration)
}

// A decision that turns the inherited fast_response off reaches its model,
// while a request that matches no decision still gets the fast response.
func TestADecisionTurnsOffAnInheritedFastResponse(t *testing.T) {
	cfg, err := config.ParseYAMLBytes([]byte(inheritedPluginConfig(config.DecisionPluginFastResponse, `{message: MAINTENANCE}`)))
	if err != nil {
		t.Fatalf("parse config: %v", err)
	}
	for mode, engineFor := range looperAdapters {
		engine := engineFor(newParityRouter(t, cfg))
		for _, tc := range []struct {
			prompt string
			fast   bool
		}{{"__routed__ hello", false}, {"hello", true}} {
			body := fmt.Sprintf(`{"model":"vllm-sr/auto","messages":[{"role":"user","content":%q}]}`, tc.prompt)
			plan, err := engine.Plan(t.Context(), globalPluginsRequest("/v1/chat/completions", body))
			if err != nil {
				t.Fatalf("%s %q: plan: %v", mode, tc.prompt, err)
			}
			answered := plan.Immediate != nil && strings.Contains(drainResponse(t, plan.Immediate), "MAINTENANCE")
			plan.Finish(nil)
			if answered != tc.fast {
				t.Fatalf("%s %q: fast response = %t, want %t", mode, tc.prompt, answered, tc.fast)
			}
		}
	}
}

// A decision that turns the inherited response_guard off delivers the answer
// the guard blocks for a request that matches no decision.
func TestADecisionTurnsOffAnInheritedResponseGuard(t *testing.T) {
	cfg, err := config.ParseYAMLBytes([]byte(inheritedPluginConfig(config.DecisionPluginResponseGuard,
		`{rules: [{when: {type: keyword, name: refuse_answer}, action: block}]}`)))
	if err != nil {
		t.Fatalf("parse config: %v", err)
	}
	for mode, engineFor := range looperAdapters {
		engine := engineFor(newParityRouter(t, cfg))
		for _, tc := range []struct {
			name, prompt string
			refused      bool
		}{{"decision", "__routed__ __refuse__ hello", false}, {"no decision", "__refuse__ hello", true}} {
			for _, stream := range []bool{false, true} {
				t.Run(fmt.Sprintf("%s/%s/stream=%t", mode, tc.name, stream), func(t *testing.T) {
					body := fmt.Sprintf(`{"model":"vllm-sr/auto","stream":%t,"messages":[{"role":"user","content":%q}]}`, stream, tc.prompt)
					plan, err := engine.Plan(t.Context(), globalPluginsRequest("/v1/chat/completions", body))
					if err != nil {
						t.Fatalf("plan: %v", err)
					}
					t.Cleanup(func() { plan.Finish(nil) })
					if plan.Call == nil {
						t.Fatalf("the Router answered the request itself: %+v", plan.Immediate)
					}
					out, err := engine.Respond(t.Context(), plan, chatUpstreamAnswer(benignAnswer, stream))
					if err != nil {
						t.Fatalf("respond: %v", err)
					}
					if tc.refused {
						assertGuardRefused(t, llmprotocol.OpenAIChatV1, stream, out)
					} else {
						assertGuardDelivered(t, out, benignAnswer)
					}
				})
			}
		}
	}
}
