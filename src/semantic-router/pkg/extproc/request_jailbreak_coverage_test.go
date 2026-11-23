package extproc

import (
	"context"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing"
)

func TestJailbreakUnreadContentRoutesByPolicy(t *testing.T) {
	const recipe = `    signals:
      jailbreak:
        - name: inspect
          on_unscanned: match
INPUT
    decisions:
      - {name: review, priority: 200, rules: {operator: AND, conditions: [{type: jailbreak, name: inspect}]}, modelRefs: [{model: deep}]}
      - {name: default, priority: 10, rules: {operator: AND, conditions: []}, modelRefs: [{model: fast}]}`
	const png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
	const text = `{"type":"text","text":"Describe the attachment."}`
	const image = `{"type":"image_url","image_url":{"url":"data:image/png;base64,` + png + `"}}`
	chat := func(content string) string {
		return `{"model":"vllm-sr/auto","messages":[{"role":"user","content":[` + content + `]}]}`
	}
	const historyImage = `{"model":"vllm-sr/auto","messages":[{"role":"user","content":[` + image + `]},` +
		`{"role":"assistant","content":"A square."},{"role":"user","content":"Tell me more."}]}`
	const toolImage = `{"model":"vllm-sr/auto","max_tokens":32,"messages":[{"role":"user","content":"Find the attachment."},` +
		`{"role":"assistant","content":[{"type":"tool_use","id":"toolu_1","name":"lookup","input":{}}]},` +
		`{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_1","content":[` + text + `,` +
		`{"type":"image","source":{"type":"base64","media_type":"image/png","data":"` + png + `"}}]},` + text + `]}]}`

	for _, tc := range []struct {
		name, body, input, route string
		anthropic, readText      bool
	}{
		{name: "text only", body: chat(text), route: "fast", readText: true},
		{name: "text and image", body: chat(text + "," + image), route: "deep", readText: true},
		{name: "image only", body: chat(image), route: "deep"},
		{name: "history excluded", body: historyImage, route: "fast", readText: true},
		{name: "history included", body: historyImage, input: "history: true", route: "deep", readText: true},
		{name: "tool result excluded", body: toolImage, anthropic: true, route: "fast", readText: true},
		{name: "tool result included", body: toolImage, input: "tool_results: true", anthropic: true, route: "deep", readText: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			input := ""
			if tc.input != "" {
				input = "          input: {" + tc.input + "}"
			}
			raw := remoteCardYAML(strings.Replace(recipe, "INPUT", input, 1), "")
			path := "/v1/chat/completions"
			if tc.anthropic {
				path = "/v1/messages"
				raw = strings.ReplaceAll(raw, "      backend_refs:", "      api_format: anthropic\n      backend_refs:")
			}
			api := &moduleSystemOneAPI{}
			router := newLooperRouter(t, moduleSystemOneRouterConfig(t, raw, api))
			prepared := len(api.answered())
			ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
			defer cancel()
			header := routing.Header{
				{Name: ":method", Value: "POST"},
				{Name: ":path", Value: path},
				{Name: ":authority", Value: "router"},
				{Name: ":scheme", Value: "http"},
				{Name: "content-type", Value: "application/json"},
				{Name: "anthropic-version", Value: "2023-06-01"},
			}
			plan, err := routing.NewEngine(router, routing.DefaultOptions).Plan(ctx, &routing.Request{Header: header, Body: []byte(tc.body)})
			if err != nil {
				t.Fatal(err)
			}
			defer plan.Finish(nil)
			if plan.Call == nil || plan.Call.Route != tc.route {
				t.Fatalf("got call %+v (immediate %+v), want route %q", plan.Call, plan.Immediate, tc.route)
			}
			if strings.Contains(tc.body, png) && !strings.Contains(string(plan.Call.Request.Body), png) {
				t.Fatal("Guard projection must not remove the attachment from the backend request")
			}
			calls := api.answered()[prepared:]
			if tc.readText && (len(calls) != 1 || !slices.Equal(calls[0], []string{"preset:jailbreak"})) || !tc.readText && len(calls) != 0 {
				t.Fatalf("guard calls = %v, want a text verdict: %v", calls, tc.readText)
			}
		})
	}
}
