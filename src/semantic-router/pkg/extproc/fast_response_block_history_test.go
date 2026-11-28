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

const (
	blockHistoryMessage = "Personal data may not leave this machine."
	blockHistoryValue   = "ada@example.com"
	blockHistoryPrompt  = "File the ticket for " + blockHistoryValue + "."
	blockHistoryNext    = "Reply with the single word after."
)

// blockHistoryConfig refuses a request whose email rule matches and sends
// every other request to model-a. The first verb adds fields to the email
// rule, the second plugins to the decision that dispatches.
const blockHistoryConfig = `
version: v0.5
listeners:
  - name: http-8899
    address: 0.0.0.0
    port: 8899
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
        - name: email
          operator: OR
          method: regex
          keywords: ['[a-z]+@example\.com']
%s
    decisions:
      - name: block_email
        priority: 100
        rules: {type: keyword, name: email}
        plugins:
          - type: fast_response
            configuration:
              message: ` + blockHistoryMessage + `
              block: true
      - name: answer
        priority: 1
        rules: {operator: AND, conditions: []}
        modelRefs:
          - model: model-a
            use_reasoning: false
%s
entrypoints:
  - model_names: [vllm-sr/auto]
    recipe: default
    default: true
`

// A streamed block ends the turn with an answer, so the harness's next
// request carries the blocked prompt and that answer as history, and a rule
// that reads only the newest user turn lets the next prompt through. What the
// model then receives is up to the configuration: the blocked prompt as sent,
// its personal data masked, or nothing, when the rule reads earlier turns and
// refuses the follow-up as well.
func TestAFollowUpCarriesTheBlockedPromptAsHistory(t *testing.T) {
	for _, tc := range []struct {
		name, rule, plugins string
		refused, masked     bool
	}{
		{name: "forwarded"},
		{name: "masked", plugins: `        plugins:
          - type: masking
            configuration:
              enabled: true
              rules: [{type: keyword, name: email}]`, masked: true},
		{name: "refused", rule: `          include_history: true`, refused: true},
	} {
		cfg, err := config.ParseYAMLBytes([]byte(fmt.Sprintf(blockHistoryConfig, tc.rule, tc.plugins)))
		if err != nil {
			t.Fatalf("parse config: %v", err)
		}
		for mode, engineFor := range looperAdapters {
			engine := engineFor(newParityRouter(t, cfg))
			for _, format := range guardProtocols {
				t.Run(fmt.Sprintf("%s/%s/%s", tc.name, mode, format), func(t *testing.T) {
					answer := blockedAnswerText(t, format, planBlockHistoryTurn(t, engine, format, blockHistoryPrompt, ""))
					plan := planBlockHistoryTurn(t, engine, format, blockHistoryPrompt, answer)
					if tc.refused {
						if blockedAnswerText(t, format, plan) != blockHistoryMessage {
							t.Fatal("the follow-up was not refused")
						}
						return
					}
					if plan.Call == nil {
						t.Fatalf("the follow-up was answered by the Router: %s", plan.Immediate.Body)
					}
					sent := string(plan.Call.Request.Body)
					if !strings.Contains(sent, blockHistoryNext) || !strings.Contains(sent, blockHistoryMessage) {
						t.Fatalf("the model receives %s, want the follow-up after the block's answer", sent)
					}
					if strings.Contains(sent, blockHistoryValue) == tc.masked || strings.Contains(sent, "[EMAIL_0]") != tc.masked {
						t.Fatalf("the model receives %s, want the blocked prompt's address masked = %v", sent, tc.masked)
					}
				})
			}
		}
	}
}

// planBlockHistoryTurn plans a streamed turn of format: prompt alone, or,
// after answer, prompt and answer as history before the follow-up.
func planBlockHistoryTurn(t *testing.T, engine routing.Engine, format llmprotocol.WireFormat, prompt, answer string) *routing.Plan {
	t.Helper()
	turns := fmt.Sprintf(`{"role":"user","content":%q}`, prompt)
	if answer != "" {
		turns += fmt.Sprintf(`,{"role":"assistant","content":%q},{"role":"user","content":%q}`, answer, blockHistoryNext)
	}
	path, body := "/v1/chat/completions", `{"model":"vllm-sr/auto","stream":true,"messages":[`+turns+`]}`
	request := globalPluginsRequest(path, body)
	switch format {
	case llmprotocol.OpenAIResponsesV1:
		request = globalPluginsRequest("/v1/responses", `{"model":"vllm-sr/auto","stream":true,"store":false,"input":[`+turns+`]}`)
	case llmprotocol.AnthropicMessagesV1:
		request = globalPluginsRequest("/v1/messages", `{"model":"vllm-sr/auto","max_tokens":64,"stream":true,"messages":[`+turns+`]}`)
		request.Header = append(request.Header, routing.HeaderField{Name: "anthropic-version", Value: "2023-06-01"})
	}
	plan, err := engine.Plan(t.Context(), request)
	if err != nil {
		t.Fatalf("plan: %v", err)
	}
	t.Cleanup(func() { plan.Finish(nil) })
	return plan
}

// blockedAnswerText is the text of the answer a block streamed to the
// client, which the harness keeps as the turn's answer.
func blockedAnswerText(t *testing.T, format llmprotocol.WireFormat, plan *routing.Plan) string {
	t.Helper()
	out := plan.Immediate
	if out == nil || out.Status != 200 || out.Header.Get(headers.VSRResponsePath) != headers.ResponsePathBlocked {
		t.Fatalf("reply = %+v, want the block's answer as a 200 stream on the blocked path", out)
	}
	answer, _, err := protocolcodec.NewBuiltinEngine().DecodeResponseStream(format, []byte(drainResponse(t, out)), llmprotocol.StreamContext{Context: t.Context()})
	if err != nil {
		t.Fatalf("decode the block's answer: %v", err)
	}
	return semanticAssistantContent(&answer)
}

// A Responses client sends the id of the block's answer back with its next
// prompt, and a Responses backend receives it unchanged. The ChatGPT backend
// refuses an input message whose id does not start with msg_, which would
// fail every later prompt of the thread.
func TestTheBlockAnswersIDReachesAResponsesBackendAsAMessageID(t *testing.T) {
	cfg, err := config.ParseYAMLBytes([]byte(strings.Replace(fmt.Sprintf(blockHistoryConfig, "", ""), "api_format: openai", "api_format: responses", 1)))
	if err != nil {
		t.Fatalf("parse config: %v", err)
	}
	for mode, engineFor := range looperAdapters {
		t.Run(mode, func(t *testing.T) {
			engine := engineFor(newParityRouter(t, cfg))
			format := llmprotocol.OpenAIResponsesV1
			out := planBlockHistoryTurn(t, engine, format, blockHistoryPrompt, "").Immediate
			blockedAnswerText(t, format, &routing.Plan{Immediate: out})
			answer, _, err := protocolcodec.NewBuiltinEngine().DecodeResponseStream(format, out.Body, llmprotocol.StreamContext{Context: t.Context()})
			if err != nil || len(answer.Output) != 1 {
				t.Fatalf("decode the block's answer: %v %+v", err, answer.Output)
			}
			id := answer.Output[0].ID
			input := fmt.Sprintf(`[{"role":"user","content":%q},`+
				`{"type":"message","id":%q,"role":"assistant","status":"completed","content":[{"type":"output_text","text":%q,"annotations":[]}]},`+
				`{"role":"user","content":%q}]`, blockHistoryPrompt, id, blockHistoryMessage, blockHistoryNext)
			plan, err := engine.Plan(t.Context(), globalPluginsRequest("/v1/responses",
				`{"model":"vllm-sr/auto","stream":true,"store":false,"input":`+input+`}`))
			if err != nil {
				t.Fatalf("plan: %v", err)
			}
			t.Cleanup(func() { plan.Finish(nil) })
			if plan.Call == nil {
				t.Fatalf("the follow-up was answered by the Router: %s", plan.Immediate.Body)
			}
			if sent := string(plan.Call.Request.Body); !strings.HasPrefix(id, "msg_") || !strings.Contains(sent, fmt.Sprintf(`"id":%q`, id)) {
				t.Fatalf("the backend receives %s, want the answer's message id %q, which starts with msg_", sent, id)
			}
		})
	}
}

// Every answer the Router makes itself, streamed or not, carries the ids its
// client's protocol gives a model's answer, a stored Responses object's
// public id included.
func TestARouterAnswerHasTheIDsOfItsProtocol(t *testing.T) {
	engine := protocolcodec.NewBuiltinEngine()
	stored := &ResponseObjectState{GeneratedResponseID: "resp_public"}
	for _, format := range guardProtocols {
		for _, streaming := range []bool{false, true} {
			for _, object := range []*ResponseObjectState{nil, stored} {
				if object != nil && format != llmprotocol.OpenAIResponsesV1 {
					continue
				}
				t.Run(fmt.Sprintf("%s/stream=%t/stored=%t", format, streaming, object != nil), func(t *testing.T) {
					ctx := &RequestContext{
						RequestID: "router-answer", RequestModel: "vllm-sr/auto", TraceContext: t.Context(),
						SourceFormat: format, ResponseObjectState: object,
					}
					body, _, _, err := (&OpenAIRouter{}).encodeSyntheticTextResponse(ctx, "fast answer", streaming)
					if err != nil {
						t.Fatalf("encode: %v", err)
					}
					var answer llmprotocol.Response
					if streaming {
						answer, _, err = engine.DecodeResponseStream(format, body, llmprotocol.StreamContext{Context: t.Context()})
					} else {
						answer, _, _, err = engine.DecodeResponse(format, body)
					}
					if err != nil {
						t.Fatalf("decode %s: %v", body, err)
					}
					assertAnswerIDs(t, format, answer)
					if object != nil && answer.ID != "resp_public" {
						t.Fatalf("answer id = %q, want the stored object's public id", answer.ID)
					}
				})
			}
		}
	}
}
