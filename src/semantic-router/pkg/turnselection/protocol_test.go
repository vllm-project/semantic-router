package turnselection

import (
	"bytes"
	"context"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/contextcompression"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
)

func TestSelectTurnsAcrossProtocols(t *testing.T) {
	cases := []struct {
		name   string
		format llmprotocol.WireFormat
		body   string
	}{
		{
			name:   "chat_completions",
			format: llmprotocol.OpenAIChatV1,
			body: `{"model":"model","messages":[` +
				`{"role":"developer","content":"Keep permissions"},` +
				`{"role":"user","content":"Database schema: orders(id, customer_id, total)."},` +
				`{"role":"assistant","content":"Recorded orders schema."},` +
				`{"role":"user","content":"What is the weekend weather forecast?"},` +
				`{"role":"assistant","content":"Sunny and warm."},` +
				`{"role":"user","content":"Quick recent check-in."},` +
				`{"role":"assistant","content":"Ready."},` +
				`{"role":"user","content":"Write a SQL query on orders by customer_id."}]}`,
		},
		{
			name:   "responses",
			format: llmprotocol.OpenAIResponsesV1,
			body: `{"model":"model","instructions":"Keep permissions","input":[` +
				`{"role":"user","content":"Database schema: orders(id, customer_id, total)."},` +
				`{"role":"assistant","content":"Recorded orders schema."},` +
				`{"role":"user","content":"What is the weekend weather forecast?"},` +
				`{"role":"assistant","content":"Sunny and warm."},` +
				`{"role":"user","content":"Quick recent check-in."},` +
				`{"role":"assistant","content":"Ready."},` +
				`{"role":"user","content":"Write a SQL query on orders by customer_id."}]}`,
		},
		{
			name:   "anthropic_messages",
			format: llmprotocol.AnthropicMessagesV1,
			body: `{"model":"model","max_tokens":128,"system":"Keep permissions","messages":[` +
				`{"role":"user","content":"Database schema: orders(id, customer_id, total)."},` +
				`{"role":"assistant","content":"Recorded orders schema."},` +
				`{"role":"user","content":"What is the weekend weather forecast?"},` +
				`{"role":"assistant","content":"Sunny and warm."},` +
				`{"role":"user","content":"Quick recent check-in."},` +
				`{"role":"assistant","content":"Ready."},` +
				`{"role":"user","content":"Write a SQL query on orders by customer_id."}]}`,
		},
	}

	policy := DefaultPolicy()
	policy.Budget.MaxPriorTurns = 2
	policy.Budget.MinRecentTurns = 1

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			engine := protocolcodec.NewBuiltinEngine()
			request, envelope, _, err := engine.DecodeRequest(tc.format, []byte(tc.body))
			if err != nil {
				t.Fatalf("decode: %v", err)
			}

			// Disabled selection must leave encoded provider bytes identical.
			baseline, err := engine.EncodeRequest(tc.format, request, envelope)
			if err != nil {
				t.Fatalf("encode baseline: %v", err)
			}
			disabledReq := cloneRequest(t, &request)
			disabledIR := contextcompression.ParseSemanticRequest(disabledReq, contextcompression.Provenance{})
			disabledStep := NewAction(policy, RequestInput{}).Step()
			disabledStep.Enabled = false
			if err = disabledIR.ApplySteps(context.Background(), []contextcompression.TransformationStep{disabledStep}); err != nil {
				t.Fatalf("apply disabled: %v", err)
			}
			unchanged, err := engine.EncodeRequest(tc.format, *disabledReq, envelope)
			if err != nil {
				t.Fatalf("encode unchanged: %v", err)
			}
			if !bytes.Equal(baseline.Body, unchanged.Body) {
				t.Fatal("disabled selection changed provider wire bytes")
			}

			// Enabled selection drops the irrelevant weather turn across all 3 formats.
			ir := contextcompression.ParseSemanticRequest(&request, contextcompression.Provenance{})
			action := NewAction(policy, RequestInput{CostEstimator: NewSemanticCostEstimator(&request, nil)})
			if err = ir.ApplySteps(context.Background(), []contextcompression.TransformationStep{action.Step()}); err != nil {
				t.Fatalf("apply selection: %v", err)
			}
			diag := action.Reconcile(ir.Transformations.Receipts())
			if diag.Outcome != OutcomeApplied || diag.RemovedTurns != 1 || diag.RemovedMessages != 2 {
				t.Fatalf("unexpected diagnostics: %+v", diag)
			}

			encoded, err := engine.EncodeRequest(tc.format, request, envelope)
			if err != nil {
				t.Fatalf("encode selected: %v", err)
			}
			wire := string(encoded.Body)
			if strings.Contains(wire, "weather forecast") {
				t.Fatalf("irrelevant weather turn survived encoding: %s", wire)
			}
			for _, required := range []string{"Keep permissions", "orders(id, customer_id", "Quick recent check-in", "SQL query on orders"} {
				if !strings.Contains(wire, required) {
					t.Fatalf("missing required content %q after encoding: %s", required, wire)
				}
			}
			decoded, _, _, err := engine.DecodeRequest(tc.format, encoded.Body)
			if err != nil {
				t.Fatalf("re-decode: %v", err)
			}
			if len(decoded.Messages) != 5 {
				t.Fatalf("re-decoded message count = %d, want 5", len(decoded.Messages))
			}
			if mustMarshalJSON(t, request.Instructions) != mustMarshalJSON(t, decoded.Instructions) {
				t.Fatal("instructions changed across round trip")
			}
		})
	}
}

func TestSelectTurnsPreservesToolAtomicityAcrossProtocols(t *testing.T) {
	cases := []struct {
		name   string
		format llmprotocol.WireFormat
		body   string
	}{
		{
			name:   "chat_parallel_tools",
			format: llmprotocol.OpenAIChatV1,
			body: `{"model":"model","messages":[` +
				`{"role":"user","content":"Inspect database metrics."},` +
				`{"role":"assistant","content":null,"tool_calls":[` +
				`{"id":"call_1","type":"function","function":{"name":"get_cpu","arguments":"{}"}},` +
				`{"id":"call_2","type":"function","function":{"name":"get_ram","arguments":"{}"}}]},` +
				`{"role":"tool","tool_call_id":"call_1","content":"cpu=40%"},` +
				`{"role":"tool","tool_call_id":"call_2","content":"ram=2GB"},` +
				`{"role":"assistant","content":"Metrics collected."},` +
				`{"role":"user","content":"Unrelated weather question."},` +
				`{"role":"assistant","content":"Cloudy today."},` +
				`{"role":"user","content":"Summarize the database metrics."}]}`,
		},
		{
			name:   "responses_function_calls",
			format: llmprotocol.OpenAIResponsesV1,
			body: `{"model":"model","input":[` +
				`{"role":"user","content":"Inspect database metrics."},` +
				`{"type":"function_call","call_id":"call_1","name":"get_cpu","arguments":"{}"},` +
				`{"type":"function_call","call_id":"call_2","name":"get_ram","arguments":"{}"},` +
				`{"type":"function_call_output","call_id":"call_1","output":"cpu=40%"},` +
				`{"type":"function_call_output","call_id":"call_2","output":"ram=2GB"},` +
				`{"role":"assistant","content":"Metrics collected."},` +
				`{"role":"user","content":"Unrelated weather question."},` +
				`{"role":"assistant","content":"Cloudy today."},` +
				`{"role":"user","content":"Summarize the database metrics."}]}`,
		},
		{
			name:   "anthropic_user_tool_results",
			format: llmprotocol.AnthropicMessagesV1,
			body: `{"model":"model","max_tokens":128,"messages":[` +
				`{"role":"user","content":"Inspect database metrics."},` +
				`{"role":"assistant","content":[` +
				`{"type":"tool_use","id":"call_1","name":"get_cpu","input":{}},` +
				`{"type":"tool_use","id":"call_2","name":"get_ram","input":{}}]},` +
				`{"role":"user","content":[` +
				`{"type":"tool_result","tool_use_id":"call_1","content":"cpu=40%"},` +
				`{"type":"tool_result","tool_use_id":"call_2","content":"ram=2GB"}]},` +
				`{"role":"assistant","content":"Metrics collected."},` +
				`{"role":"user","content":"Unrelated weather question."},` +
				`{"role":"assistant","content":"Cloudy today."},` +
				`{"role":"user","content":"Summarize the database metrics."}]}`,
		},
	}

	policy := DefaultPolicy()
	policy.Budget.MaxPriorTurns = 1
	policy.Budget.MinRecentTurns = 0

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			engine := protocolcodec.NewBuiltinEngine()
			request, envelope, _, err := engine.DecodeRequest(tc.format, []byte(tc.body))
			if err != nil {
				t.Fatalf("decode: %v", err)
			}
			ir := contextcompression.ParseSemanticRequest(&request, contextcompression.Provenance{})
			action := NewAction(policy, RequestInput{CostEstimator: NewSemanticCostEstimator(&request, nil)})
			if err = ir.ApplySteps(context.Background(), []contextcompression.TransformationStep{action.Step()}); err != nil {
				t.Fatalf("apply: %v", err)
			}
			encoded, err := engine.EncodeRequest(tc.format, request, envelope)
			if err != nil {
				t.Fatalf("encode: %v", err)
			}
			wire := string(encoded.Body)
			if strings.Contains(wire, "weather question") {
				t.Fatalf("irrelevant turn survived: %s", wire)
			}
			for _, id := range []string{"call_1", "call_2", "cpu=40%", "ram=2GB"} {
				if !strings.Contains(wire, id) {
					t.Fatalf("tool exchange member %q lost in %s: %s", id, tc.name, wire)
				}
			}
			if _, _, _, err = engine.DecodeRequest(tc.format, encoded.Body); err != nil {
				t.Fatalf("re-decode after tool-atomic selection failed: %v", err)
			}
		})
	}
}

func TestSelectTurnsPreservesMidConversationChatInstructions(t *testing.T) {
	req := &llmprotocol.Request{
		Model: "chat-model",
		Instructions: []llmprotocol.InstructionBlock{
			{Role: llmprotocol.RoleSystem, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "top instruction"}}, Position: 0},
			{Role: llmprotocol.RoleSystem, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "after weather turn"}}, Position: 4},
		},
		Messages: []llmprotocol.Message{
			textMsg(llmprotocol.RoleUser, "Database schema orders(id, amount)."),
			textMsg(llmprotocol.RoleAssistant, "Orders schema noted."),
			textMsg(llmprotocol.RoleUser, "Weather chat about rain."),
			textMsg(llmprotocol.RoleAssistant, "Rain expected."),
			textMsg(llmprotocol.RoleUser, "Query the orders database schema."),
		},
	}

	policy := DefaultPolicy()
	policy.Budget.MaxPriorTurns = 1
	policy.Budget.MinRecentTurns = 0

	ir := contextcompression.ParseSemanticRequest(req, contextcompression.Provenance{})
	action := NewAction(policy, RequestInput{CostEstimator: NewSemanticCostEstimator(req, nil)})
	if err := ir.ApplySteps(context.Background(), []contextcompression.TransformationStep{action.Step()}); err != nil {
		t.Fatalf("apply: %v", err)
	}
	// Removing messages 2,3 leaves messages [0,1,4] -> second instruction shifts from position 4 to 2.
	if len(req.Instructions) != 2 || req.Instructions[0].Position != 0 || req.Instructions[1].Position != 2 {
		t.Fatalf("mid-conversation instruction positions corrupted: %+v", req.Instructions)
	}
}

func TestSelectTurnsPipelineOrderingResetThenSelectThenCompress(t *testing.T) {
	longResult := strings.Repeat("database query execution plan index scan cost analysis ", 120)
	req := &llmprotocol.Request{
		Model: "pipeline-model",
		Messages: []llmprotocol.Message{
			// Turn 0 (IDs 0,1): removed by upstream TransformReset.
			textMsg(llmprotocol.RoleUser, "stale topic before reset"),
			textMsg(llmprotocol.RoleAssistant, "stale answer before reset"),
			// Turn 1 (IDs 2,3,4,5): relevant DB tool turn with verbose output for downstream compression.
			textMsg(llmprotocol.RoleUser, "analyze database query index plan"),
			toolCallMsg(llmprotocol.ToolCall{ID: "sql_1", Name: "explain_sql", Arguments: `{"sql":"SELECT * FROM orders"}`}),
			toolResultMsg(llmprotocol.RoleTool, "sql_1", longResult),
			textMsg(llmprotocol.RoleAssistant, "index scan explanation ready"),
			// Turn 2 (IDs 6,7): unrelated chatter to be removed by TransformSelectTurns.
			textMsg(llmprotocol.RoleUser, "what movie should I watch tonight"),
			textMsg(llmprotocol.RoleAssistant, "watch a comedy"),
			// Turn 3 (ID 8): live turn.
			textMsg(llmprotocol.RoleUser, "optimize the database query index plan"),
		},
	}

	ir := contextcompression.ParseSemanticRequest(req, contextcompression.Provenance{})
	resetStep := contextcompression.TransformationStep{
		Kind:        contextcompression.TransformReset,
		Enabled:     true,
		FailureMode: contextcompression.FailureClosed,
		Propose: func(context.Context, contextcompression.TransformationView) (contextcompression.TransformationEdits, error) {
			return contextcompression.TransformationEdits{RemoveMessages: []int{0, 1}}, nil
		},
	}
	dedupStep := contextcompression.TransformationStep{
		Kind:    contextcompression.TransformDeduplicate,
		Enabled: false,
	}

	selectPolicy := DefaultPolicy()
	selectPolicy.Budget.MaxPriorTurns = 1
	selectPolicy.Budget.MinRecentTurns = 0
	selectAction := NewAction(selectPolicy, RequestInput{CostEstimator: NewSemanticCostEstimator(req, nil)})

	if err := ir.ApplySteps(context.Background(), []contextcompression.TransformationStep{
		resetStep,
		dedupStep,
		selectAction.Step(),
	}); err != nil {
		t.Fatalf("ApplySteps failed: %v", err)
	}

	// Run existing block compression AFTER turn selection.
	compResult := contextcompression.NewService().Apply(context.Background(), contextcompression.Request{
		Model:   req.Model,
		Request: ir,
		Policy: contextcompression.Policy{
			Mode: contextcompression.ModeAlways,
			Targets: contextcompression.Targets{
				ToolOutputs: contextcompression.TargetPolicy{
					Mode:         contextcompression.TargetExtractive,
					MinTokens:    20,
					TargetTokens: 60,
				},
			},
			FailureMode: contextcompression.FailureClosed,
		},
		TokenCounter: contextcompression.HeuristicTokenCounter{},
	})
	if compResult.Failure != nil || !compResult.Applied {
		t.Fatalf("downstream compression failed: %+v", compResult)
	}

	receipts := ir.Transformations.Receipts()
	if len(receipts) != 4 ||
		receipts[0].Kind != contextcompression.TransformReset ||
		receipts[1].Kind != contextcompression.TransformDeduplicate ||
		receipts[2].Kind != contextcompression.TransformSelectTurns ||
		receipts[3].Kind != contextcompression.TransformCompress {
		t.Fatalf("unexpected pipeline receipt order: %+v", receipts)
	}
	diag := selectAction.Reconcile(receipts)
	if diag.Outcome != OutcomeApplied || diag.RemovedTurns != 1 || diag.RemovedMessages != 2 {
		t.Fatalf("unexpected selection diagnostics: %+v", diag)
	}
	texts := survivingTurnTexts(req)
	if containsSubstring(texts, "stale topic") || containsSubstring(texts, "what movie") {
		t.Fatalf("removed turns survived or were resurrected: %v", texts)
	}
	if !containsSubstring(texts, "index scan") || !containsSubstring(texts, "optimize the database") {
		t.Fatalf("retained turns missing after selection + compression: %v", texts)
	}
}
