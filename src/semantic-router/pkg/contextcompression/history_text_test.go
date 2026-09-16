package contextcompression

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func TestHistoryCompressionAcceptsPlainTextBracketHeadings(t *testing.T) {
	for _, heading := range []string{"[Request heading]", "{Request heading}"} {
		for _, representation := range []string{"semantic", "raw"} {
			t.Run(heading+"/"+representation, func(t *testing.T) {
				original := heading + "\nHEADER\n" + strings.Repeat("Archived unrelated text. ", 15000) + "\nEND"
				var ir *RequestIR
				if representation == "semantic" {
					request := &llmprotocol.Request{Messages: []llmprotocol.Message{
						{Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: original}}},
						{Role: llmprotocol.RoleAssistant, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "First answer."}}},
						{Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "Use calculator for 997*991."}}},
					}}
					ir = ParseSemanticRequest(request, Provenance{})
				} else {
					ir = ParseRequestIR(map[string]interface{}{"messages": []interface{}{
						map[string]interface{}{"role": "user", "content": original},
						map[string]interface{}{"role": "assistant", "content": "First answer."},
						map[string]interface{}{"role": "user", "content": "Use calculator for 997*991."},
					}}, Provenance{})
				}
				result := NewService().Apply(context.Background(), Request{Request: ir,
					Policy: Policy{Mode: ModeAlways, Budget: Budget{TargetTokens: 1000}, Targets: Targets{
						History:     TargetPolicy{Mode: TargetExtractive, MinTokens: 2000, TargetTokens: 500},
						CurrentUser: TargetPolicy{Mode: TargetPreserve}, ToolOutputs: TargetPolicy{Mode: TargetPreserve},
					}}})
				got := ir.Messages[0].Blocks[0].Text
				if result.Failure != nil || !result.Applied || got == original || !strings.Contains(got, omissionMarker) || result.TokensAfter > 1000 {
					t.Fatalf("bracket history was not compressed: %+v", result)
				}
				if ir.Messages[1].Blocks[0].Text != "First answer." || ir.Messages[2].Blocks[0].Text != "Use calculator for 997*991." {
					t.Fatal("latest answer or current instruction changed")
				}
				if ir.Semantic != nil && ir.Semantic.Messages[0].Content[0].Text != got {
					t.Fatal("semantic history edit was not committed")
				}
				if ir.Semantic == nil && ir.Raw["messages"].([]interface{})[0].(map[string]interface{})["content"] != got {
					t.Fatal("raw history edit was not committed")
				}
			})
		}
	}
}

func TestHistoryCompressionRetainsJSONAndToolOutputContracts(t *testing.T) {
	long := strings.Repeat("Archived unrelated text. ", 1000)
	for name, document := range map[string]string{
		"object": `{"value":"` + long + `","code":17}`, "array": `["` + long + `",17]`,
		"string": `"` + long + `"`, "number": strings.Repeat("7", 10000), "boolean": "true", "null": "null",
	} {
		t.Run(name, func(t *testing.T) {
			candidate := plannedCandidate{block: &TextBlockIR{Text: document}, plan: TargetPlan{
				Kind: TargetHistory, OriginalTokens: EstimateTokens(document), TargetTokens: 100,
			}}
			got := compressCandidateText("model", HeuristicTokenCounter{}, candidate)
			if !json.Valid([]byte(got.Content)) {
				t.Fatal("history emitted invalid JSON")
			}
			var leaf string
			switch name {
			case "object":
				var object map[string]json.RawMessage
				if err := json.Unmarshal([]byte(got.Content), &object); err != nil || len(object) != 2 || string(object["code"]) != "17" {
					t.Fatalf("JSON object structure changed: %s", got.Content)
				}
				if err := json.Unmarshal(object["value"], &leaf); err != nil {
					t.Fatal(err)
				}
			case "array":
				var array []json.RawMessage
				if err := json.Unmarshal([]byte(got.Content), &array); err != nil || len(array) != 2 || string(array[1]) != "17" {
					t.Fatalf("JSON array structure changed: %s", got.Content)
				}
				if err := json.Unmarshal(array[0], &leaf); err != nil {
					t.Fatal(err)
				}
			case "string":
				if err := json.Unmarshal([]byte(got.Content), &leaf); err != nil {
					t.Fatal(err)
				}
			default:
				if got.Applied || got.Content != document {
					t.Fatal("indivisible JSON scalar changed")
				}
				return
			}
			if !got.Applied || leaf == long || !strings.Contains(leaf, omissionMarker) {
				t.Fatal("valid JSON string-leaf compression was lost")
			}
		})
	}
	malformed := "[Malformed tool JSON " + long
	for _, kind := range []TargetKind{TargetToolOutput, TargetRAG, TargetMemory} {
		candidate := plannedCandidate{block: &TextBlockIR{Text: malformed}, plan: TargetPlan{
			Kind: kind, OriginalTokens: EstimateTokens(malformed), TargetTokens: 100,
		}}
		if got := compressCandidateText("model", HeuristicTokenCounter{}, candidate); got.Applied || got.Content != malformed {
			t.Fatalf("changed conservative malformed JSON protection for %s", kind)
		}
	}
}
