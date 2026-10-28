package contextcompression

import (
	"context"
	"strings"
	"testing"
	"unicode/utf8"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

type byteCounter struct{}

func (byteCounter) CountText(_ string, text string) (int, string) { return len(text), "bytes" }
func (byteCounter) CountRequest(_ string, request *RequestIR) (int, string) {
	count := 0
	for _, message := range request.Messages {
		for _, block := range message.Blocks {
			count += len(block.Text)
		}
	}
	return count, "bytes"
}

func TestCurrentUserTruncationRequiresOptInAndPreservesBoundaries(t *testing.T) {
	for _, text := range []string{strings.Repeat(" a", 40000), strings.Repeat("字", 40000), strings.Repeat("🙂", 40000), strings.Repeat("x", 160000)} {
		for _, mode := range []TargetMode{TargetPreserve, TargetTruncate} {
			original := "HEAD-keep-this\n" + text + "\nTAIL-follow-this"
			request := &llmprotocol.Request{Messages: []llmprotocol.Message{{Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: original}}}}}
			result := NewService().Apply(context.Background(), Request{Request: ParseSemanticRequest(request, Provenance{}), TokenCounter: byteCounter{}, Capabilities: ModelContextCapabilities{ContextWindow: 32000}, Policy: Policy{Mode: ModeAlways, Budget: Budget{TargetTokens: 32000}, Targets: Targets{CurrentUser: TargetPolicy{Mode: mode}, History: TargetPolicy{Mode: TargetPreserve}}}})
			got := request.Messages[0].Content[0].Text
			if mode == TargetPreserve {
				if got != original || result.Applied {
					t.Fatal("default changed current user")
				}
				continue
			}
			if result.Failure != nil || !result.Applied || len(got) > 32000 || !utf8.ValidString(got) || !strings.HasPrefix(got, "HEAD-keep-this\n") || !strings.HasSuffix(got, "\nTAIL-follow-this") || !strings.Contains(got, omissionMarker) {
				t.Fatalf("unsafe/nonreducing truncation: %+v bytes=%d", result, len(got))
			}
		}
	}
}

func TestCurrentUserTruncationAcceptsPlainTextBracketHeadings(t *testing.T) {
	for _, heading := range []string{"[Request heading]", "{Request heading}"} {
		for _, representation := range []string{"semantic", "raw"} {
			t.Run(heading+"/"+representation, func(t *testing.T) {
				original := heading + "\nHEAD instruction\n" + strings.Repeat("archive text ", 30000) + "\nTAIL instruction"
				var ir *RequestIR
				if representation == "semantic" {
					request := &llmprotocol.Request{Messages: []llmprotocol.Message{{Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: original}}}}}
					ir = ParseSemanticRequest(request, Provenance{})
				} else {
					request := map[string]interface{}{"messages": []interface{}{map[string]interface{}{"role": "user", "content": original}}}
					ir = ParseRequestIR(request, Provenance{})
				}
				if !ir.Messages[0].Blocks[0].JSON {
					t.Fatal("fixture must exercise the broad JSON-prefix heuristic")
				}
				result := NewService().Apply(context.Background(), Request{Request: ir, TokenCounter: byteCounter{}, Capabilities: ModelContextCapabilities{ContextWindow: 32000}, Policy: Policy{Mode: ModeAlways, Budget: Budget{TargetTokens: 32000}, Targets: Targets{CurrentUser: TargetPolicy{Mode: TargetTruncate}, History: TargetPolicy{Mode: TargetPreserve}}}})
				got := ir.Messages[0].Blocks[0].Text
				if result.Failure != nil || !result.Applied || len(got) > 32000 || !strings.HasPrefix(got, heading+"\nHEAD instruction\n") || !strings.HasSuffix(got, "\nTAIL instruction") || !strings.Contains(got, omissionMarker) {
					t.Fatalf("plain heading blocked truncation: result=%+v bytes=%d", result, len(got))
				}
				if ir.Semantic != nil && ir.Semantic.Messages[0].Content[0].Text != got {
					t.Fatal("semantic request did not receive the committed edit")
				}
				if ir.Semantic == nil && ir.Raw["messages"].([]interface{})[0].(map[string]interface{})["content"] != got {
					t.Fatal("raw request did not receive the committed edit")
				}
			})
		}
	}
}

func TestCurrentUserTruncationProtectsStructuredAndTrustedContent(t *testing.T) {
	long := strings.Repeat("data ", 10000)
	for _, name := range []string{"system", "developer", "json", "json_array", "json_string", "json_number", "json_boolean", "json_null", "tool", "image", "citation", "authorization", "safety"} {
		t.Run(name, func(t *testing.T) {
			request := &llmprotocol.Request{Messages: []llmprotocol.Message{{Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: long}}}}}
			provenance := Provenance{}
			switch name {
			case "system":
				request.Messages[0].Role = llmprotocol.RoleSystem
			case "developer":
				request.Messages[0].Role = llmprotocol.RoleDeveloper
			case "json":
				request.Messages[0].Content[0].Text = `{"data":"` + long + `"}`
			case "json_array":
				request.Messages[0].Content[0].Text = `["` + long + `"]`
			case "json_string":
				request.Messages[0].Content[0].Text = `"` + long + `"`
			case "json_number":
				request.Messages[0].Content[0].Text = strings.Repeat("7", 10000)
			case "json_boolean":
				request.Messages[0].Content[0].Text = `true`
			case "json_null":
				request.Messages[0].Content[0].Text = `null`
			case "tool":
				request.Messages[0].Content = append(request.Messages[0].Content, llmprotocol.Content{Kind: llmprotocol.ContentToolResult, ToolResult: &llmprotocol.ToolResult{CallID: "call_1"}})
			case "image":
				request.Messages[0].Content = append(request.Messages[0].Content, llmprotocol.Content{Kind: llmprotocol.ContentImage, URL: "https://example.invalid/image"})
			case "citation":
				request.Messages[0].Content[0].Citations = []llmprotocol.Citation{{StartIndex: 0, EndIndex: 4}}
			case "authorization":
				provenance.ProtectedMessages = map[int]Protection{0: ProtectAuthorization}
			case "safety":
				provenance.ProtectedMessages = map[int]Protection{0: ProtectSafety}
			}
			ir := ParseSemanticRequest(request, provenance)
			if strings.HasPrefix(name, "json") && ir.currentUserTextBlock(ir.Messages[0], ir.Messages[0].Blocks[0]) {
				t.Fatal("complete JSON document is eligible for current-user truncation")
			}
			before := encoded(t, request)
			result := NewService().Apply(context.Background(), Request{Request: ir, TokenCounter: byteCounter{}, Capabilities: ModelContextCapabilities{ContextWindow: 1000}, Policy: Policy{Mode: ModeAlways, Targets: Targets{CurrentUser: TargetPolicy{Mode: TargetTruncate}, History: TargetPolicy{Mode: TargetPreserve}, ToolOutputs: TargetPolicy{Mode: TargetPreserve}}}})
			if result.Applied || encoded(t, request) != before {
				t.Fatalf("changed protected %s: %+v", name, result)
			}
		})
	}
}
