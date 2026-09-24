package contextdedup

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/contextcompression"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func TestNormalizeOnlyCollapsesWhitespace(t *testing.T) {
	if Normalize("  Hello,   World \n", NormalizationExact) != "  Hello,   World \n" {
		t.Fatal("exact mode changed text")
	}
	if Normalize("  Hello,   World \n", NormalizationWhitespace) != "Hello, World" {
		t.Fatal("whitespace mode did not collapse runs")
	}
	if Normalize("Hello", NormalizationWhitespace) == Normalize("hello", NormalizationWhitespace) {
		t.Fatal("whitespace mode must not fold case")
	}
}

func TestIdentityDistinguishesStructureNotPosition(t *testing.T) {
	base := []contextcompression.MessageView{
		historyMessage(0, 0, "user", "question"),
		historyMessage(1, 0, "assistant", "answer"),
	}
	same := []contextcompression.MessageView{
		historyMessage(10, 10, "user", "question"),
		historyMessage(11, 10, "assistant", "answer"),
	}
	if !identityOf(base, NormalizationExact).equals(identityOf(same, NormalizationExact)) {
		t.Fatal("identity must ignore message ids and turn ids")
	}
	for name, other := range map[string][]contextcompression.MessageView{
		"role": {
			historyMessage(0, 0, "user", "question"),
			historyMessage(1, 0, "user", "answer"),
		},
		"text": {
			historyMessage(0, 0, "user", "question"),
			historyMessage(1, 0, "assistant", "answer."),
		},
		"block_count": {
			withBlocks(historyMessage(0, 0, "user", "question"), "question", ""),
			historyMessage(1, 0, "assistant", "answer"),
		},
		"block_boundary": {
			withBlocks(historyMessage(0, 0, "user", "question"), "quest", "ion"),
			historyMessage(1, 0, "assistant", "answer"),
		},
		"block_source": {
			withBlockSource(historyMessage(0, 0, "user", "question"), contextcompression.TargetMemory),
			historyMessage(1, 0, "assistant", "answer"),
		},
		"message_count": {
			historyMessage(0, 0, "user", "question"),
		},
	} {
		if identityOf(base, NormalizationExact).equals(identityOf(other, NormalizationExact)) {
			t.Fatalf("%s difference must change identity", name)
		}
	}
	spaced := []contextcompression.MessageView{
		historyMessage(0, 0, "user", " question "),
		historyMessage(1, 0, "assistant", "answer"),
	}
	if identityOf(base, NormalizationExact).equals(identityOf(spaced, NormalizationExact)) {
		t.Fatal("exact mode must see the surrounding spaces")
	}
	if !identityOf(base, NormalizationWhitespace).equals(identityOf(spaced, NormalizationWhitespace)) {
		t.Fatal("whitespace mode must ignore the surrounding spaces")
	}
}

func TestEquivalentMessages(t *testing.T) {
	text := func(role llmprotocol.Role, text string) llmprotocol.Message {
		return llmprotocol.Message{Role: role, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: text}}}
	}
	withID := func(message llmprotocol.Message, id string) llmprotocol.Message {
		message.ID = id
		return message
	}
	content := func(role llmprotocol.Role, blocks ...llmprotocol.Content) llmprotocol.Message {
		return llmprotocol.Message{Role: role, Content: blocks}
	}
	errored := true
	for _, tc := range []struct {
		name    string
		earlier llmprotocol.Message
		later   llmprotocol.Message
		mode    Normalization
		ok      bool
		reason  string
	}{
		{"same_text", text("user", "hello"), text("user", "hello"), NormalizationExact, true, ""},
		{"same_ids", withID(text("user", "hello"), "msg_1"), withID(text("user", "hello"), "msg_1"), NormalizationExact, true, ""},
		{"later_without_id", withID(text("user", "hello"), "msg_1"), text("user", "hello"), NormalizationExact, true, ""},
		{"earlier_without_id", text("user", "hello"), withID(text("user", "hello"), "msg_2"), NormalizationExact, false, RetainedIdentityMismatch},
		{"distinct_ids", withID(text("user", "hello"), "msg_1"), withID(text("user", "hello"), "msg_2"), NormalizationExact, false, RetainedIdentityMismatch},
		{"role", text("user", "hello"), text("assistant", "hello"), NormalizationExact, false, RetainedIdentityMismatch},
		{"text", text("user", "hello"), text("user", "hello!"), NormalizationExact, false, RetainedIdentityMismatch},
		{"whitespace_exact", text("user", "hello  there"), text("user", "hello there"), NormalizationExact, false, RetainedIdentityMismatch},
		{"whitespace_mode", text("user", "hello  there"), text("user", "hello there"), NormalizationWhitespace, true, ""},
		{"block_count", content("user", llmprotocol.Content{Kind: llmprotocol.ContentText, Text: "a"}, llmprotocol.Content{Kind: llmprotocol.ContentText, Text: "b"}), text("user", "a"), NormalizationExact, false, RetainedIdentityMismatch},
		{"kind", content("assistant", llmprotocol.Content{Kind: llmprotocol.ContentReasoning, Text: "a"}), text("assistant", "a"), NormalizationExact, false, RetainedIdentityMismatch},
		{"reasoning_same", content("assistant", llmprotocol.Content{Kind: llmprotocol.ContentReasoning, Text: "a", Reasoning: llmprotocol.ReasoningScopeSummary}), content("assistant", llmprotocol.Content{Kind: llmprotocol.ContentReasoning, Text: "a", Reasoning: llmprotocol.ReasoningScopeSummary}), NormalizationExact, true, ""},
		{"reasoning_scope", content("assistant", llmprotocol.Content{Kind: llmprotocol.ContentReasoning, Text: "a", Reasoning: llmprotocol.ReasoningScopeText}), content("assistant", llmprotocol.Content{Kind: llmprotocol.ContentReasoning, Text: "a", Reasoning: llmprotocol.ReasoningScopeSummary}), NormalizationExact, false, RetainedIdentityMismatch},
		{"signature", content("assistant", llmprotocol.Content{Kind: llmprotocol.ContentReasoning, Text: "a", Signature: "sig1"}), content("assistant", llmprotocol.Content{Kind: llmprotocol.ContentReasoning, Text: "a", Signature: "sig2"}), NormalizationExact, false, RetainedIdentityMismatch},
		{"refusal", content("assistant", llmprotocol.Content{Kind: llmprotocol.ContentRefusal, Text: "no"}), content("assistant", llmprotocol.Content{Kind: llmprotocol.ContentRefusal, Text: "no"}), NormalizationExact, false, RetainedRefusal},
		{"citations", content("assistant", llmprotocol.Content{Kind: llmprotocol.ContentText, Text: "a", Citations: []llmprotocol.Citation{{URL: "https://a"}}}), text("assistant", "a"), NormalizationExact, false, RetainedIdentityMismatch},
		{"cache_directive", content("user", llmprotocol.Content{Kind: llmprotocol.ContentText, Text: "a", Cache: &llmprotocol.CacheDirective{Type: "ephemeral"}}), text("user", "a"), NormalizationExact, false, RetainedIdentityMismatch},
		{"cache_directive_same", content("user", llmprotocol.Content{Kind: llmprotocol.ContentText, Text: "a", Cache: &llmprotocol.CacheDirective{Type: "ephemeral"}}), content("user", llmprotocol.Content{Kind: llmprotocol.ContentText, Text: "a", Cache: &llmprotocol.CacheDirective{Type: "ephemeral"}}), NormalizationExact, true, ""},
		{"tool_call", content("assistant", llmprotocol.Content{Kind: llmprotocol.ContentToolCall, ToolCall: &llmprotocol.ToolCall{ID: "c", Name: "f"}}), content("assistant", llmprotocol.Content{Kind: llmprotocol.ContentToolCall, ToolCall: &llmprotocol.ToolCall{ID: "c", Name: "f"}}), NormalizationExact, false, RetainedIdentityMismatch},
		{"tool_result", content("tool", llmprotocol.Content{Kind: llmprotocol.ContentToolResult, ToolResult: &llmprotocol.ToolResult{CallID: "c", IsError: &errored}}), content("tool", llmprotocol.Content{Kind: llmprotocol.ContentToolResult, ToolResult: &llmprotocol.ToolResult{CallID: "c"}}), NormalizationExact, false, RetainedIdentityMismatch},
		{"image", content("user", llmprotocol.Content{Kind: llmprotocol.ContentImage, URL: "https://img"}), content("user", llmprotocol.Content{Kind: llmprotocol.ContentImage, URL: "https://img"}), NormalizationExact, false, RetainedIdentityMismatch},
		{"text_with_media_fields", content("user", llmprotocol.Content{Kind: llmprotocol.ContentText, Text: "a", MediaType: "text/plain"}), text("user", "a"), NormalizationExact, false, RetainedIdentityMismatch},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ok, reason := EquivalentMessages(tc.earlier, tc.later, tc.mode)
			if ok != tc.ok || reason != tc.reason {
				t.Fatalf("got ok=%v reason=%q, want ok=%v reason=%q", ok, reason, tc.ok, tc.reason)
			}
		})
	}
}

func TestRequestResolverFollowsExecutorRemovals(t *testing.T) {
	request := &llmprotocol.Request{Messages: []llmprotocol.Message{
		textMessage(llmprotocol.RoleUser, "old"), textMessage(llmprotocol.RoleAssistant, "old answer"),
		textMessage(llmprotocol.RoleUser, "kept"), textMessage(llmprotocol.RoleAssistant, "kept answer"),
		textMessage(llmprotocol.RoleUser, "live"),
	}}
	ir := contextcompression.ParseSemanticRequest(request, contextcompression.Provenance{})
	resolver := RequestResolver(ir)
	if message, ok := resolver(2); !ok || message.Content[0].Text != "kept" {
		t.Fatalf("unexpected resolution before removal: %+v %v", message, ok)
	}
	ir.Messages = ir.Messages[2:]
	request.Messages = request.Messages[2:]
	if message, ok := resolver(2); !ok || message.Content[0].Text != "kept" {
		t.Fatalf("unexpected resolution after removal: %+v %v", message, ok)
	}
	if _, ok := resolver(0); ok {
		t.Fatal("removed message must not resolve")
	}
	if _, ok := RequestResolver(nil)(0); ok {
		t.Fatal("nil IR must not resolve")
	}
	if _, ok := RequestResolver(&contextcompression.RequestIR{})(0); ok {
		t.Fatal("raw IR without semantic request must not resolve")
	}
}
