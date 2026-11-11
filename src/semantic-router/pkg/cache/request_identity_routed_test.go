package cache

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func routedRequest(memories, question, runtimeContext string) llmprotocol.Request {
	text := func(role llmprotocol.Role, value string) llmprotocol.Message {
		return llmprotocol.Message{Role: role, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: value}}}
	}
	return llmprotocol.Request{Model: "vllm-sr/auto", Messages: []llmprotocol.Message{
		text(llmprotocol.RoleUser, memories+"\n\n"+question),
		text(llmprotocol.RoleUser, runtimeContext),
	}}
}

func routedIdentity(t *testing.T, memories, question, runtimeContext string) RequestIdentity {
	t.Helper()
	identity, err := BuildRoutedRequestIdentity(routedRequest(memories, question, runtimeContext), RoutedQuery{
		Text: question, MessageIndex: 0, Retained: []string{memories},
	})
	if err != nil {
		t.Fatalf("BuildRoutedRequestIdentity() error = %v", err)
	}
	return identity
}

func TestBuildRoutedRequestIdentityComparesTheRoutingText(t *testing.T) {
	const (
		memories = "<relevant-memories>prefers short answers</relevant-memories>"
		question = "What is on my calendar tomorrow?"
		context  = "Client context: chat-example, msg-1"
	)
	routed := routedIdentity(t, memories, question, context)
	if routed.Query != question || !routed.SemanticSafe {
		t.Fatalf("routed identity = %+v, want the question as a semantic query", routed)
	}
	plain, err := BuildSemanticRequestIdentity(routedRequest(memories, question, context))
	if err != nil {
		t.Fatal(err)
	}
	if plain.Query != context {
		t.Fatalf("without routing_input the query is the last user message, got %q", plain.Query)
	}
	if routed.ExactFingerprint != plain.ExactFingerprint {
		t.Fatal("the exact fingerprint must keep covering the whole request")
	}

	if similar := routedIdentity(t, memories, "What is on my calendar for tomorrow?", context); similar.CompatibilityFingerprint != routed.CompatibilityFingerprint {
		t.Fatal("requests that differ only in the routing text must share semantic entries")
	}
	if other := routedIdentity(t, "<relevant-memories>works nights</relevant-memories>", question, context); other.CompatibilityFingerprint == routed.CompatibilityFingerprint {
		t.Fatal("stripped blocks must stay part of the identity, or recalled context of one user would serve another")
	}
	if other := routedIdentity(t, memories, question, "Client context: chat-other, msg-1"); other.CompatibilityFingerprint == routed.CompatibilityFingerprint {
		t.Fatal("skipped messages must stay part of the identity")
	}
}

func TestBuildRoutedRequestIdentityWithoutRoutingTextSkipsSemanticLookup(t *testing.T) {
	for name, test := range map[string]struct {
		routed      RoutedQuery
		roleAtIndex llmprotocol.Role
	}{
		"every user message skipped": {routed: RoutedQuery{MessageIndex: -1}},
		"no routing text":            {routed: RoutedQuery{MessageIndex: 0}},
		"index out of range":         {routed: RoutedQuery{Text: "question", MessageIndex: 7}},
		"index names another role":   {routed: RoutedQuery{Text: "question", MessageIndex: 0}, roleAtIndex: llmprotocol.RoleAssistant},
	} {
		t.Run(name, func(t *testing.T) {
			request := routedRequest("<m>x</m>", "question", "Client context")
			if test.roleAtIndex != "" {
				request.Messages[test.routed.MessageIndex].Role = test.roleAtIndex
			}
			identity, err := BuildRoutedRequestIdentity(request, test.routed)
			if err != nil {
				t.Fatal(err)
			}
			if identity.SemanticSafe {
				t.Fatalf("identity %+v must not allow a semantic lookup", identity)
			}
		})
	}
}
