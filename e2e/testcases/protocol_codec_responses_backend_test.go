package testcases

import (
	"encoding/json"
	"reflect"
	"testing"
)

func TestProtocolCodecE2EClientMatrixIsClosed(t *testing.T) {
	want := []protocolCodecE2EClient{
		{name: "chat_completions", path: "/v1/chat/completions"},
		{name: "openai_responses", path: "/v1/responses"},
		{name: "anthropic_messages", path: "/v1/messages"},
	}
	if !reflect.DeepEqual(protocolCodecE2EClients, want) {
		t.Fatalf("protocol codec E2E clients = %#v, want %#v", protocolCodecE2EClients, want)
	}

	for _, client := range protocolCodecE2EClients {
		t.Run(client.name, func(t *testing.T) {
			for _, stream := range []bool{false, true} {
				request := client.request("matrix-model", "matrix prompt", stream)
				if request["model"] != "matrix-model" || request["stream"] != stream {
					t.Fatalf("request(stream=%t) = %#v", stream, request)
				}
				if _, err := json.Marshal(request); err != nil {
					t.Fatalf("request(stream=%t) is not JSON encodable: %v", stream, err)
				}
				assertProtocolCodecE2ERequestShape(t, client.path, request)
			}
		})
	}
}

func assertProtocolCodecE2ERequestShape(t *testing.T, path string, request map[string]any) {
	t.Helper()
	_, hasMessages := request["messages"]
	_, hasInput := request["input"]
	_, hasMaxTokens := request["max_tokens"]
	switch path {
	case "/v1/chat/completions":
		if !hasMessages || hasInput || hasMaxTokens {
			t.Fatalf("Chat request uses the wrong protocol shape: %#v", request)
		}
	case "/v1/responses":
		if hasMessages || !hasInput || hasMaxTokens || request["store"] != false {
			t.Fatalf("Responses request uses the wrong protocol shape: %#v", request)
		}
	case "/v1/messages":
		if !hasMessages || hasInput || !hasMaxTokens {
			t.Fatalf("Messages request uses the wrong protocol shape: %#v", request)
		}
	default:
		t.Fatalf("unregistered client path %q", path)
	}
}
