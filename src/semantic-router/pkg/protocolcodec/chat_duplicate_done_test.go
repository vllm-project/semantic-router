package protocolcodec

import (
	"context"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// A gateway that re-splits frames can repeat the sentinel. The second one is
// not data after the terminal: the response is already complete and must stay
// complete instead of turning into a server_error frame.
func TestChatDuplicateDoneKeepsStreamComplete(t *testing.T) {
	decoder := OpenAIChatCodec{}.NewDecoder(llmprotocol.StreamContext{Context: context.Background()}, llmprotocol.DefaultPolicy())
	frames := []string{
		`{"id":"chatcmpl-dup","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"role":"assistant","content":"hi"},"finish_reason":null}]}`,
		`{"id":"chatcmpl-dup","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
	}
	for _, frame := range frames {
		if _, _, err := decoder.Push([]byte("data: " + frame + "\n\n")); err != nil {
			t.Fatalf("push: %v", err)
		}
	}

	completed := 0
	for i := 0; i < 2; i++ {
		events, _, err := decoder.Push([]byte("data: [DONE]\n\n"))
		if err != nil {
			t.Fatalf("duplicate [DONE] must not fail the stream (push %d): %v", i, err)
		}
		for _, event := range events {
			if event.Type == llmprotocol.EventResponseCompleted {
				completed++
			}
		}
	}
	if completed != 1 {
		t.Fatalf("completed events = %d, want 1", completed)
	}

	// Tolerance stops at the sentinel: a real chunk after it is still an error.
	if _, _, err := decoder.Push([]byte(`data: {"id":"chatcmpl-dup","choices":[{"index":0,"delta":{"content":"late"}}]}` + "\n\n")); err == nil {
		t.Fatal("a content frame after the terminal must still fail")
	}
}
