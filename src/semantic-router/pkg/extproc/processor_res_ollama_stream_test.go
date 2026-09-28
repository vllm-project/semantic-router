package extproc

import (
	"os"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// Ollama sends "content":"" beside every reasoning delta and beside streamed
// tool calls. These frames are captured from Ollama 0.34.3 with qwen3:8b.
func ollamaStreamFrames(t *testing.T, name string) []string {
	t.Helper()
	raw, err := os.ReadFile("../protocolcodec/testdata/providers/" + name)
	if err != nil {
		t.Fatal(err)
	}
	var frames []string
	for _, frame := range strings.SplitAfter(string(raw), "\n\n") {
		if strings.TrimSpace(frame) != "" {
			frames = append(frames, frame)
		}
	}
	return frames
}

// streamOllamaThroughRouter sends each upstream frame through the response
// stream path and returns exactly what the client receives.
func streamOllamaThroughRouter(t *testing.T, client llmprotocol.WireFormat, fixture string, includeUsage bool) string {
	t.Helper()
	router, model := routingTestRouterForFormat(llmprotocol.OpenAIChatV1)
	request := testNeutralRequest(model, "hello")
	request.Stream = true
	if includeUsage {
		request.StreamOptions.IncludeUsage = llmprotocol.Bool(true)
	}
	ctx := routingTestContext(client, request)
	if _, err := router.prepareProviderDispatch(request, model, "", false, ctx); err != nil {
		t.Fatal(err)
	}
	frames := ollamaStreamFrames(t, fixture)
	var received strings.Builder
	for index, frame := range frames {
		response := router.handleSemanticStreamingResponseBody([]byte(frame), index == len(frames)-1, ctx)
		if mutation := response.GetResponseBody().GetResponse().GetBodyMutation(); mutation != nil {
			received.Write(mutation.GetBody())
		} else {
			received.WriteString(frame)
		}
	}
	return received.String()
}

func TestOllamaToolCallStreamReachesChatClientOnce(t *testing.T) {
	received := streamOllamaThroughRouter(t, llmprotocol.OpenAIChatV1, "ollama-chat-tool-call-stream.sse", true)
	if count := strings.Count(received, "data: [DONE]"); count != 1 {
		t.Fatalf("client received %d [DONE] markers:\n%s", count, received)
	}
	if count := strings.Count(received, `"finish_reason":"tool_calls"`); count != 1 {
		t.Fatalf("client received %d tool_calls finish chunks:\n%s", count, received)
	}
}

func TestOllamaReasoningStreamReachesMessagesClient(t *testing.T) {
	received := streamOllamaThroughRouter(t, llmprotocol.AnthropicMessagesV1, "ollama-chat-reasoning-stream.sse", false)
	if strings.Contains(received, "event: error") {
		t.Fatalf("Messages stream failed:\n%s", received)
	}
	thinking := strings.Index(received, `"content_block":{"thinking"`)
	text := strings.Index(received, `"text_delta","text":"OK"`)
	if thinking < 0 || text < thinking {
		t.Fatalf("want a thinking block followed by the OK text:\n%s", received)
	}
	if strings.Contains(received, `"content_block":{"text":"","type":"text"}`+"}\n\nevent: content_block_stop") {
		t.Fatalf("Messages stream opened an empty text block:\n%s", received)
	}
}

func TestOllamaToolCallStreamHasNoEmptyResponsesMessage(t *testing.T) {
	received := streamOllamaThroughRouter(t, llmprotocol.OpenAIResponsesV1, "ollama-chat-tool-call-stream.sse", false)
	if strings.Contains(received, `"type":"message"`) {
		t.Fatalf("Responses stream carried an empty message item before the function call:\n%s", received)
	}
	if !strings.Contains(received, `"type":"function_call"`) || !strings.Contains(received, "response.completed") {
		t.Fatalf("Responses stream lost the function call:\n%s", received)
	}
}
