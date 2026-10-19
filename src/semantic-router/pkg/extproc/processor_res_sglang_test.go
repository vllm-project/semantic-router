package extproc

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func TestSGLangRecordedResponseBoundaries(t *testing.T) {
	for _, client := range []llmprotocol.WireFormat{llmprotocol.OpenAIChatV1, llmprotocol.AnthropicMessagesV1} {
		for _, name := range []string{"out.json", "stop-out.json", "stream.sse", "stop-stream.sse"} {
			t.Run(string(client)+"/"+name, func(t *testing.T) {
				body, err := os.ReadFile(filepath.Join("..", "protocolcodec", "testdata", "providers", "sglang-0.5.22-chat-"+name))
				if err != nil {
					t.Fatal(err)
				}
				ctx := &RequestContext{
					SourceFormat: client, TargetFormat: llmprotocol.OpenAIChatV1,
					RequestModel: "public-model", TraceContext: t.Context(),
				}
				var wire []byte
				if strings.HasSuffix(name, ".json") {
					response := (&OpenAIRouter{}).handleNonStreamingResponseBody(body, ctx, 0)
					if response.GetImmediateResponse() != nil || response.GetResponseBody() == nil {
						t.Fatalf("successful SGLang response became an error: %+v", response)
					}
					wire = body
					if mutation := response.GetResponseBody().GetResponse().GetBodyMutation(); mutation != nil {
						wire = mutation.GetBody()
					}
				} else {
					if err := (&OpenAIRouter{}).ensureSemanticResponseStream(ctx); err != nil {
						t.Fatal(err)
					}
					var rendered bytes.Buffer
					for _, chunk := range bytes.SplitAfter(body, []byte("\n\n")) {
						frames, events, diagnostics, pushErr := ctx.ProtocolResponseStream.Push(chunk)
						if pushErr != nil {
							t.Fatal(pushErr)
						}
						observeExtProcStream(ctx, frames, events, diagnostics, &rendered)
					}
					frames, events, diagnostics, finalErr := ctx.ProtocolResponseStream.Finalize(nil)
					if finalErr != nil {
						t.Fatal(finalErr)
					}
					observeExtProcStream(ctx, frames, events, diagnostics, &rendered)
					semantic, responseErr := ctx.SemanticStreamState.response()
					if responseErr != nil || semantic == nil || len(semantic.Output) == 0 {
						t.Fatalf("semantic stream did not complete with content: %+v %v", semantic, responseErr)
					}
					wire = rendered.Bytes()
					if client == llmprotocol.AnthropicMessagesV1 && strings.Count(string(wire), "event: message_stop\n") != 1 {
						t.Fatalf("missing successful client completion: %s", wire)
					}
				}
				content, terminal := "Golden", `"finish_reason":"stop"`
				if strings.HasPrefix(name, "stop-") {
					content = "Count"
				}
				if client == llmprotocol.AnthropicMessagesV1 {
					terminal = `"stop_reason":"end_turn","stop_sequence":null`
					if strings.HasPrefix(name, "stop-") {
						terminal = `"stop_reason":"stop_sequence","stop_sequence":"7"`
					}
				}
				if !bytes.Contains(wire, []byte(content)) || !bytes.Contains(wire, []byte(terminal)) {
					t.Fatalf("client lost content or terminal %s: %s", terminal, wire)
				}
			})
		}
	}
}
