package protocolcodec

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func TestOpenAIChatResponseMetadataRoundTrip(t *testing.T) {
	// Captured from SGLang 0.5.22 serving Qwen/Qwen3-0.6B. In particular,
	// metadata.weight_versions is an array of version/start/end objects.
	sglang, err := os.ReadFile(filepath.Join("testdata", "providers", "sglang-0.5.22-chat-out.json"))
	if err != nil {
		t.Fatal(err)
	}

	for _, test := range []struct {
		name  string
		body  []byte
		text  string
		total int64
	}{
		{
			name:  "sglang",
			body:  sglang,
			text:  "Golden leaves fall,  \nThe air turns crisp,  \nA season of change.",
			total: 36,
		},
		{
			name:  "string_map",
			body:  []byte(`{"id":"chatcmpl_1","object":"chat.completion","created":1,"model":"m","choices":[{"index":0,"message":{"role":"assistant","content":"done"},"finish_reason":"stop"}],"usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3},"metadata":{"trace":"1"}}`),
			text:  "done",
			total: 3,
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			result, err := NewBuiltinEngine().TranslateResponse(
				llmprotocol.OpenAIChatV1, llmprotocol.OpenAIChatV1, test.body, nil,
			)
			if err != nil {
				t.Fatalf("TranslateResponse() error = %v", err)
			}
			output := result.Response.Output
			if len(output) != 1 || len(output[0].Content) != 1 || output[0].Content[0].Text != test.text {
				t.Fatalf("output = %+v, want %q", output, test.text)
			}
			if result.Response.StopReason != llmprotocol.StopEndTurn {
				t.Errorf("stop reason = %s, want end_turn", result.Response.StopReason)
			}
			if total := result.Response.Usage.Total.Value; total == nil || *total != test.total {
				t.Errorf("usage.Total = %+v, want %d", result.Response.Usage.Total, test.total)
			}
			if !hasDiagnostic(result.Diagnostics, "metadata", llmprotocol.DiagnosticDropped) {
				t.Errorf("missing metadata omission: %+v", result.Diagnostics)
			}
			if !bytes.Equal(result.Body, test.body) {
				t.Errorf("same-format replay changed the upstream bytes: %s", result.Body)
			}
		})
	}
}
