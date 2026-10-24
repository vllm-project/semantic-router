package protocolcodec

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// A vLLM-style response: non-null prompt_logprobs plus an unknown vendor field.
const chatPassthroughResponseBody = `{"id":"chatcmpl-1","object":"chat.completion","created":1700000000,` +
	`"model":"vllm-model","choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],` +
	`"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2},` +
	`"prompt_logprobs":{"1":-0.5},"vendor_extra":{"trace":"t1"}}`

func TestChatResponsePassthroughReplaysUnchangedBody(t *testing.T) {
	engine := newChatPassthroughEngine(t)
	result, err := engine.TranslateResponse(llmprotocol.OpenAIChatV1, llmprotocol.OpenAIChatV1, []byte(chatPassthroughResponseBody), nil)
	if err != nil {
		t.Fatalf("TranslateResponse: %v", err)
	}
	if string(result.Body) != chatPassthroughResponseBody {
		t.Fatalf("unchanged response must replay verbatim, got:\n%s", result.Body)
	}
}

// A response-id rewrite forces a re-render; preserved top-level fields must
// be merged back into the encoded body.
func TestChatResponsePassthroughMergesOnRerender(t *testing.T) {
	engine := newChatPassthroughEngine(t)
	mutated := false
	result, err := engine.TranslateResponse(llmprotocol.OpenAIChatV1, llmprotocol.OpenAIChatV1, []byte(chatPassthroughResponseBody), func(response *llmprotocol.Response) error {
		mutated = true
		response.ID = "public-id"
		return nil
	})
	if err != nil {
		t.Fatalf("TranslateResponse: %v", err)
	}
	if !mutated {
		t.Fatal("mutation did not run")
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(result.Body, &fields); err != nil {
		t.Fatal(err)
	}
	if string(fields["prompt_logprobs"]) != `{"1":-0.5}` {
		t.Fatalf("prompt_logprobs not forwarded on re-render:\n%s", result.Body)
	}
	if string(fields["vendor_extra"]) != `{"trace":"t1"}` {
		t.Fatalf("vendor_extra not forwarded on re-render:\n%s", result.Body)
	}
	if string(fields["id"]) != `"public-id"` {
		t.Fatalf("encoder-produced id must win:\n%s", result.Body)
	}
}

// Strict regression: the closed provider contract still rejects non-null
// execution fields and unknown response fields.
func TestChatResponseStrictRejectsProviderExtensions(t *testing.T) {
	engine, err := NewEngine(NewBuiltinRegistry(), llmprotocol.DefaultPolicy())
	if err != nil {
		t.Fatal(err)
	}
	if _, err := engine.TranslateResponse(llmprotocol.OpenAIChatV1, llmprotocol.OpenAIChatV1, []byte(chatPassthroughResponseBody), nil); err == nil {
		t.Fatal("strict policy must reject prompt_logprobs/vendor_extra responses")
	}
}

// A case variant of a modelled response field must stay rejected even under
// the preserve policy: encoding/json's case-insensitive fallback would fill
// the modelled field while the capture also keeps the variant, splitting
// one field into two on a re-render.
func TestChatResponsePassthroughRejectsCaseVariant(t *testing.T) {
	body := `{"id":"chatcmpl-1","object":"chat.completion","created":1700000000,` +
		`"model":"m","ID":"duplicate-id","choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}]}`
	engine := newChatPassthroughEngine(t)
	if _, err := engine.TranslateResponse(llmprotocol.OpenAIChatV1, llmprotocol.OpenAIChatV1, []byte(body), nil); err == nil {
		t.Fatal("case variant of a modelled response field must be rejected under preserve")
	}
}

const chatPassthroughStreamBody = `data: {"id":"1","object":"chat.completion.chunk","created":1700000000,"model":"vllm-model",` +
	`"choices":[{"index":0,"delta":{"role":"assistant","content":"hi"},` +
	`"logprobs":{"content":[{"token":"hi","logprob":-0.5,"top_logprobs":[{"token":"hi","logprob":-0.5}]}]},"finish_reason":null}]}` + "\n\n" +
	`data: {"id":"1","object":"chat.completion.chunk","created":1700000000,"model":"vllm-model",` +
	`"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}` + "\n\n" +
	`data: [DONE]` + "\n\n"

func translateChatStream(t *testing.T, engine *Engine, body string) (string, error) {
	t.Helper()
	stream, err := engine.NewStreamWithMutation(llmprotocol.OpenAIChatV1, llmprotocol.OpenAIChatV1, llmprotocol.StreamContext{
		PublicModel: "public-model",
	}, nil)
	if err != nil {
		t.Fatalf("NewStreamWithMutation: %v", err)
	}
	var out strings.Builder
	frames, _, _, pushErr := stream.Push([]byte(body))
	for _, frame := range frames {
		out.Write(frame)
	}
	frames, _, _, finalErr := stream.Finalize(pushErr)
	for _, frame := range frames {
		out.Write(frame)
	}
	if pushErr != nil {
		return out.String(), pushErr
	}
	return out.String(), finalErr
}

// Streamed logprobs chunks are forwarded as opaque provider frames under the
// preserve policy: the neutral stream contract has no evidence slot, so the
// raw chunk reaches the client unchanged. The output must stay parseable
// SSE: every frame re-parses with the frame parser.
func TestChatStreamPassthroughForwardsLogprobsChunk(t *testing.T) {
	engine := newChatPassthroughEngine(t)
	out, err := translateChatStream(t, engine, chatPassthroughStreamBody)
	if err != nil {
		t.Fatalf("stream failed: %v\n%s", err, out)
	}
	if !strings.Contains(out, `"logprobs":{"content":[{"token":"hi","logprob":-0.5`) {
		t.Fatalf("logprobs chunk was not forwarded verbatim:\n%s", out)
	}
	if !strings.Contains(out, "data: [DONE]") {
		t.Fatalf("terminal sentinel missing:\n%s", out)
	}
	assertStreamOutputReframes(t, out)
}

// assertStreamOutputReframes re-parses an SSE output body into complete
// data frames; a forwarded raw chunk missing its `data: ` prefix or
// terminator would fail or get glued to the next frame.
func assertStreamOutputReframes(t *testing.T, out string) {
	t.Helper()
	remaining := out
	for strings.HasPrefix(remaining, "data: ") {
		end := strings.Index(remaining, "\n\n")
		if end < 0 {
			t.Fatalf("output ends with an unterminated SSE frame:\n%s", out)
		}
		if _, err := parseSSEFrame([]byte(remaining[:end+2]), llmprotocol.DefaultPolicy().Limits.SSEFrameBytes); err != nil {
			t.Fatalf("output frame does not re-parse as SSE: %v\n%s", err, out)
		}
		remaining = remaining[end+2:]
	}
	if strings.TrimSpace(remaining) != "" {
		t.Fatalf("output contains non-frame residue %q:\n%s", remaining, out)
	}
}

// A gateway that decorates every chunk with an unknown field produces an
// all-opaque stream; it must still complete normally with an accurate stop
// reason extracted from the forwarded finish frame.
func TestChatStreamPassthroughAllOpaqueStreamCompletes(t *testing.T) {
	body := `data: {"id":"1","object":"chat.completion.chunk","created":1700000000,"model":"vllm-model",` +
		`"vendor_meta":{"seq":1},"choices":[{"index":0,"delta":{"role":"assistant","content":"hi"},"finish_reason":null}]}` + "\n\n" +
		`data: {"id":"1","object":"chat.completion.chunk","created":1700000000,"model":"vllm-model",` +
		`"vendor_meta":{"seq":2},"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}` + "\n\n" +
		`data: [DONE]` + "\n\n"
	engine := newChatPassthroughEngine(t)
	out, err := translateChatStream(t, engine, body)
	if err != nil {
		t.Fatalf("all-opaque stream must complete, got: %v\n%s", err, out)
	}
	assertStreamOutputReframes(t, out)
	if !strings.Contains(out, `"vendor_meta":{"seq":1}`) || !strings.Contains(out, `"vendor_meta":{"seq":2}`) {
		t.Fatalf("opaque frames not forwarded verbatim:\n%s", out)
	}
	if !strings.Contains(out, `"finish_reason":"stop"`) {
		t.Fatalf("forwarded finish frame missing:\n%s", out)
	}
	if !strings.Contains(out, "data: [DONE]") {
		t.Fatalf("terminal sentinel missing:\n%s", out)
	}
}

// Strict regression: the same stream fails closed.
func TestChatStreamStrictRejectsLogprobsChunk(t *testing.T) {
	engine, err := NewEngine(NewBuiltinRegistry(), llmprotocol.DefaultPolicy())
	if err != nil {
		t.Fatal(err)
	}
	if _, streamErr := translateChatStream(t, engine, chatPassthroughStreamBody); streamErr == nil {
		t.Fatal("strict policy must reject streamed logprobs")
	}
}

// Chunks without passthrough fields keep the ordinary re-encode path, so the
// public model rewrite still applies to them.
func TestChatStreamPassthroughKeepsPublicModelRewrite(t *testing.T) {
	engine := newChatPassthroughEngine(t)
	out, err := translateChatStream(t, engine, chatPassthroughStreamBody)
	if err != nil {
		t.Fatalf("stream failed: %v\n%s", err, out)
	}
	if strings.Contains(out, `"model":"vllm-model","choices":[{"index":0,"delta":{}}`) {
		t.Fatalf("plain chunk must be re-encoded with the public model:\n%s", out)
	}
	if !strings.Contains(out, `"model":"public-model"`) {
		t.Fatalf("public model missing from re-encoded chunks:\n%s", out)
	}
}

// Streamed output audio is forwarded verbatim under the preserve policy and
// still fails closed under the strict policy.
func TestChatStreamAudioPassthroughAndStrictRejection(t *testing.T) {
	const audioStream = `data: {"id":"1","object":"chat.completion.chunk","created":1700000000,"model":"vllm-model",` +
		`"choices":[{"index":0,"delta":{"audio":{"id":"audio-1","data":"QklUVw==","expires_at":1700003600,"transcript":"hi"}},"finish_reason":null}]}` + "\n\n" +
		`data: {"id":"1","object":"chat.completion.chunk","created":1700000000,"model":"vllm-model",` +
		`"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}` + "\n\n" +
		`data: [DONE]` + "\n\n"
	preserve := newChatPassthroughEngine(t)
	out, err := translateChatStream(t, preserve, audioStream)
	if err != nil {
		t.Fatalf("preserve stream failed: %v\n%s", err, out)
	}
	if !strings.Contains(out, `"audio":{"id":"audio-1"`) {
		t.Fatalf("audio delta not forwarded verbatim:\n%s", out)
	}
	assertStreamOutputReframes(t, out)

	strict, engineErr := NewEngine(NewBuiltinRegistry(), llmprotocol.DefaultPolicy())
	if engineErr != nil {
		t.Fatal(engineErr)
	}
	if _, streamErr := translateChatStream(t, strict, audioStream); streamErr == nil {
		t.Fatal("strict policy must reject streamed audio")
	}
}
