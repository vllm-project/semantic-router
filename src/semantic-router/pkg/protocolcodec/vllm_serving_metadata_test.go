package protocolcodec

import (
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// Replies captured from real servers under testdata/providers, byte for byte
// (the JSON ones end with the newline the repository requires):
//   - vllm-0.33-chat-*: vLLM v0.33.0 serving Qwen/Qwen3-0.6B as "vllm-small"
//     with --reasoning-parser qwen3, --tool-call-parser hermes and
//     --enable-prompt-tokens-details. The return-extensions requests send
//     return_token_ids, return_weight_versions and return_prompt_text; the
//     request-metrics server adds --enable-per-request-metrics,
//     --per-request-spec-decode-metrics summary and n-gram speculative decoding.
//   - vllm-0.33-omni-thinker-chat-*: the same vLLM serving an omni model's
//     thinker; the audio requests ask a spoken question as input_audio.
//   - vllm-omni-0.31-chat-*: vLLM-Omni v0.31.0rc1 with modalities ["text"].
//
// vLLM 0.33 sends choices[].weight_versions on every reply and on the final
// chunk of each choice, and kv_transfer_params on stream chunks; vLLM-Omni
// sends "metrics": {} and "modality": "text" on its chunks.
var vllmReplyCaptures = []string{
	"vllm-0.33-chat-text-out.json",
	"vllm-0.33-chat-reasoning-out.json",
	"vllm-0.33-chat-tool-call-out.json",
	"vllm-0.33-chat-logprobs-out.json",
	"vllm-0.33-chat-stop-string-out.json",
	"vllm-0.33-chat-length-out.json",
	"vllm-0.33-chat-return-extensions-out.json",
	"vllm-0.33-chat-request-metrics-out.json",
	"vllm-0.33-omni-thinker-chat-text-out.json",
	"vllm-0.33-omni-thinker-chat-audio-out.json",
	"vllm-omni-0.31-chat-text-out.json",
}

var vllmStreamCaptures = []string{
	"vllm-0.33-chat-text-stream.sse",
	"vllm-0.33-chat-text-no-usage-stream.sse",
	"vllm-0.33-chat-reasoning-stream.sse",
	"vllm-0.33-chat-tool-call-stream.sse",
	"vllm-0.33-chat-stop-string-stream.sse",
	"vllm-0.33-chat-length-stream.sse",
	"vllm-0.33-chat-return-extensions-stream.sse",
	"vllm-0.33-chat-request-metrics-stream.sse",
	"vllm-0.33-omni-thinker-chat-text-stream.sse",
	"vllm-0.33-omni-thinker-chat-audio-stream.sse",
	"vllm-omni-0.31-chat-text-stream.sse",
}

func vllmCapture(t *testing.T, name string) []byte {
	t.Helper()
	body, err := os.ReadFile(filepath.Join("testdata", "providers", name))
	if err != nil {
		t.Fatal(err)
	}
	return body
}

func vllmPolicies() []llmprotocol.Policy {
	return []llmprotocol.Policy{llmprotocol.DefaultPolicy(), conformancePolicy()}
}

func servingMetadataFields(diagnostics llmprotocol.Diagnostics) []string {
	var fields []string
	for _, diagnostic := range diagnostics {
		if diagnostic.Reason == vllmServingMetadataReason {
			fields = append(fields, diagnostic.Field)
		}
	}
	return fields
}

func hasServingMetadataDiagnostic(diagnostics llmprotocol.Diagnostics, field string) bool {
	for _, got := range servingMetadataFields(diagnostics) {
		if got == field {
			return true
		}
	}
	return false
}

// translateVLLMStream streams a capture from Chat to target and returns the
// client bytes, the diagnostics and the first error.
func translateVLLMStream(
	engine *Engine,
	target llmprotocol.WireFormat,
	body []byte,
) (string, llmprotocol.Diagnostics, error) {
	stream, err := engine.NewStreamWithMutation(llmprotocol.OpenAIChatV1, target, llmprotocol.StreamContext{PublicModel: "m"}, nil)
	if err != nil {
		return "", nil, err
	}
	var out strings.Builder
	frames, _, diagnostics, pushErr := stream.Push(body)
	for _, frame := range frames {
		out.Write(frame)
	}
	frames, _, finalDiagnostics, finalErr := stream.Finalize(pushErr)
	for _, frame := range frames {
		out.Write(frame)
	}
	diagnostics = append(diagnostics, finalDiagnostics...)
	if pushErr != nil {
		return out.String(), diagnostics, pushErr
	}
	return out.String(), diagnostics, finalErr
}

// Every member of these replies is declared, so the conformance policy, which
// rejects undeclared members, decodes them too.
func TestVLLMRepliesDecode(t *testing.T) {
	for _, name := range vllmReplyCaptures {
		t.Run(name, func(t *testing.T) {
			body := vllmCapture(t, name)
			for _, policy := range vllmPolicies() {
				response, _, diagnostics, err := OpenAIChatCodec{}.DecodeResponse(body, policy)
				if err != nil {
					t.Fatalf("DecodeResponse (conformance %t): %v", policy.RejectUnknownUpstreamFields, err)
				}
				if DiagnosticsDroppedUnknownFields(diagnostics) {
					t.Fatalf("undeclared members dropped: %+v", diagnostics)
				}
				if len(response.Output) == 0 {
					t.Fatal("no output decoded")
				}
			}
		})
	}
}

func TestVLLMStreamsTranslateToEveryClient(t *testing.T) {
	targets := []llmprotocol.WireFormat{
		llmprotocol.OpenAIChatV1, llmprotocol.OpenAIResponsesV1, llmprotocol.AnthropicMessagesV1,
	}
	for _, name := range vllmStreamCaptures {
		t.Run(name, func(t *testing.T) {
			body := vllmCapture(t, name)
			for _, policy := range vllmPolicies() {
				engine, err := NewEngine(NewBuiltinRegistry(), policy)
				if err != nil {
					t.Fatal(err)
				}
				for _, target := range targets {
					out, _, err := translateVLLMStream(engine, target, body)
					if err != nil {
						t.Fatalf("to %s (conformance %t): %v\n%s", target, policy.RejectUnknownUpstreamFields, err, out)
					}
				}
			}
		})
	}
}

func TestVLLMOmniThinkerAnswersSpokenQuestion(t *testing.T) {
	result, err := conformanceEngine().TranslateResponse(llmprotocol.OpenAIChatV1, llmprotocol.AnthropicMessagesV1,
		vllmCapture(t, "vllm-0.33-omni-thinker-chat-audio-out.json"), nil)
	if err != nil {
		t.Fatalf("TranslateResponse: %v", err)
	}
	if !strings.Contains(string(result.Body), "Paris") {
		t.Fatalf("answer lost: %s", result.Body)
	}
	out, _, err := translateVLLMStream(conformanceEngine(), llmprotocol.AnthropicMessagesV1,
		vllmCapture(t, "vllm-0.33-omni-thinker-chat-audio-stream.sse"))
	if err != nil || !strings.Contains(out, "Paris") {
		t.Fatalf("streamed answer lost (%v):\n%s", err, out)
	}
}

// Serving metadata that carries a value is reported once as dropped from the
// neutral response; the defaults (null, and vLLM-Omni's empty metrics) report
// nothing.
func TestVLLMServingMetadataDiagnostics(t *testing.T) {
	replies := map[string][]string{
		"vllm-0.33-chat-text-out.json":              nil,
		"vllm-0.33-chat-request-metrics-out.json":   {"metrics"},
		"vllm-0.33-chat-return-extensions-out.json": {"choices.weight_versions", "prompt_text"},
	}
	for name, want := range replies {
		_, _, diagnostics, err := OpenAIChatCodec{}.DecodeResponse(vllmCapture(t, name), llmprotocol.DefaultPolicy())
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		if got := servingMetadataFields(diagnostics); !reflect.DeepEqual(got, want) {
			t.Errorf("%s: serving metadata diagnostics = %v, want %v", name, got, want)
		}
	}
	streams := map[string][]string{
		"vllm-0.33-chat-text-stream.sse":              nil,
		"vllm-0.33-chat-request-metrics-stream.sse":   {"stream.metrics"},
		"vllm-0.33-chat-return-extensions-stream.sse": {"stream.prompt_text", "stream.choices.weight_versions"},
		"vllm-omni-0.31-chat-text-stream.sse":         nil,
	}
	for name, want := range streams {
		_, diagnostics, err := translateVLLMStream(NewBuiltinEngine(), llmprotocol.OpenAIChatV1, vllmCapture(t, name))
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		if got := servingMetadataFields(diagnostics); !reflect.DeepEqual(got, want) {
			t.Errorf("%s: serving metadata diagnostics = %v, want %v", name, got, want)
		}
	}
}

// A vLLM-Omni chunk in any modality but text carries encoded media in
// delta.content, so it is refused instead of being read as answer text.
func TestVLLMOmniNonTextChunkIsRefused(t *testing.T) {
	capture := string(vllmCapture(t, "vllm-omni-0.31-chat-text-stream.sse"))
	for modality, code := range map[string]string{
		"audio": "unsupported_upstream_audio",
		"image": "unsupported_upstream_modality",
	} {
		body := strings.Replace(capture, `"metrics":{},"modality":"text"`, `"metrics":{},"modality":"`+modality+`"`, 1)
		if body == capture {
			t.Fatal("the capture has no text chunk to rewrite")
		}
		_, _, err := translateVLLMStream(NewBuiltinEngine(), llmprotocol.OpenAIChatV1, []byte(body))
		var protocolErr *llmprotocol.ProtocolError
		if !errors.As(err, &protocolErr) || protocolErr.Code != code {
			t.Fatalf("modality %q: error = %v, want %s", modality, err, code)
		}
	}
}
