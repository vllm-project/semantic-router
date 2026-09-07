package extproc

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
)

func decodeSignalRequest(t *testing.T, format llmprotocol.WireFormat, body string) *llmprotocol.Request {
	t.Helper()
	request, _, _, err := protocolcodec.NewBuiltinEngine().DecodeRequest(format, []byte(body))
	if err != nil {
		t.Fatalf("DecodeRequest(%s): %v", format, err)
	}
	return &request
}

func assertInputModalityFacts(t *testing.T, got, want classification.InputModalityFacts) {
	t.Helper()
	if got != want {
		t.Fatalf("InputModality = %+v, want %+v", got, want)
	}
}

func TestExtractSemanticRequestSignalsCountsChatInputModalities(t *testing.T) {
	request := decodeSignalRequest(t, llmprotocol.OpenAIChatV1, `{
		"model": "vision-model",
		"messages": [
			{"role": "system", "content": "You are helpful."},
			{"role": "user", "content": [
				{"type": "text", "text": "what is in these?"},
				{"type": "image_url", "image_url": {"url": "https://example.com/a.png"}},
				{"type": "input_audio", "input_audio": {"data": "aGVsbG8=", "format": "wav"}}
			]},
			{"role": "assistant", "content": "An image."},
			{"role": "user", "content": "and now text only"}
		]
	}`)
	snapshot := extractSemanticRequestSignals(request)
	assertInputModalityFacts(t, snapshot.InputModality, classification.InputModalityFacts{
		TextContentCount: 2, ImageContentCount: 1, AudioContentCount: 1,
	})
}

func TestExtractSemanticRequestSignalsTextOnlyChatHasNoMediaCounts(t *testing.T) {
	request := decodeSignalRequest(t, llmprotocol.OpenAIChatV1, `{"model":"m","messages":[{"role":"user","content":"hello"}]}`)
	snapshot := extractSemanticRequestSignals(request)
	assertInputModalityFacts(t, snapshot.InputModality, classification.InputModalityFacts{TextContentCount: 1})
}

// A Responses input_image referenced by file_id has no URL or inline data at
// ingress; the modality fact must still come from the neutral content kind.
func TestExtractSemanticRequestSignalsCountsResponsesFileIDImage(t *testing.T) {
	request := decodeSignalRequest(t, llmprotocol.OpenAIResponsesV1, `{
		"model": "vision-model",
		"input": [{"type": "message", "role": "user", "content": [
			{"type": "input_text", "text": "what is shown here?"},
			{"type": "input_image", "file_id": "file_abc123"}
		]}]
	}`)
	snapshot := extractSemanticRequestSignals(request)
	assertInputModalityFacts(t, snapshot.InputModality, classification.InputModalityFacts{
		TextContentCount: 1, ImageContentCount: 1,
	})
}

func TestExtractSemanticRequestSignalsCountsAnthropicInputModalities(t *testing.T) {
	request := decodeSignalRequest(t, llmprotocol.AnthropicMessagesV1, `{
		"model": "claude",
		"max_tokens": 32,
		"messages": [{"role": "user", "content": [
			{"type": "text", "text": "describe this"},
			{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "aGVsbG8="}}
		]}]
	}`)
	snapshot := extractSemanticRequestSignals(request)
	assertInputModalityFacts(t, snapshot.InputModality, classification.InputModalityFacts{
		TextContentCount: 1, ImageContentCount: 1,
	})
}

// No ingress codec decodes a video part today, but the walker must already
// recognize the neutral video kind so a future codec lights the fact up.
func TestExtractSemanticRequestSignalsCountsNeutralVideoContent(t *testing.T) {
	request := &llmprotocol.Request{Messages: []llmprotocol.Message{{
		Role:    llmprotocol.RoleUser,
		Content: []llmprotocol.Content{{Kind: llmprotocol.ContentVideo, URL: "https://example.com/a.mp4"}},
	}}}
	snapshot := extractSemanticRequestSignals(request)
	assertInputModalityFacts(t, snapshot.InputModality, classification.InputModalityFacts{VideoContentCount: 1})
}

// Input-modality facts are scoped to user turns: media carried by other roles
// contributes to the request-wide image count but not to the family's facts.
func TestExtractSemanticRequestSignalsScopesInputModalitiesToUserMessages(t *testing.T) {
	request := &llmprotocol.Request{Messages: []llmprotocol.Message{
		{Role: llmprotocol.RoleAssistant, Content: []llmprotocol.Content{
			{Kind: llmprotocol.ContentText, Text: "here is the picture"},
			{Kind: llmprotocol.ContentImage, URL: "https://example.com/reply.png"},
		}},
		{Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "thanks"}}},
	}}
	snapshot := extractSemanticRequestSignals(request)
	assertInputModalityFacts(t, snapshot.InputModality, classification.InputModalityFacts{TextContentCount: 1})
	if snapshot.ImageContentCount != 1 {
		t.Fatalf("ImageContentCount = %d, want 1", snapshot.ImageContentCount)
	}
}

// Whitespace-only text is not text input; the classify walk applies the same
// rule so preview and data plane agree on image_input AND NOT text_input.
func TestExtractSemanticRequestSignalsIgnoresWhitespaceOnlyText(t *testing.T) {
	request := decodeSignalRequest(t, llmprotocol.OpenAIChatV1, `{
		"model": "vision-model",
		"messages": [{"role": "user", "content": [
			{"type": "text", "text": "   "},
			{"type": "image_url", "image_url": {"url": "https://example.com/a.png"}}
		]}]
	}`)
	snapshot := extractSemanticRequestSignals(request)
	assertInputModalityFacts(t, snapshot.InputModality, classification.InputModalityFacts{ImageContentCount: 1})
}
