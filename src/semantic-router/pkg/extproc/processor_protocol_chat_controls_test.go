package extproc

import (
	"encoding/json"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func TestChatDispatchPreservesInferenceControlsAndCacheSalt(t *testing.T) {
	router := &OpenAIRouter{}
	ctx := &RequestContext{SourceFormat: llmprotocol.OpenAIChatV1, TargetFormat: llmprotocol.OpenAIChatV1}
	body := []byte(`{"model":"auto","messages":[{"role":"user","content":"same prompt"}],"stream":true,"top_k":-1,"min_p":0,"repetition_penalty":1,"cache_salt":"evaluation-phase","reasoning_effort":"high","chat_template_kwargs":{"enable_thinking":true}}`)
	request, response := router.prepareProtocolRequest(body, ctx)
	if response != nil || request == nil {
		t.Fatalf("strict ingress rejected supported controls: %+v", response)
	}
	request.Model = "backend-model"
	request.Generation++
	request.Messages[0].Content[0].Text = "compressed prompt"
	encoded, err := router.encodeDispatchRequest(ctx)
	if err != nil {
		t.Fatal(err)
	}
	var fields map[string]json.RawMessage
	if err = json.Unmarshal(encoded, &fields); err != nil {
		t.Fatal(err)
	}
	for key, want := range map[string]string{"model": `"backend-model"`, "top_k": "-1", "min_p": "0", "repetition_penalty": "1", "cache_salt": `"evaluation-phase"`, "reasoning_effort": `"high"`, "chat_template_kwargs": `{"enable_thinking":true}`, "stream_options": `{"include_usage":true}`} {
		if string(fields[key]) != want {
			t.Fatalf("%s lost during dispatch: %s", key, encoded)
		}
	}
	if string(fields["messages"]) != `[{"role":"user","content":"compressed prompt"}]` {
		t.Fatalf("cache metadata leaked into prompt or mutation was lost: %s", encoded)
	}
	if request.StreamOptions.IncludeUsage != nil {
		t.Fatal("backend accounting changed the client stream preference")
	}
}
