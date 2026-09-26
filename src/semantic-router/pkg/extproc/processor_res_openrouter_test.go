package extproc

import (
	"bytes"
	"encoding/json"
	"os"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func TestSameFormatOpenRouterBufferedReplyRemovesDroppedDecorations(t *testing.T) {
	body, err := os.ReadFile("../protocolcodec/testdata/contracts/openrouter-chat-response-in.json")
	if err != nil {
		t.Fatal(err)
	}
	ctx := &RequestContext{
		SourceFormat: llmprotocol.OpenAIChatV1,
		TargetFormat: llmprotocol.OpenAIChatV1,
		TraceContext: t.Context(),
	}
	response := (&OpenAIRouter{}).handleNonStreamingResponseBody(body, ctx, 0)
	if response.GetImmediateResponse() != nil {
		t.Fatalf("buffered reply failed: %+v", response.GetImmediateResponse())
	}
	mutation := response.GetResponseBody().GetResponse().GetBodyMutation()
	if mutation == nil {
		t.Fatal("same-format reply forwarded decorated provider bytes")
	}
	publicBody := mutation.GetBody()
	for _, field := range []string{
		`"provider":`, `"native_finish_reason":`, `"cost":`, `"is_byok":`,
		`"cost_details":`, `"server_tool_use":`, `"video_tokens":`, `"image_tokens":`,
	} {
		if bytes.Contains(publicBody, []byte(field)) {
			t.Errorf("public reply retained %s: %s", field, publicBody)
		}
	}
	if !bytes.Contains(publicBody, []byte("Hello from OpenRouter.")) ||
		ctx.SemanticResponse == nil || ctx.SemanticResponse.Usage.Total.Value == nil ||
		*ctx.SemanticResponse.Usage.Total.Value != 13 {
		t.Fatalf("content or usage lost: body=%s response=%+v", publicBody, ctx.SemanticResponse)
	}
}

func TestSameFormatOpenRouterStreamRemovesDecorationsAcrossPartialFrames(t *testing.T) {
	fixture, err := os.ReadFile("../protocolcodec/testdata/contracts/openrouter-chat-stream-in.json")
	if err != nil {
		t.Fatal(err)
	}
	var captured struct {
		Chunks []json.RawMessage `json:"chunks"`
	}
	if err := json.Unmarshal(fixture, &captured); err != nil {
		t.Fatal(err)
	}
	var source bytes.Buffer
	for _, chunk := range captured.Chunks {
		var compact bytes.Buffer
		if err := json.Compact(&compact, chunk); err != nil {
			t.Fatal(err)
		}
		source.WriteString("data: ")
		source.Write(compact.Bytes())
		source.WriteString("\n\n")
	}
	source.WriteString("data: [DONE]\n\n")

	for _, includeUsage := range []bool{false, true} {
		t.Run(map[bool]string{false: "usage_not_requested", true: "usage_requested"}[includeUsage], func(t *testing.T) {
			ctx := &RequestContext{
				SourceFormat: llmprotocol.OpenAIChatV1,
				TargetFormat: llmprotocol.OpenAIChatV1,
				RequestModel: "public-model",
				TraceContext: t.Context(),
				SemanticRequest: &llmprotocol.Request{
					Generation: 1, Model: "public-model", Stream: true,
					StreamOptions: llmprotocol.StreamOptions{IncludeUsage: &includeUsage},
				},
			}
			var public bytes.Buffer
			payload := source.Bytes()
			for offset := 0; offset < len(payload); {
				end := min(offset+1+offset%17, len(payload))
				response := (&OpenAIRouter{}).handleSemanticStreamingResponseBody(payload[offset:end], end == len(payload), ctx)
				if response.GetImmediateResponse() != nil || response.GetResponseBody() == nil {
					t.Fatalf("streaming reply failed: %+v", response)
				}
				mutation := response.GetResponseBody().GetResponse().GetBodyMutation()
				if mutation == nil {
					t.Fatalf("raw partial SSE frame was forwarded at offset %d", offset)
				}
				public.Write(mutation.GetBody())
				offset = end
			}
			if ctx.StreamingAborted || !ctx.StreamingComplete {
				t.Fatalf("stream did not complete: aborted=%t complete=%t", ctx.StreamingAborted, ctx.StreamingComplete)
			}
			for _, field := range []string{
				`"provider":`, `"native_finish_reason":`, `"cost":`, `"is_byok":`,
				`"cost_details":`, `"server_tool_use":`, `"video_tokens":`, `"image_tokens":`,
			} {
				if bytes.Contains(public.Bytes(), []byte(field)) {
					t.Errorf("public stream retained %s: %s", field, public.Bytes())
				}
			}
			if !bytes.Contains(public.Bytes(), []byte("Hello from OpenRouter.")) ||
				bytes.Count(public.Bytes(), []byte(`"finish_reason":"stop"`)) != 1 ||
				bytes.Count(public.Bytes(), []byte("data: [DONE]")) != 1 {
				t.Fatalf("stream content or terminal changed: %s", public.Bytes())
			}
			if bytes.Contains(public.Bytes(), []byte(`"usage":`)) != includeUsage {
				t.Fatalf("client usage preference was not respected: %s", public.Bytes())
			}
			if ctx.SemanticResponse == nil || ctx.SemanticResponse.Usage.Total.Value == nil ||
				*ctx.SemanticResponse.Usage.Total.Value != 13 {
				t.Fatalf("internal usage was lost: %+v", ctx.SemanticResponse)
			}
		})
	}
}
