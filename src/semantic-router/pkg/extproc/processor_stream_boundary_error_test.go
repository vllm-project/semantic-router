package extproc

import (
	"bytes"
	"context"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func TestStreamBoundaryRejectsDynamoPrefixOnDecodeError(t *testing.T) {
	for _, tc := range []struct{ name, frame, code string }{
		{"nvext", "data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"model\":\"model-a\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"forbidden-prefix\"},\"finish_reason\":null}],\"nvext\":{\"token_ids\":[1]}}\n\n", "unexpected_dynamo_nvext_backend"},
		{"request_id", "event: request_id\n: \"forbidden-prefix\"\n\n", "unexpected_dynamo_request_id_backend"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			router := dynamoBoundaryTestRouter("vllm")
			ctx := &RequestContext{
				SourceFormat: llmprotocol.OpenAIChatV1, TargetFormat: llmprotocol.OpenAIChatV1,
				RequestModel: "model-a", UpstreamBackendType: "vllm",
				TraceContext: context.Background(), SemanticRequest: &llmprotocol.Request{},
			}
			body := []byte(tc.frame + "data: {malformed\n\n")
			// Verify the actual codec returns a valid prefix together with an error.
			probe := *ctx
			if err := router.ensureSemanticResponseStream(&probe); err != nil {
				t.Fatal(err)
			}
			frames, events, _, err := probe.ProtocolResponseStream.Push(body)
			if err == nil || len(frames) == 0 || len(events) == 0 {
				t.Fatalf("fixture must produce prefix and decode error: frames=%d events=%d err=%v", len(frames), len(events), err)
			}
			if err := validateDynamoResponseEvents(ctx, events); err == nil || !strings.Contains(err.Error(), tc.code) {
				t.Fatalf("prefix lacks expected Dynamo event: %v", err)
			}
			if err := router.ensureSemanticResponseStream(ctx); err != nil {
				t.Fatal(err)
			}
			buffers := semanticStreamBuffers{}
			buffers.push(body, ctx)
			if buffers.streamErr == nil || !strings.Contains(buffers.streamErr.Error(), tc.code) {
				t.Fatalf("want boundary error, got %v", buffers.streamErr)
			}
			if !ctx.StreamingAborted {
				t.Fatal("decode error did not abort request")
			}
			if len(ctx.SemanticStreamState.items) != 0 || ctx.SemanticStreamState.terminal {
				t.Fatal("rejected prefix entered response state")
			}
			mutation := buffers.processingResponse(ctx).GetResponseBody().GetResponse().GetBodyMutation()
			if mutation == nil || len(mutation.GetBody()) != 0 {
				t.Fatalf("rejected prefix released to client: %v", mutation)
			}
			buffers.finalize(ctx)
			router.finalizeSemanticStreamingResponse(ctx, buffers.streamErr)
			finalBody := buffers.processingResponse(ctx).GetResponseBody().GetResponse().GetBodyMutation().GetBody()
			if bytes.Contains(finalBody, []byte("forbidden-prefix")) || bytes.Contains(finalBody, []byte(`"nvext"`)) || bytes.Contains(finalBody, []byte("event: request_id")) {
				t.Fatalf("finalization released rejected prefix: %s", finalBody)
			}
			if !ctx.StreamingAborted || !ctx.StreamingComplete {
				t.Fatal("finalization lost aborted state or cleanup")
			}
		})
	}
}

func TestStreamBoundaryRejectionPersistsAcrossResponseBodies(t *testing.T) {
	router := dynamoBoundaryTestRouter("vllm")
	ctx := &RequestContext{
		SourceFormat: llmprotocol.OpenAIChatV1, TargetFormat: llmprotocol.OpenAIChatV1,
		RequestModel: "model-a", UpstreamBackendType: "vllm",
		TraceContext: context.Background(), SemanticRequest: &llmprotocol.Request{},
	}
	body := []byte("data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"model\":\"model-a\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"nvext\":{\"token_ids\":[1]}}\n\n")
	first := router.handleSemanticStreamingResponseBody(body, false, ctx)
	mutation := first.GetResponseBody().GetResponse().GetBodyMutation()
	if mutation == nil || mutation.GetMutation() == nil || len(mutation.GetBody()) != 0 {
		t.Fatalf("rejected terminal chunk must be replaced with an empty body: %v", mutation)
	}
	if !ctx.StreamingAborted || ctx.StreamingComplete {
		t.Fatal("first body must abort the request without finalizing it")
	}

	// Use the real handler again: each response-body call creates a new buffer,
	// but the request-scoped rejection must survive and suppress normal completion.
	second := router.handleSemanticStreamingResponseBody([]byte("data: [DONE]\n\n"), true, ctx)
	mutation = second.GetResponseBody().GetResponse().GetBodyMutation()
	if mutation == nil || mutation.GetMutation() == nil || len(mutation.GetBody()) != 0 {
		t.Fatalf("second body must not emit a finish frame or [DONE]: %v", mutation)
	}
	if !ctx.StreamingAborted || !ctx.StreamingComplete {
		t.Fatal("final body must preserve rejection and complete cleanup")
	}
	if ctx.SemanticStreamState.terminal || len(ctx.SemanticStreamState.items) != 0 {
		t.Fatal("rejected events must not enter response reconstruction state")
	}
}
