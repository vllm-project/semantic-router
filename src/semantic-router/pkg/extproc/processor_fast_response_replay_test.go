package extproc

import (
	"testing"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
)

func TestAddRouterReplayHeaderToImmediateResponse(t *testing.T) {
	response := &ext_proc.ProcessingResponse{
		Response: &ext_proc.ProcessingResponse_ImmediateResponse{
			ImmediateResponse: &ext_proc.ImmediateResponse{},
		},
	}

	addRouterReplayHeaderToImmediateResponse(response, "replay-123")

	immediate := response.GetImmediateResponse()
	if immediate.Headers == nil || len(immediate.Headers.SetHeaders) != 1 {
		t.Fatalf("headers = %#v", immediate.Headers)
	}
	header := immediate.Headers.SetHeaders[0].Header
	if header.Key != headers.RouterReplayID ||
		string(header.RawValue) != "replay-123" {
		t.Fatalf("header = %#v", header)
	}
}

func TestFastResponseReplayAttachesBody(t *testing.T) {
	storage := store.NewMemoryStore(10, 0)
	recorder := routerreplay.NewRecorder(storage)
	recorder.SetCapturePolicy(false, true, 4096)
	const replayID = "fast-replay"
	if _, err := recorder.AddRecord(routerreplay.RoutingRecord{
		ID: replayID,
	}); err != nil {
		t.Fatalf("add replay record: %v", err)
	}
	ctx := &RequestContext{
		RouterReplayID:       replayID,
		RouterReplayRecorder: recorder,
	}
	router := &OpenAIRouter{ReplayRecorder: recorder}
	body := []byte(`{"choices":[{"message":{"content":"policy response"}}]}`)

	router.attachRouterReplayResponse(ctx, body, true)

	record, ok := recorder.GetRecord(replayID)
	if !ok {
		t.Fatal("replay record not found")
	}
	if record.ResponseBody != string(body) {
		t.Fatalf("response body = %q", record.ResponseBody)
	}
}

func TestFastResponseReplayUsesObservedUpstreamErrorForLifecycle(t *testing.T) {
	storage := store.NewMemoryStore(10, 0)
	recorder := routerreplay.NewRecorder(storage)
	const replayID = "fast-replay-upstream-error"
	if _, err := recorder.AddRecord(routerreplay.RoutingRecord{ID: replayID}); err != nil {
		t.Fatalf("add replay record: %v", err)
	}
	ctx := &RequestContext{
		RouterReplayID:       replayID,
		RouterReplayRecorder: recorder,
		UpstreamStatusCode:   503,
	}

	(&OpenAIRouter{ReplayRecorder: recorder}).attachRouterReplayResponse(
		ctx,
		[]byte(`{"error":{"message":"backend unavailable"}}`),
		true,
	)

	record, ok := recorder.GetRecord(replayID)
	if !ok {
		t.Fatal("replay record not found")
	}
	if record.LifecycleState != routerreplay.LifecycleFailed || record.TerminalReason != "upstream_error_response" {
		t.Fatalf("lifecycle = %q / %q, want failed upstream error", record.LifecycleState, record.TerminalReason)
	}
}
