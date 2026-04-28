package extproc

// Tests for the panic-recovery guard added to Process() in processor_core.go.
//
// Background: the router crashed silently (OOM kill, no container logs) when
// the mmBERT-32K classifier received ~4 000-token prompts. The Rust fix
// enforces a 512-token cap in the tokenizer; the Go fix adds a `defer recover()`
// to Process() so that even if a Rust OOM surfaces as a CGO runtime panic it
// is caught, logged, and returned to the caller as a gRPC Internal error rather
// than crashing the entire server. (GitHub issue #1843)
//
// These tests exercise the Go-level panic recovery on mocked ext_proc streams.
// They use CreateTestConfig/CreateTestRouter like other extproc tests; when CGO
// and model assets are available that may perform real initialization, but the
// panic/assertion path does not depend on mmBERT or long-prompt fixtures.

import (
	"fmt"
	"io"
	"testing"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// buildMinimalHeaderRequest returns the smallest valid ProcessingRequest_RequestHeaders
// that lets Process() advance past the initial Recv() and reach Send().
func buildMinimalHeaderRequest() *ext_proc.ProcessingRequest {
	return &ext_proc.ProcessingRequest{
		Request: &ext_proc.ProcessingRequest_RequestHeaders{
			RequestHeaders: &ext_proc.HttpHeaders{
				Headers: &core.HeaderMap{
					Headers: []*core.HeaderValue{
						{Key: "content-type", Value: "application/json"},
					},
				},
			},
		},
	}
}

// panicOnSendStream is a MockStream whose Send() always panics. This simulates
// a Rust CGO inference call that triggers an unrecoverable error (e.g. OOM).
// Because the panic propagates up through the call stack and reaches the
// `defer recover()` in Process(), the server must return a gRPC Internal error
// rather than crashing.
type panicOnSendStream struct {
	MockStream
	panicMsg string
}

func (p *panicOnSendStream) Send(_ *ext_proc.ProcessingResponse) error {
	panic(p.panicMsg)
}

// panicOnRecvStream panics during the first Recv() call — simulating a panic
// that fires before any response is sent (e.g. inside a CGO classify call
// reached from handleRequestBody).
// panicVal may be any type (string, error, etc.) to test different panic payloads.
type panicOnRecvStream struct {
	MockStream
	panicVal  any
	callCount int
}

func (p *panicOnRecvStream) Recv() (*ext_proc.ProcessingRequest, error) {
	p.callCount++
	if p.callCount == 1 {
		panic(p.panicVal)
	}
	return nil, io.EOF
}

// TestProcessPanicRecovery_RecvPanic verifies that a panic raised during
// Recv() (e.g. a CGO OOM originating inside request processing) is caught by
// the defer/recover guard in Process() and returned as a gRPC Internal error.
func TestProcessPanicRecovery_RecvPanic(t *testing.T) {
	cfg := CreateTestConfig()
	router, err := CreateTestRouter(cfg)
	if err != nil {
		t.Fatalf("CreateTestRouter: %v", err)
	}

	const wantMsg = "simulated CGO OOM panic"
	stream := &panicOnRecvStream{
		MockStream: MockStream{Ctx: t.Context()},
		panicVal:   wantMsg,
	}

	err = router.Process(stream)

	if err == nil {
		t.Fatal("Process() returned nil; expected gRPC Internal error after panic")
	}

	st, ok := status.FromError(err)
	if !ok {
		t.Fatalf("Process() returned non-gRPC error: %v", err)
	}
	if st.Code() != codes.Internal {
		t.Errorf("Process() returned gRPC code %v; want %v", st.Code(), codes.Internal)
	}
	if st.Message() == "" {
		t.Error("Process() returned empty error message; want panic description")
	}
	t.Logf("recovered gRPC error: code=%v msg=%q", st.Code(), st.Message())
}

// TestProcessPanicRecovery_SendPanic verifies that a panic raised during
// Send() (e.g. from a downstream inference OOM reached via processRequestHeaders)
// is also caught, preventing server crash.
func TestProcessPanicRecovery_SendPanic(t *testing.T) {
	cfg := CreateTestConfig()
	router, err := CreateTestRouter(cfg)
	if err != nil {
		t.Fatalf("CreateTestRouter: %v", err)
	}

	const wantMsg = "simulated send-path CGO panic"

	// Provide a minimal valid header request so Process() reaches Send().
	stream := &panicOnSendStream{
		MockStream: *NewMockStream([]*ext_proc.ProcessingRequest{
			buildMinimalHeaderRequest(),
		}),
		panicMsg: wantMsg,
	}

	err = router.Process(stream)

	if err == nil {
		t.Fatal("Process() returned nil; expected gRPC Internal error after panic")
	}

	st, ok := status.FromError(err)
	if !ok {
		t.Fatalf("Process() returned non-gRPC error: %v", err)
	}
	if st.Code() != codes.Internal {
		t.Errorf("Process() returned gRPC code %v; want %v", st.Code(), codes.Internal)
	}
	t.Logf("recovered gRPC error: code=%v msg=%q", st.Code(), st.Message())
}

// TestProcessPanicRecovery_StringPanic verifies that a string panic (not an
// error value) is also handled correctly.
func TestProcessPanicRecovery_StringPanic(t *testing.T) {
	cfg := CreateTestConfig()
	router, err := CreateTestRouter(cfg)
	if err != nil {
		t.Fatalf("CreateTestRouter: %v", err)
	}

	stream := &panicOnRecvStream{
		MockStream: MockStream{Ctx: t.Context()},
		panicVal:   "runtime: out of memory",
	}

	err = router.Process(stream)
	if err == nil {
		t.Fatal("Process() returned nil; expected gRPC Internal error")
	}

	st, ok := status.FromError(err)
	if !ok {
		t.Fatalf("non-gRPC error returned: %v", err)
	}
	if st.Code() != codes.Internal {
		t.Errorf("wrong code %v; want codes.Internal", st.Code())
	}
}

// TestProcessPanicRecovery_ErrorPanic verifies that an error-typed panic value
// (i.e. panic(err) where err is a non-nil error) is also handled correctly.
// This is distinct from a string panic and exercises the %v formatting path of
// the status.Errorf call in Process()'s recover block.
func TestProcessPanicRecovery_ErrorPanic(t *testing.T) {
	cfg := CreateTestConfig()
	router, err := CreateTestRouter(cfg)
	if err != nil {
		t.Fatalf("CreateTestRouter: %v", err)
	}

	ep := &panicOnRecvStream{
		MockStream: MockStream{Ctx: t.Context()},
		panicVal:   fmt.Errorf("candle OOM: alloc failed"),
	}

	err = router.Process(ep)
	if err == nil {
		t.Fatal("Process() returned nil; expected gRPC Internal error")
	}
	st, _ := status.FromError(err)
	if st.Code() != codes.Internal {
		t.Errorf("wrong code %v; want codes.Internal", st.Code())
	}
}

// TestProcessNoPanicOnNormalEOF confirms that a normal EOF stream (no panic)
// still returns nil — we haven't accidentally broken the happy path.
func TestProcessNoPanicOnNormalEOF(t *testing.T) {
	cfg := CreateTestConfig()
	router, err := CreateTestRouter(cfg)
	if err != nil {
		t.Fatalf("CreateTestRouter: %v", err)
	}

	// Empty request list → Recv immediately returns io.EOF → Process returns nil.
	stream := NewMockStream([]*ext_proc.ProcessingRequest{})

	err = router.Process(stream)
	if err != nil {
		t.Errorf("Process() returned unexpected error on clean EOF: %v", err)
	}
}
