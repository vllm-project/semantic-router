package extproc

import (
	"context"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
)

func TestStartContextWithoutExtProcServesTheRoutingCore(t *testing.T) {
	initial := &config.RouterConfig{}
	server := &Server{service: NewRouterService(&OpenAIRouter{Config: initial}), runtime: routerruntime.NewRegistry(initial)}
	ctx, cancel := context.WithCancel(context.Background())
	served := make(chan struct{})
	done := make(chan error, 1)
	go func() { done <- server.StartContextWithoutExtProc(ctx, func() { close(served) }) }()
	select {
	case <-served:
	case <-time.After(5 * time.Second):
		t.Fatal("onServing was not called")
	}
	waitCtx, cancelWait := context.WithTimeout(context.Background(), time.Second)
	defer cancelWait()
	if err := server.WaitForServing(waitCtx); err != nil {
		t.Fatalf("WaitForServing = %v", err)
	}
	lease, err := server.Pin()
	if err != nil {
		t.Fatalf("the routing core must serve: %v", err)
	}
	session, err := lease.Router.Open(context.Background())
	if err != nil {
		t.Fatalf("the routing core must be open for sessions: %v", err)
	}
	session.Close(nil)
	lease.Release()
	if err := server.StartContextWithoutExtProc(context.Background(), nil); err == nil {
		t.Fatal("a second start must be refused")
	}
	cancel()
	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("StartContextWithoutExtProc = %v", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("the server did not stop with its context")
	}
	if server.server != nil {
		t.Fatal("no ext_proc gRPC server may be started")
	}
}
