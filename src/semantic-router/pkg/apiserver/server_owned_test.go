//go:build !windows && cgo

package apiserver

import (
	"context"
	"errors"
	"net"
	"net/http"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/services"
)

type countedAPIResource struct{ closes atomic.Int32 }

func (r *countedAPIResource) Close() error { r.closes.Add(1); return nil }

func TestOwnedAPIResourceClosesAfterCanceledHandlerCompletes(t *testing.T) {
	started, finish := make(chan struct{}), make(chan struct{})
	var once sync.Once
	unblock := func() { once.Do(func() { close(finish) }) }
	t.Cleanup(unblock)
	owner := &countedAPIResource{}
	httpServer := &http.Server{Addr: "127.0.0.1:0", ReadHeaderTimeout: time.Second, Handler: http.HandlerFunc(func(_ http.ResponseWriter, _ *http.Request) {
		close(started)
		<-finish // A non-preemptible model call outlives cancellation.
	})}
	server, err := startHTTPServer(httpServer, owner)
	if err != nil {
		t.Fatal(err)
	}
	go func() {
		response, _ := http.Get("http://" + httpServer.Addr)
		if response != nil {
			_ = response.Body.Close()
		}
	}()
	<-started
	ctx, cancel := context.WithTimeout(context.Background(), 25*time.Millisecond)
	defer cancel()
	if err := server.Shutdown(ctx); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("shutdown error=%v", err)
	}
	if owner.closes.Load() != 0 {
		t.Fatal("native owner closed before canceled handler returned")
	}
	unblock()
	select {
	case <-server.ownedDone:
	case <-time.After(time.Second):
		t.Fatal("owner never closed after handler drain")
	}
	if owner.closes.Load() != 1 {
		t.Fatalf("close count=%d", owner.closes.Load())
	}
	if err := server.Shutdown(context.Background()); err != nil {
		t.Fatal(err)
	}
	if owner.closes.Load() != 1 {
		t.Fatal("repeated shutdown closed owner twice")
	}
}

func TestOwnedAPIResourceClosesOnListenFailure(t *testing.T) {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer listener.Close()
	owner := &countedAPIResource{}
	_, err = startHTTPServer(&http.Server{Addr: listener.Addr().String(), ReadHeaderTimeout: time.Second}, owner)
	if err == nil || owner.closes.Load() != 1 {
		t.Fatalf("listen failure leaked owner: err=%v closes=%d", err, owner.closes.Load())
	}
}

func TestAPIStartupBorrowsExistingRouterAndGlobalService(t *testing.T) {
	cfg := &config.RouterConfig{}
	service := services.NewPlaceholderClassificationService()
	registry := routerruntime.NewRegistry(cfg)
	registry.SetClassificationService(service)
	actual, owner := classificationServiceForStartup(cfg, registry)
	if actual != service || owner != nil {
		t.Fatal("router-owned service was adopted by API")
	}
	previous := services.GetGlobalClassificationService()
	services.SetGlobalClassificationService(service)
	t.Cleanup(func() { services.SetGlobalClassificationService(previous) })
	actual, owner = classificationServiceForStartup(cfg, nil)
	if actual != service || owner != nil {
		t.Fatal("global service was adopted by API")
	}
}
