package extproc

import (
	"context"
	"errors"
	"net"
	"strconv"
	"sync"
	"testing"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	healthpb "google.golang.org/grpc/health/grpc_health_v1"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
)

func TestKubernetesActivationWaitsForListenerAndKeepsServingOnFailure(t *testing.T) {
	restore := stubReloadSeams(t)
	defer restore()
	reservation, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	port := reservation.Addr().(*net.TCPAddr).Port
	_ = reservation.Close()
	initial := &config.RouterConfig{ConfigSource: config.ConfigSourceKubernetes, DocumentHash: "initial"}
	server := &Server{port: port, service: NewRouterService(&OpenAIRouter{Config: initial}), runtime: routerruntime.NewRegistry(initial)}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	candidate := &config.RouterConfig{ConfigSource: config.ConfigSourceKubernetes, DocumentHash: "candidate"}
	buildReloadRouter = func(cfg *config.RouterConfig, _ ...*binding.Pool) (*OpenAIRouter, error) {
		return &OpenAIRouter{Config: cfg}, nil
	}
	warmupEntered := make(chan struct{})
	allowWarmup := make(chan struct{})
	var releaseOnce sync.Once
	releaseWarmup := func() { releaseOnce.Do(func() { close(allowWarmup) }) }
	defer releaseWarmup()
	warmupReloadRouter = func(*OpenAIRouter) error {
		close(warmupEntered)
		<-allowWarmup
		return nil
	}
	applied := make(chan error, 1)
	go func() { applied <- server.ActivateKubernetesConfig(ctx, candidate) }()
	select {
	case activationErr := <-applied:
		t.Fatalf("activation before listener: %v", activationErr)
	case <-time.After(20 * time.Millisecond):
	}
	serving := make(chan error, 1)
	go func() { serving <- server.StartContext(ctx) }()
	t.Cleanup(func() { cancel(); server.Stop(); <-serving })
	select {
	case <-warmupEntered:
	case <-ctx.Done():
		t.Fatal(ctx.Err())
	}
	conn, err := grpc.NewClient("127.0.0.1:"+strconv.Itoa(port), grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = conn.Close() }()
	health := healthpb.NewHealthClient(conn)
	check := func(want healthpb.HealthCheckResponse_ServingStatus) {
		t.Helper()
		response, err := health.Check(ctx, &healthpb.HealthCheckRequest{})
		if err != nil {
			t.Fatal(err)
		}
		if response.Status != want {
			t.Fatalf("health = %v, want %v", response.Status, want)
		}
	}
	check(healthpb.HealthCheckResponse_NOT_SERVING)
	releaseWarmup()
	if err := <-applied; err != nil {
		t.Fatal(err)
	}
	check(healthpb.HealthCheckResponse_SERVING)
	if server.CurrentConfig() != candidate {
		t.Fatal("candidate was not published before acknowledgement")
	}
	buildReloadRouter = func(*config.RouterConfig, ...*binding.Pool) (*OpenAIRouter, error) {
		return nil, errors.New("candidate dependency unavailable")
	}
	if err := server.ActivateKubernetesConfig(ctx, &config.RouterConfig{ConfigSource: config.ConfigSourceKubernetes, DocumentHash: "failed"}); err == nil {
		t.Fatal("failed candidate acknowledged")
	}
	check(healthpb.HealthCheckResponse_SERVING)
	if server.CurrentConfig() != candidate {
		t.Fatal("failure replaced last serving generation")
	}
	if state := server.runtime.ConfigActivation(); state.Status != "failed" {
		t.Fatalf("activation = %+v", state)
	}
}

func TestKubernetesActivationCancellationBeforeListener(t *testing.T) {
	server := &Server{}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := server.ActivateKubernetesConfig(ctx, &config.RouterConfig{}); !errors.Is(err, context.Canceled) {
		t.Fatalf("activation = %v", err)
	}
}

func TestKubernetesActivationCancellationDuringWarmupDoesNotPublish(t *testing.T) {
	restore := stubReloadSeams(t)
	defer restore()
	initial := &config.RouterConfig{ConfigSource: config.ConfigSourceKubernetes}
	ready := make(chan struct{})
	close(ready)
	server := &Server{servingReady: ready, service: NewRouterService(&OpenAIRouter{Config: initial}), runtime: routerruntime.NewRegistry(initial)}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	buildReloadRouter = func(cfg *config.RouterConfig, _ ...*binding.Pool) (*OpenAIRouter, error) {
		return &OpenAIRouter{Config: cfg}, nil
	}
	warmupReloadRouter = func(*OpenAIRouter) error { cancel(); return nil }
	if err := server.ActivateKubernetesConfig(ctx, &config.RouterConfig{ConfigSource: config.ConfigSourceKubernetes}); !errors.Is(err, context.Canceled) {
		t.Fatalf("activation = %v", err)
	}
	if server.CurrentConfig() != initial {
		t.Fatal("cancelled candidate was published")
	}
}
