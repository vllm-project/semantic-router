package extproc

import (
	"context"
	"errors"
	"net"
	"os"
	"path/filepath"
	"testing"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/test/bufconn"
	"google.golang.org/protobuf/types/known/emptypb"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/tools"
)

type shutdownTestService interface{}

type blockingWarmupEmbeddingProvider struct {
	started chan struct{}
	release chan struct{}
}

func (p *blockingWarmupEmbeddingProvider) Embed(context.Context, string) ([]float32, error) {
	close(p.started)
	<-p.release
	return []float32{1}, nil
}

func (p *blockingWarmupEmbeddingProvider) EmbedBatch(context.Context, []string) ([][]float32, error) {
	return nil, errors.New("unexpected batch embedding")
}

func (*blockingWarmupEmbeddingProvider) Dimension() int  { return 1 }
func (*blockingWarmupEmbeddingProvider) Backend() string { return "test" }

type kubernetesReloadShutdownFixture struct {
	server          *Server
	resourcesClosed chan struct{}
	releaseReload   chan struct{}
	reloadDone      chan error
}

func startKubernetesReloadShutdownFixture(t *testing.T) *kubernetesReloadShutdownFixture {
	t.Helper()
	resourcesClosed := make(chan struct{})
	resources := newResourceScope()
	resources.add(func() error {
		close(resourcesClosed)
		return nil
	})
	server := &Server{
		service: NewRouterService((&routerComponents{resources: resources}).buildRouter()),
	}
	t.Cleanup(func() { _ = server.service.Close() })

	_, watcherDone := server.lifecycle.startWatcher(context.Background())
	reloadStarted := make(chan struct{})
	releaseReload := make(chan struct{})
	reloadDone := make(chan error, 1)
	prepareReloadRuntime = func(*config.RouterConfig) (modelruntime.EmbeddingRuntimeState, error) {
		close(reloadStarted)
		<-releaseReload
		return modelruntime.EmbeddingRuntimeState{}, nil
	}
	buildReloadRouter = func(cfg *config.RouterConfig) (*OpenAIRouter, error) {
		return &OpenAIRouter{Config: cfg}, nil
	}
	warmupReloadRouter = func(*OpenAIRouter, modelruntime.EmbeddingRuntimeState) error {
		return nil
	}
	go func() {
		defer watcherDone()
		reloadDone <- server.reloadRouterFromConfig("kubernetes", "", &config.RouterConfig{})
	}()
	<-reloadStarted
	return &kubernetesReloadShutdownFixture{
		server:          server,
		resourcesClosed: resourcesClosed,
		releaseReload:   releaseReload,
		reloadDone:      reloadDone,
	}
}

func TestServerShutdownServingLeavesGenerationResourcesOpen(t *testing.T) {
	resourcesClosed := make(chan struct{})
	resources := newResourceScope()
	resources.add(func() error {
		close(resourcesClosed)
		return nil
	})
	router := (&routerComponents{resources: resources}).buildRouter()
	server := &Server{service: NewRouterService(router)}

	if err := server.ShutdownServing(context.Background()); err != nil {
		t.Fatalf("ShutdownServing() error = %v", err)
	}
	select {
	case <-resourcesClosed:
		t.Fatal("ShutdownServing() closed generation resources")
	default:
	}

	if err := server.ShutdownResources(context.Background()); err != nil {
		t.Fatalf("ShutdownResources() error = %v", err)
	}
	select {
	case <-resourcesClosed:
	default:
		t.Fatal("ShutdownResources() left generation resources open")
	}
}

func TestServerShutdownDoesNotCloseResourcesUnderCanceledStartupWarmup(t *testing.T) {
	toolsPath := filepath.Join(t.TempDir(), "tools.json")
	if err := os.WriteFile(toolsPath, []byte(`[{"tool":{"type":"function","function":{"name":"lookup"}},"description":"lookup"}]`), 0o600); err != nil {
		t.Fatal(err)
	}
	provider := &blockingWarmupEmbeddingProvider{
		started: make(chan struct{}),
		release: make(chan struct{}),
	}
	released := false
	t.Cleanup(func() {
		if !released {
			close(provider.release)
		}
	})
	toolsDatabase := tools.NewToolsDatabase(tools.ToolsDatabaseOptions{
		Enabled:         true,
		Provider:        provider,
		TargetDimension: 1,
	})
	resourcesClosed := make(chan struct{})
	resources := newResourceScope()
	resources.add(func() error {
		close(resourcesClosed)
		return nil
	})
	router := (&routerComponents{
		cfg: &config.RouterConfig{ToolSelection: config.ToolSelection{
			Tools: config.ToolsConfig{
				Enabled:     true,
				ToolsDBPath: toolsPath,
			},
		}},
		resources:     resources,
		toolsDatabase: toolsDatabase,
	}).buildRouter()
	server := &Server{service: NewRouterService(router)}

	warmupCtx, cancelWarmup := context.WithCancel(context.Background())
	warmupDone := make(chan error, 1)
	go func() {
		warmupDone <- server.WarmupRouter(
			warmupCtx,
			modelruntime.EmbeddingRuntimeState{ToolsReady: true},
			modelruntime.WarmupRouterOptions{MaxParallelism: 1},
		)
	}()
	select {
	case <-provider.started:
	case <-time.After(time.Second):
		t.Fatal("startup warmup did not begin")
	}
	cancelWarmup()
	if err := <-warmupDone; !errors.Is(err, context.Canceled) {
		t.Fatalf("WarmupRouter() error = %v, want context canceled", err)
	}

	shutdownCtx, cancelShutdown := context.WithTimeout(context.Background(), 25*time.Millisecond)
	defer cancelShutdown()
	if err := server.ShutdownResources(shutdownCtx); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("ShutdownResources() error = %v, want deadline exceeded", err)
	}
	select {
	case <-resourcesClosed:
		t.Fatal("generation resources closed while startup warmup was still running")
	default:
	}

	close(provider.release)
	released = true
	select {
	case <-resourcesClosed:
	case <-time.After(time.Second):
		t.Fatal("generation resources remained open after startup warmup exited")
	}
}

func TestServerShutdownResourcesWaitsForKubernetesReload(t *testing.T) {
	restoreReloadSeams := stubReloadSeams(t)
	defer restoreReloadSeams()
	fixture := startKubernetesReloadShutdownFixture(t)

	if err := fixture.server.ShutdownServing(context.Background()); err != nil {
		t.Fatalf("ShutdownServing() error = %v", err)
	}
	shutdownDone := make(chan error, 1)
	go func() {
		shutdownDone <- fixture.server.ShutdownResources(context.Background())
	}()

	closedDuringReload := false
	select {
	case <-fixture.resourcesClosed:
		closedDuringReload = true
	case <-time.After(50 * time.Millisecond):
	}

	close(fixture.releaseReload)
	reloadErr := <-fixture.reloadDone
	shutdownErr := <-shutdownDone
	if closedDuringReload {
		t.Fatal("generation resources closed while reload was still running")
	}
	if reloadErr != nil {
		t.Fatalf("reload error = %v", reloadErr)
	}
	if shutdownErr != nil {
		t.Fatalf("ShutdownResources() error = %v", shutdownErr)
	}
	select {
	case <-fixture.resourcesClosed:
	default:
		t.Fatal("generation resources remained open after reload exited")
	}
}

func TestServerShutdownResourcesDoesNotCloseUnderStuckKubernetesReload(t *testing.T) {
	restoreReloadSeams := stubReloadSeams(t)
	defer restoreReloadSeams()
	fixture := startKubernetesReloadShutdownFixture(t)

	if err := fixture.server.ShutdownServing(context.Background()); err != nil {
		t.Fatalf("ShutdownServing() error = %v", err)
	}
	shutdownCtx, cancelShutdown := context.WithTimeout(context.Background(), 25*time.Millisecond)
	defer cancelShutdown()
	err := fixture.server.ShutdownResources(shutdownCtx)
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("ShutdownResources() error = %v, want deadline exceeded", err)
	}
	select {
	case <-fixture.resourcesClosed:
		t.Fatal("generation resources closed under stuck reload")
	default:
	}

	close(fixture.releaseReload)
	if reloadErr := <-fixture.reloadDone; reloadErr != nil {
		t.Fatalf("reload error = %v", reloadErr)
	}
}

func TestServerShutdownDrainsGenerationAfterForcedGRPCStopWithinDeadline(t *testing.T) {
	resourcesClosed := make(chan struct{})
	resources := newResourceScope()
	resources.add(func() error {
		close(resourcesClosed)
		return nil
	})
	router := (&routerComponents{resources: resources}).buildRouter()
	service := NewRouterService(router)

	rpcStarted := make(chan struct{})
	grpcServer := grpc.NewServer()
	grpcServer.RegisterService(&grpc.ServiceDesc{
		ServiceName: "extproc.shutdown.test",
		HandlerType: (*shutdownTestService)(nil),
		Streams: []grpc.StreamDesc{{
			StreamName:    "Hold",
			ClientStreams: true,
			ServerStreams: true,
			Handler: func(_ any, stream grpc.ServerStream) error {
				release, acquired := service.current.Load().acquire()
				if !acquired {
					return errors.New("failed to acquire generation")
				}
				defer release()
				close(rpcStarted)
				<-stream.Context().Done()
				return stream.Context().Err()
			},
		}},
	}, struct{}{})

	listener := bufconn.Listen(1024 * 1024)
	go func() { _ = grpcServer.Serve(listener) }()
	t.Cleanup(func() { _ = listener.Close() })
	connection, err := grpc.NewClient(
		"passthrough:///bufnet",
		grpc.WithContextDialer(func(context.Context, string) (net.Conn, error) { return listener.Dial() }),
		grpc.WithTransportCredentials(insecure.NewCredentials()),
	)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = connection.Close() })
	stream, err := connection.NewStream(context.Background(), &grpc.StreamDesc{ClientStreams: true, ServerStreams: true}, "/extproc.shutdown.test/Hold")
	if err != nil {
		t.Fatal(err)
	}
	if sendErr := stream.SendMsg(&emptypb.Empty{}); sendErr != nil {
		t.Fatal(sendErr)
	}
	select {
	case <-rpcStarted:
	case <-time.After(time.Second):
		t.Fatal("test RPC did not start")
	}

	server := &Server{server: grpcServer, service: service}
	shutdownCtx, cancel := context.WithTimeout(context.Background(), 200*time.Millisecond)
	defer cancel()
	started := time.Now()
	err = server.Shutdown(shutdownCtx)
	if elapsed := time.Since(started); elapsed < 50*time.Millisecond {
		t.Fatalf("Shutdown() took %v, want short budgets shared with graceful stop", elapsed)
	} else if elapsed > 250*time.Millisecond {
		t.Fatalf("Shutdown() took %v, want it bounded by the caller deadline", elapsed)
	}
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("Shutdown() error = %v, want deadline exceeded after forced stop", err)
	}
	select {
	case <-resourcesClosed:
	default:
		t.Fatal("generation resources were not closed before forced shutdown returned")
	}
}
