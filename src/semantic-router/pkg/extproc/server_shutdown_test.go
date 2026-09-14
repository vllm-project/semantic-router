package extproc

import (
	"context"
	"errors"
	"net"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"github.com/fsnotify/fsnotify"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/test/bufconn"
	"google.golang.org/protobuf/types/known/emptypb"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
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

type scheduledReloadShutdownFixture struct {
	server                   *Server
	resourcesClosed          chan struct{}
	candidateResourcesClosed chan struct{}
	releaseReload            func()
}

func startScheduledReloadShutdownFixture(t *testing.T) *scheduledReloadShutdownFixture {
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

	watchCtx, watcherDone := server.lifecycle.startWatcher(context.Background())
	watcher, err := fsnotify.NewWatcher()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = watcher.Close() })
	reloadStarted := make(chan struct{})
	releaseReloadCh := make(chan struct{})
	var releaseReloadOnce sync.Once
	releaseReload := func() { releaseReloadOnce.Do(func() { close(releaseReloadCh) }) }
	t.Cleanup(func() {
		releaseReload()
		cleanupCtx, cancel := context.WithTimeout(context.Background(), time.Second)
		defer cancel()
		if err := server.lifecycle.stopAndWaitForBackgroundWork(cleanupCtx); err != nil {
			t.Errorf("wait for scheduled reload cleanup: %v", err)
		}
	})
	candidateCfg := &config.RouterConfig{}
	candidateResourcesClosed := make(chan struct{})
	parseReloadConfig = func(string) (*config.RouterConfig, error) {
		return candidateCfg, nil
	}
	ensureReloadConfigModels = func(*config.RouterConfig) error { return nil }
	prepareReloadRuntime = func(*config.RouterConfig) (modelruntime.EmbeddingRuntimeState, error) {
		close(reloadStarted)
		<-releaseReloadCh
		return modelruntime.EmbeddingRuntimeState{}, nil
	}
	buildReloadRouter = func(cfg *config.RouterConfig, _ ...*binding.Pool) (*OpenAIRouter, error) {
		resources := newResourceScope()
		resources.add(func() error {
			close(candidateResourcesClosed)
			return nil
		})
		return (&routerComponents{cfg: cfg, resources: resources}).buildRouter(), nil
	}
	warmupReloadRouter = func(*OpenAIRouter, modelruntime.EmbeddingRuntimeState) error {
		return nil
	}
	configPath := filepath.Join(t.TempDir(), "router.yaml")
	loop := configFileReloadLoop{server: server, watcher: watcher, cfgFile: configPath}
	go func() {
		defer watcherDone()
		loop.run(watchCtx)
	}()
	loop.scheduleReload(watchCtx, fsnotify.Event{Name: configPath, Op: fsnotify.Write})
	select {
	case <-reloadStarted:
	case <-time.After(time.Second):
		t.Fatal("scheduled reload did not start")
	}
	return &scheduledReloadShutdownFixture{
		server:                   server,
		resourcesClosed:          resourcesClosed,
		candidateResourcesClosed: candidateResourcesClosed,
		releaseReload:            releaseReload,
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

func TestServerStartContextRejectsStartupAfterShutdown(t *testing.T) {
	server := &Server{service: NewRouterService(nil)}
	if err := server.ShutdownServing(context.Background()); err != nil {
		t.Fatalf("ShutdownServing() error = %v", err)
	}

	err := server.StartContext(context.Background())
	if err == nil || err.Error() != "router server is shutting down" {
		t.Fatalf("StartContext() error = %v, want router server is shutting down", err)
	}
	server.servingMu.Lock()
	defer server.servingMu.Unlock()
	if server.server != nil {
		t.Fatal("StartContext() installed a gRPC server after shutdown")
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

func TestServerShutdownResourcesWaitsForScheduledReload(t *testing.T) {
	restoreReloadSeams := stubReloadSeams(t)
	t.Cleanup(restoreReloadSeams)
	fixture := startScheduledReloadShutdownFixture(t)

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

	fixture.releaseReload()
	shutdownErr := <-shutdownDone
	if closedDuringReload {
		t.Fatal("generation resources closed while reload was still running")
	}
	if shutdownErr != nil {
		t.Fatalf("ShutdownResources() error = %v", shutdownErr)
	}
	select {
	case <-fixture.resourcesClosed:
	default:
		t.Fatal("generation resources remained open after reload exited")
	}
	select {
	case <-fixture.candidateResourcesClosed:
	default:
		t.Fatal("published reload generation remained open after shutdown")
	}
}

func TestServerShutdownResourcesDoesNotCloseUnderStuckScheduledReload(t *testing.T) {
	restoreReloadSeams := stubReloadSeams(t)
	t.Cleanup(restoreReloadSeams)
	fixture := startScheduledReloadShutdownFixture(t)

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

	fixture.releaseReload()
	select {
	case <-fixture.candidateResourcesClosed:
	case <-time.After(time.Second):
		t.Fatal("published reload generation remained open after timed-out shutdown resumed")
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
