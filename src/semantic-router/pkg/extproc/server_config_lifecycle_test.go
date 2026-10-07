package extproc

import (
	"context"
	"errors"
	"fmt"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/configsnapshot"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
)

// newLifecycleTestServer starts a server the way NewServer does: the startup
// document is installed as snapshot version 1 before the router serves.
func newLifecycleTestServer(t *testing.T) (*Server, *routerruntime.Registry, string) {
	t.Helper()
	path := filepath.Join(t.TempDir(), "router.yaml")
	startup := &config.RouterConfig{}
	writeReloadTestDocument(t, path, "startup", startup)
	registry := routerruntime.NewRegistry(startup)
	server := &Server{configPath: path, runtime: registry}
	snapshot, err := server.configManager().Install(context.Background(), configsnapshot.Update{
		Origin: configsnapshot.Origin{Source: configsnapshot.SourceStartup}, Config: startup,
		Document: parsedDocument(path, startup),
	})
	if err != nil {
		t.Fatalf("Install() error = %v", err)
	}
	router := &OpenAIRouter{Config: startup, resources: newResourceScope()}
	server.service = newRouterServiceWithSnapshot(router, snapshot)
	t.Cleanup(func() { _ = server.service.Close() })
	return server, registry, path
}

func stubPassingReload(t *testing.T) {
	t.Helper()
	restore := stubReloadSeams(t)
	t.Cleanup(restore)
	ensureReloadConfigModels = func(*config.RouterConfig) error { return nil }
	buildReloadRouter = func(cfg *config.RouterConfig, _ ...*binding.Pool) (*OpenAIRouter, error) {
		return &OpenAIRouter{Config: cfg, resources: newResourceScope()}, nil
	}
	warmupReloadRouter = func(*OpenAIRouter) error { return nil }
}

func TestFileReloadActivatesTheNextSnapshotVersion(t *testing.T) {
	server, registry, path := newLifecycleTestServer(t)
	stubPassingReload(t)
	candidate := &config.RouterConfig{}
	writeReloadTestDocument(t, path, "candidate", candidate)
	parseReloadConfig = func(string) (*config.RouterConfig, error) { return candidate, nil }

	if err := server.reloadRouterFromFile(path); err != nil {
		t.Fatalf("reloadRouterFromFile() error = %v", err)
	}
	snapshot := server.service.Snapshot()
	if snapshot == nil || snapshot.Version() != 2 || snapshot.Hash() != candidate.DocumentHash ||
		string(snapshot.Document()) != "candidate" || snapshot.Origin().Source != configsnapshot.SourceFile {
		t.Fatalf("serving snapshot = %+v", snapshot)
	}
	if server.service.GetRouter().Config != candidate || registry.ConfigSnapshot() != snapshot {
		t.Fatal("the router and the published snapshot are not the activated candidate")
	}
	activation := registry.ConfigActivation()
	if activation.Status != "active" || activation.Version != 2 || activation.Source != "file" ||
		activation.DocumentHash != candidate.DocumentHash {
		t.Fatalf("published activation = %+v", activation)
	}
}

func TestServerRecordsEveryActivationBesideItsConfiguration(t *testing.T) {
	t.Setenv(configsnapshot.HistoryDirEnv, "")
	t.Setenv(config.ConfigBaseDirEnv, "")
	path := filepath.Join(t.TempDir(), "config.yaml")
	startup := &config.RouterConfig{}
	writeReloadTestDocument(t, path, "startup", startup)
	registry := routerruntime.NewRegistry(startup)
	server := &Server{configPath: path, runtime: registry}
	server.configs = server.newConfigManager(openConfigHistory(path, 3))
	snapshot, err := server.configManager().Install(context.Background(), configsnapshot.Update{
		Origin: configsnapshot.Origin{Source: configsnapshot.SourceStartup}, Config: startup,
		Document: parsedDocument(path, startup),
	})
	if err != nil {
		t.Fatal(err)
	}
	server.service = newRouterServiceWithSnapshot(&OpenAIRouter{Config: startup, resources: newResourceScope()}, snapshot)
	t.Cleanup(func() { _ = server.service.Close() })
	if registry.ConfigLifecycle() != server.configs {
		t.Fatal("the registry does not publish the server's lifecycle")
	}

	stubPassingReload(t)
	candidate := &config.RouterConfig{}
	writeReloadTestDocument(t, path, "candidate", candidate)
	parseReloadConfig = func(string) (*config.RouterConfig, error) { return candidate, nil }
	if err := server.reloadRouterFromFile(path); err != nil {
		t.Fatal(err)
	}
	historyDir := filepath.Join(filepath.Dir(path), ".vllm-sr", "config-backups")
	documents, _ := filepath.Glob(filepath.Join(historyDir, "config.*.yaml"))
	sidecars, _ := filepath.Glob(filepath.Join(historyDir, "config.*.snapshot.json"))
	if len(documents) != 2 || len(sidecars) != 2 {
		t.Fatalf("history holds %d documents and %d sidecars, want 2 each", len(documents), len(sidecars))
	}
	if record, ok := server.configs.History().ByVersion(2); !ok || string(record.Document) != "candidate" ||
		record.Origin.Source != configsnapshot.SourceFile {
		t.Fatalf("version 2 record = %+v, %v", record, ok)
	}
}

func newPersistentLifecycleServer(t *testing.T, path string) *Server {
	t.Helper()
	startup := &config.RouterConfig{}
	writeReloadTestDocument(t, path, filepath.Base(path)+" startup", startup)
	server := &Server{configPath: path, runtime: routerruntime.NewRegistry(startup)}
	server.configs = server.newConfigManager(openConfigHistory(path, 10))
	snapshot, err := server.configManager().Install(context.Background(), configsnapshot.Update{
		Origin: configsnapshot.Origin{Source: configsnapshot.SourceStartup}, Config: startup,
		Document: parsedDocument(path, startup),
	})
	if err != nil {
		t.Fatal(err)
	}
	server.service = newRouterServiceWithSnapshot(&OpenAIRouter{Config: startup, resources: newResourceScope()}, snapshot)
	t.Cleanup(func() { _ = server.service.Close() })
	return server
}

// Two Routers whose documents sit in one directory keep one history each, so
// the versions they serve never cross.
func TestRoutersWhoseDocumentsShareADirectoryKeepSeparateHistories(t *testing.T) {
	t.Setenv(configsnapshot.HistoryDirEnv, "")
	t.Setenv(config.ConfigBaseDirEnv, "")
	dir := t.TempDir()
	paths := []string{filepath.Join(dir, "router-a.yaml"), filepath.Join(dir, "router-b.yaml")}
	servers := []*Server{newPersistentLifecycleServer(t, paths[0]), newPersistentLifecycleServer(t, paths[1])}
	stubPassingReload(t)
	for turn := 1; turn <= 3; turn++ {
		for i, server := range servers {
			candidate := &config.RouterConfig{}
			writeReloadTestDocument(t, paths[i], fmt.Sprintf("%s %d", filepath.Base(paths[i]), turn), candidate)
			parseReloadConfig = func(string) (*config.RouterConfig, error) { return candidate, nil }
			if err := server.reloadRouterFromFile(paths[i]); err != nil {
				t.Fatal(err)
			}
			if version := server.service.Snapshot().Version(); version != uint64(turn+1) {
				t.Fatalf("%s serves v%d after %d reloads, want v%d", paths[i], version, turn, turn+1)
			}
		}
	}
	for i, path := range paths {
		dir := filepath.Join(filepath.Dir(path), ".vllm-sr", "config-backups", filepath.Base(path))
		documents, _ := filepath.Glob(filepath.Join(dir, "config.*.yaml"))
		if len(documents) != 4 {
			t.Fatalf("%s holds %d documents, want this Router's 4", dir, len(documents))
		}
		for _, record := range servers[i].configs.History().List() {
			if !strings.HasPrefix(string(record.Document), filepath.Base(path)) {
				t.Fatalf("%s's history holds another document's v%d: %q", path, record.Version, record.Document)
			}
		}
	}
}

func TestRejectedReloadKeepsTheGenerationAndPublishesItsReasons(t *testing.T) {
	server, registry, path := newLifecycleTestServer(t)
	stubPassingReload(t)
	previous := server.service.GetRouter()
	candidate := &config.RouterConfig{}
	writeReloadTestDocument(t, path, "candidate", candidate)
	parseReloadConfig = func(string) (*config.RouterConfig, error) { return candidate, nil }
	buildReloadRouter = func(*config.RouterConfig, ...*binding.Pool) (*OpenAIRouter, error) {
		return nil, errors.New("classifier deployment unreachable")
	}

	err := server.reloadRouterFromFile(path)
	if err == nil || err.Error() != "classifier deployment unreachable" {
		t.Fatalf("reloadRouterFromFile() error = %v", err)
	}
	if server.service.GetRouter() != previous || server.service.Snapshot().Version() != 1 {
		t.Fatal("a rejected update replaced the serving generation")
	}
	activation := registry.ConfigActivation()
	if activation.Status != "failed" || activation.Stage != "model_prepare" || len(activation.Reasons) != 1 {
		t.Fatalf("published activation = %+v", activation)
	}
	if reason := activation.Reasons[0]; reason.Stage != configsnapshot.StageWarm || reason.Code != configsnapshot.CodeBuildFailed {
		t.Fatalf("reason = %+v", reason)
	}
	if rejection, ok := registry.LastConfigRejection(); !ok || rejection.DocumentHash != candidate.DocumentHash {
		t.Fatalf("last rejection = %+v, %v", rejection, ok)
	}
}

func TestUnparsableFileIsRecordedAsAParseRejection(t *testing.T) {
	server, registry, path := newLifecycleTestServer(t)
	stubPassingReload(t)
	writeReloadTestDocument(t, path, "routing: [", &config.RouterConfig{})
	parseReloadConfig = func(string) (*config.RouterConfig, error) {
		return nil, errors.New("yaml: did not find expected node content")
	}

	if err := server.reloadRouterFromFile(path); err == nil {
		t.Fatal("reloadRouterFromFile() accepted an unparsable document")
	}
	activation := registry.ConfigActivation()
	if activation.Status != "failed" || activation.Stage != "parse" || activation.DocumentHash != documentDigest([]byte("routing: [")) {
		t.Fatalf("published activation = %+v", activation)
	}
	if reasons := activation.Reasons; len(reasons) != 1 || reasons[0].Code != configsnapshot.CodeInvalidDocument {
		t.Fatalf("reasons = %+v", reasons)
	}
	if server.service.Snapshot().Version() != 1 {
		t.Fatal("a parse failure changed the serving snapshot")
	}
}

func TestKubernetesCandidateRunsTheSameLifecycle(t *testing.T) {
	server, registry, _ := newLifecycleTestServer(t)
	stubPassingReload(t)
	ensureReloadConfigModels = func(*config.RouterConfig) error {
		t.Fatal("the lifecycle downloaded models the Kubernetes source prepares")
		return nil
	}
	candidate := &config.RouterConfig{DocumentHash: documentDigest([]byte("crd"))}
	if err := server.reloadRouterFromConfig("kubernetes", server.configPath, candidate); err != nil {
		t.Fatalf("reloadRouterFromConfig() error = %v", err)
	}
	snapshot := server.service.Snapshot()
	if snapshot.Version() != 2 || snapshot.Origin().Source != configsnapshot.SourceKubernetes {
		t.Fatalf("serving snapshot = v%d %+v", snapshot.Version(), snapshot.Origin())
	}

	failure := configsnapshot.Reject(configsnapshot.StageWarm, configsnapshot.CodeModelUnavailable, errors.New("download failed"))
	if err := server.RejectConfigUpdate(configsnapshot.SourceKubernetes, candidate, failure); err == nil {
		t.Fatal("RejectConfigUpdate() returned nil")
	}
	if activation := registry.ConfigActivation(); activation.Status != "failed" || activation.Source != "kubernetes" ||
		activation.Reasons[0].Code != configsnapshot.CodeModelUnavailable {
		t.Fatalf("published activation = %+v", activation)
	}
}

// Requests lease a generation, which pins its router and its snapshot until
// the lease ends. Under concurrent reloads no request fails, none sees its
// snapshot change, and no router closes while a request still holds it.
func TestHotReloadUnderLoadNeverSplitsOrDropsARequest(t *testing.T) {
	server, _, _ := newLifecycleTestServer(t)
	stubPassingReload(t)
	type tracked struct {
		leases atomic.Int32
		closed atomic.Bool
	}
	var mu sync.Mutex
	routers := map[*OpenAIRouter]*tracked{}
	var closedUnderLease atomic.Int32
	track := func(router *OpenAIRouter) {
		state := &tracked{}
		router.resources.add(func() error {
			if state.leases.Load() != 0 {
				closedUnderLease.Add(1)
			}
			state.closed.Store(true)
			return nil
		})
		mu.Lock()
		routers[router] = state
		mu.Unlock()
	}
	track(server.service.GetRouter())
	buildReloadRouter = func(cfg *config.RouterConfig, _ ...*binding.Pool) (*OpenAIRouter, error) {
		router := &OpenAIRouter{Config: cfg, resources: newResourceScope()}
		track(router)
		return router, nil
	}

	stop := make(chan struct{})
	var requests, failures, splits atomic.Int64
	var wg sync.WaitGroup
	for range 16 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			var lastVersion uint64
			for {
				select {
				case <-stop:
					return
				default:
				}
				router, release, err := server.service.lease()
				if err != nil {
					failures.Add(1)
					continue
				}
				mu.Lock()
				state := routers[router]
				mu.Unlock()
				state.leases.Add(1)
				snapshot := router.generation.snapshot
				time.Sleep(50 * time.Microsecond)
				if router.generation.snapshot != snapshot || state.closed.Load() {
					splits.Add(1)
				}
				if snapshot.Version() < lastVersion {
					splits.Add(1)
				}
				lastVersion = snapshot.Version()
				state.leases.Add(-1)
				release()
				requests.Add(1)
			}
		}()
	}

	const reloads = 20
	for i := range reloads {
		candidate := &config.RouterConfig{DocumentHash: documentDigest([]byte{byte(i)})}
		if err := server.reloadRouterFromConfig("kubernetes", server.configPath, candidate); err != nil {
			t.Fatalf("reload %d: %v", i, err)
		}
		time.Sleep(time.Millisecond)
	}
	close(stop)
	wg.Wait()

	if failures.Load() != 0 || splits.Load() != 0 || closedUnderLease.Load() != 0 {
		t.Fatalf("requests %d: failures %d, splits %d, routers closed under a lease %d",
			requests.Load(), failures.Load(), splits.Load(), closedUnderLease.Load())
	}
	t.Logf("%d requests across %d reloads: none failed, none split", requests.Load(), reloads)
	if got := server.service.Snapshot().Version(); got != reloads+1 {
		t.Fatalf("serving version = %d, want %d", got, reloads+1)
	}
	deadline := time.Now().Add(5 * time.Second)
	for {
		retiredOpen := 0
		mu.Lock()
		for router, state := range routers {
			if router != server.service.GetRouter() && !state.closed.Load() {
				retiredOpen++
			}
		}
		mu.Unlock()
		if retiredOpen == 0 {
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("%d retired routers never closed after their requests drained", retiredOpen)
		}
		time.Sleep(10 * time.Millisecond)
	}
}
