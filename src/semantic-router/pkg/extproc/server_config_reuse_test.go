package extproc

import (
	"context"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/configsnapshot"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
)

func parityCorpusDocument(t *testing.T) string {
	t.Helper()
	_, file, _, _ := runtime.Caller(0)
	data, err := os.ReadFile(filepath.Join(filepath.Dir(file), "../routing/parity/testdata/corpus/config.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	return string(data)
}

// newRealRouterServer starts a server whose router is built for real from
// document, the way NewServer builds it, so reloads can share its parts.
func newRealRouterServer(t *testing.T, document string) *Server {
	t.Helper()
	cfg, err := config.ParseYAMLBytes([]byte(document))
	if err != nil {
		t.Fatal(err)
	}
	router, err := buildOpenAIRouterFromConfig(cfg)
	if err != nil {
		t.Fatal(err)
	}
	server := &Server{configPath: filepath.Join(t.TempDir(), "config.yaml"), runtime: routerruntime.NewRegistry(cfg)}
	snapshot, err := server.configManager().Install(context.Background(), configsnapshot.Update{
		Origin: configsnapshot.Origin{Source: configsnapshot.SourceStartup}, Config: cfg,
	})
	if err != nil {
		t.Fatal(err)
	}
	router.nameSignals(snapshot.ComponentKey(configsnapshot.ComponentSignals))
	server.service = newRouterServiceWithSnapshot(router, snapshot)
	t.Cleanup(func() { _ = server.service.Close() })
	return server
}

func reloadDocument(t *testing.T, server *Server, document string) *OpenAIRouter {
	t.Helper()
	cfg, err := config.ParseYAMLBytes([]byte(document))
	if err != nil {
		t.Fatal(err)
	}
	if err := server.reloadRouterFromConfig("kubernetes", server.configPath, cfg); err != nil {
		t.Fatalf("reload: %v", err)
	}
	return server.service.GetRouter()
}

func TestEndpointOnlyChangeKeepsSignalsAndModelBindings(t *testing.T) {
	document := parityCorpusDocument(t)
	server := newRealRouterServer(t, document)
	first := server.service.GetRouter()

	second := reloadDocument(t, server, strings.Replace(document, "127.0.0.1:18000", "127.0.0.1:18100", 1))
	if second == first {
		t.Fatal("the endpoint change did not build a new routing pipeline")
	}
	if second.signals != first.signals || second.RecipeClassifiers != first.RecipeClassifiers ||
		second.Classifier != first.Classifier || second.Embeddings != first.Embeddings ||
		second.signals.serving != first.signals.serving {
		t.Fatal("an endpoint-only change rebuilt the signal runtime: classifiers, embeddings or model bindings")
	}
	if second.ClassificationService == first.ClassificationService {
		t.Fatal("the new generation shares the old one's classification service, which wraps its configuration")
	}
	expectedReuse := []configsnapshot.Component{configsnapshot.ComponentModelService, configsnapshot.ComponentSignals}
	if reused := server.service.Snapshot().Reused(); !slices.Equal(reused, expectedReuse) {
		t.Fatalf("Reused() = %v, want %v", reused, expectedReuse)
	}
	if attempt := server.configs.Status().Latest; !slices.Equal(attempt.Reused, expectedReuse) {
		t.Fatalf("the attempt does not record the reuse: %+v", attempt)
	}

	third := reloadDocument(t, server, strings.Replace(document, `"urgent", "asap"`, `"urgent", "asap", "now"`, 1))
	if third.signals == second.signals || third.RecipeClassifiers == second.RecipeClassifiers {
		t.Fatal("a signal change kept the old classifiers")
	}
	if len(server.service.Snapshot().Reused()) != 0 {
		t.Fatalf("a signal change reported reuse: %v", server.service.Snapshot().Reused())
	}
}

func waitFor(t *testing.T, condition func() bool) {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for !condition() {
		if time.Now().After(deadline) {
			t.Fatal("condition not reached")
		}
		time.Sleep(5 * time.Millisecond)
	}
}

// The shared runtime closes only after the last generation that holds it
// has drained.
func TestSharedSignalRuntimeClosesAfterItsLastGeneration(t *testing.T) {
	document := parityCorpusDocument(t)
	server := newRealRouterServer(t, document)
	shared := server.service.GetRouter().signals
	lease, err := server.service.Pin()
	if err != nil {
		t.Fatal(err)
	}
	reloadDocument(t, server, strings.Replace(document, "127.0.0.1:18000", "127.0.0.1:18100", 1))
	if got := shared.refs.Load(); got != 2 {
		t.Fatalf("shared runtime holds = %d while both generations live, want 2", got)
	}
	lease.Release()
	waitFor(t, func() bool { return shared.refs.Load() == 1 })
	reloadDocument(t, server, strings.Replace(document, `"urgent", "asap"`, `"urgent", "asap", "now"`, 1))
	waitFor(t, func() bool { return shared.refs.Load() == 0 })
}
