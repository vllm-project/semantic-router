package recipe

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
)

func TestValidationProvenanceVerifiesStableActiveRouterHashes(t *testing.T) {
	directory := writeManagedRecipe(t)
	config, err := os.ReadFile(filepath.Join(directory, "config.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	routerHash := strings.TrimPrefix(digestBytes(config), "sha256:")
	server, hashCalls := stableCountingProvenanceRouter(t, routerHash)

	service := NewService(Options{
		Directory:         directory,
		RuntimeConfigPath: filepath.Join(directory, "config.yaml"),
		RouterAPIURL:      server.URL,
		HTTPClient:        server.Client(),
	})
	result, err := service.Validate(context.Background(), "lane-a", "variant-a")
	if err != nil || !result.Passed {
		t.Fatalf("Validate() = %#v, %v", result, err)
	}
	if result.Provenance.Status != ProvenanceVerified || result.Provenance.Reason != "" {
		t.Fatalf("provenance = %#v", result.Provenance)
	}
	if result.Provenance.PackageHash != result.RecipeDigest ||
		result.Provenance.PackageConfigHash != digestBytes(config) ||
		result.Provenance.Before.SourceConfigHash != digestBytes(config) ||
		result.Provenance.Before != result.Provenance.After {
		t.Fatalf("provenance hashes = %#v", result.Provenance)
	}
	if hashCalls.Load() != 2 {
		t.Fatalf("/config/hash calls = %d, want pre and post observations", hashCalls.Load())
	}
}

func stableCountingProvenanceRouter(t *testing.T, routerHash string) (*httptest.Server, *atomic.Int32) {
	t.Helper()
	var hashCalls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/config/hash":
			hashCalls.Add(1)
			_, _ = fmt.Fprintf(w, `{"hash":%q,"runtime_hash":%q,"active_hash":%q,"status":"active"}`, routerHash, routerHash, routerHash)
		case "/api/v1/eval":
			_, _ = w.Write([]byte(successfulManagedRecipeEvalResponse))
		default:
			http.NotFound(w, r)
		}
	}))
	t.Cleanup(server.Close)
	return server, &hashCalls
}

func TestValidationUsesManagedBearerCredentialForEvalAndProvenance(t *testing.T) {
	directory := writeManagedRecipe(t)
	configPath := filepath.Join(directory, "config.yaml")
	config, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatal(err)
	}
	store := NewStore(StoreOptions{Root: filepath.Join(t.TempDir(), "store"), ConfigPath: configPath})
	token, err := store.EnsureManagementCredential()
	if err != nil {
		t.Fatal(err)
	}
	hash := strings.TrimPrefix(digestBytes(config), "sha256:")
	var authenticated atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer "+token {
			http.Error(w, "missing service credential", http.StatusUnauthorized)
			return
		}
		authenticated.Add(1)
		switch r.URL.Path {
		case "/config/hash":
			_, _ = fmt.Fprintf(w, `{"hash":%q,"runtime_hash":%q,"active_hash":%q,"status":"active"}`, hash, hash, hash)
		case "/api/v1/eval":
			_, _ = w.Write([]byte(successfulManagedRecipeEvalResponse))
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()
	service := NewService(Options{
		Directory: directory, Store: store, RuntimeConfigPath: configPath,
		RouterAPIURL: server.URL, HTTPClient: server.Client(),
	})
	result, err := service.Validate(context.Background(), "lane-a", "variant-a")
	if err != nil || !result.Passed || authenticated.Load() != 3 {
		t.Fatalf("Validate()=%#v err=%v authenticated=%d", result, err, authenticated.Load())
	}
}

func TestValidationProvenanceSupportsBoundPlatformRuntimeOverrides(t *testing.T) {
	directory := writeManagedRecipe(t)
	config, err := os.ReadFile(filepath.Join(directory, "config.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	runtime := append(append([]byte(nil), config...), []byte("\n# platform realization\n")...)
	runtimePath := filepath.Join(t.TempDir(), "runtime-config.demo.yaml")
	if writeErr := os.WriteFile(runtimePath, runtime, 0o600); writeErr != nil {
		t.Fatal(writeErr)
	}
	provenancePath := strings.TrimSuffix(runtimePath, filepath.Ext(runtimePath)) + ".provenance.json"
	if writeErr := os.WriteFile(
		provenancePath,
		[]byte(runtimeProvenanceJSON(digestBytes(config), digestBytes(runtime))),
		0o600,
	); writeErr != nil {
		t.Fatal(writeErr)
	}
	sourceHash := strings.TrimPrefix(digestBytes(config), "sha256:")
	runtimeHash := strings.TrimPrefix(digestBytes(runtime), "sha256:")
	server := stableProvenanceRouter(t, sourceHash, runtimeHash, "active", runtimeHash)

	service := NewService(Options{
		Directory:         directory,
		RuntimeConfigPath: runtimePath,
		RouterAPIURL:      server.URL,
		HTTPClient:        server.Client(),
	})
	result, err := service.Validate(context.Background(), "lane-a", "variant-a")
	if err != nil || !result.Passed || result.Provenance.Status != ProvenanceVerified {
		t.Fatalf("Validate() = %#v, %v", result, err)
	}
	if result.Provenance.PackageConfigHash != result.Provenance.Before.SourceConfigHash ||
		result.Provenance.Before.SourceConfigHash == result.Provenance.Before.GeneratedRuntimeHash {
		t.Fatalf("source/runtime distinction = %#v", result.Provenance)
	}
}

func TestBoundRuntimeConfigDigestUsesCommittedPackageActivationReceipt(t *testing.T) {
	directory := copyMaintainedRecipe(t, "accuracy")
	archive := recipeZIPFromDirectory(t, directory, nil)
	archiveServer := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write(archive)
	}))
	defer archiveServer.Close()
	root := t.TempDir()
	runtimePath := filepath.Join(root, "runtime-config.yaml")
	previous := []byte("version: v0.3\n")
	if err := os.WriteFile(runtimePath, previous, 0o600); err != nil {
		t.Fatal(err)
	}
	store := NewStore(StoreOptions{
		Root:       filepath.Join(root, "store"),
		ConfigPath: runtimePath,
		HTTPClient: archiveServer.Client(),
	})
	installed, _, err := store.Import(context.Background(), ImportRequest{URL: archiveServer.URL + "/accuracy.zip"})
	if err != nil {
		t.Fatal(err)
	}
	target, rawConfig, err := store.ActivationTarget(installed.RecipeDigest, true)
	if err != nil {
		t.Fatal(err)
	}
	realized := append(append([]byte(nil), rawConfig...), []byte("\n# platform realization\n")...)
	transaction, err := store.BeginActivation(target.RecipeDigest, previous)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(runtimePath, realized, 0o600); err != nil {
		t.Fatal(err)
	}
	if err := store.CommitActivation(transaction, ActivePointer{
		RecipeDigest:         target.RecipeDigest,
		ConfigDigest:         target.ConfigDigest,
		RealizedConfigDigest: digestBytes(realized),
	}); err != nil {
		t.Fatal(err)
	}
	service := NewService(Options{Store: store, RuntimeConfigPath: runtimePath})
	if got, err := service.boundRuntimeConfigDigest(rawConfig); err != nil || got != digestBytes(realized) {
		t.Fatalf("boundRuntimeConfigDigest() = %q, %v", got, err)
	}
}

func TestValidationPassDoesNotConflatePendingRuntimeProvenance(t *testing.T) {
	directory := writeManagedRecipe(t)
	config, err := os.ReadFile(filepath.Join(directory, "config.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	routerHash := strings.TrimPrefix(digestBytes(config), "sha256:")
	server := stableProvenanceRouter(t, routerHash, routerHash, "pending", strings.Repeat("f", 64))
	service := NewService(Options{
		Directory:         directory,
		RuntimeConfigPath: filepath.Join(directory, "config.yaml"),
		RouterAPIURL:      server.URL,
		HTTPClient:        server.Client(),
	})
	result, err := service.Validate(context.Background(), "lane-a", "variant-a")
	if err != nil || !result.Passed {
		t.Fatalf("routing assertions should pass independently: result=%#v err=%v", result, err)
	}
	if result.Provenance.Status != ProvenanceUnverified || result.Provenance.Reason != "router_runtime_not_active" {
		t.Fatalf("provenance = %#v", result.Provenance)
	}
}

func TestValidationProvenanceDetectsRouterChangeDuringEval(t *testing.T) {
	directory := writeManagedRecipe(t)
	config, err := os.ReadFile(filepath.Join(directory, "config.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	first := strings.TrimPrefix(digestBytes(config), "sha256:")
	second := strings.Repeat("b", 64)
	var hashCalls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/config/hash":
			hash := first
			if hashCalls.Add(1) > 1 {
				hash = second
			}
			_, _ = fmt.Fprintf(w, `{"hash":%q,"runtime_hash":%q,"active_hash":%q,"status":"active"}`, hash, hash, hash)
		case "/api/v1/eval":
			_, _ = w.Write([]byte(successfulManagedRecipeEvalResponse))
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()
	service := NewService(Options{
		Directory:         directory,
		RuntimeConfigPath: filepath.Join(directory, "config.yaml"),
		RouterAPIURL:      server.URL,
		HTTPClient:        server.Client(),
	})
	result, err := service.Validate(context.Background(), "lane-a", "variant-a")
	if err != nil || !result.Passed {
		t.Fatalf("Validate() = %#v, %v", result, err)
	}
	if result.Provenance.Status != ProvenanceUnverified || result.Provenance.Reason != "router_config_changed_during_validation" {
		t.Fatalf("provenance = %#v", result.Provenance)
	}
}

func TestValidationProvenanceIsUnverifiedWithOlderRouter(t *testing.T) {
	directory := writeManagedRecipe(t)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/v1/eval" {
			_, _ = w.Write([]byte(successfulManagedRecipeEvalResponse))
			return
		}
		http.NotFound(w, r)
	}))
	defer server.Close()
	service := NewService(Options{
		Directory:         directory,
		RuntimeConfigPath: filepath.Join(directory, "config.yaml"),
		RouterAPIURL:      server.URL,
		HTTPClient:        server.Client(),
	})
	result, err := service.Validate(context.Background(), "lane-a", "variant-a")
	if err != nil || !result.Passed {
		t.Fatalf("Validate() = %#v, %v", result, err)
	}
	if result.Provenance.Status != ProvenanceUnverified || result.Provenance.Reason != "before_observation_unavailable" {
		t.Fatalf("provenance = %#v", result.Provenance)
	}
}

func TestConfigHashObservationDoesNotFollowRedirect(t *testing.T) {
	var forwarded atomic.Bool
	target := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		forwarded.Store(true)
	}))
	defer target.Close()
	redirect := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Location", target.URL)
		w.WriteHeader(http.StatusTemporaryRedirect)
	}))
	defer redirect.Close()
	evaluator := &HTTPRouterEvaluator{BaseURL: redirect.URL, Client: redirect.Client()}
	if _, err := evaluator.ObserveConfigHash(context.Background()); err == nil {
		t.Fatal("ObserveConfigHash() unexpectedly accepted a redirect")
	}
	if forwarded.Load() {
		t.Fatal("Router config identity request was forwarded across a redirect")
	}
}

func stableProvenanceRouter(t *testing.T, sourceHash, runtimeHash, status, activeHash string) *httptest.Server {
	t.Helper()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/config/hash":
			_, _ = fmt.Fprintf(w, `{"hash":%q,"runtime_hash":%q,"active_hash":%q,"status":%q}`, sourceHash, runtimeHash, activeHash, status)
		case "/api/v1/eval":
			_, _ = w.Write([]byte(successfulManagedRecipeEvalResponse))
		default:
			http.NotFound(w, r)
		}
	}))
	t.Cleanup(server.Close)
	return server
}

const successfulManagedRecipeEvalResponse = `{
  "requested_model":"vllm-sr/mom-test-v1",
  "selected_model":"leaf-a",
  "selection_status":"selected",
  "selection_method":"static",
  "recipe":"balanced",
  "routing_decision":"decision-a",
  "decision_result":{
    "decision_name":"decision-a",
    "algorithm":"static",
    "plugins":["tools"],
    "matched_signals":{"keywords":["signal-a"]}
  },
  "recommended_models":["leaf-a"],
  "eval_trace":[{"decision_name":"decision-a","matched":true}]
}`
