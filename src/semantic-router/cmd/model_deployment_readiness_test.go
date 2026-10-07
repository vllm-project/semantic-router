package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	healthpb "google.golang.org/grpc/health/grpc_health_v1"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/configsnapshot"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice/runtimetest"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/startupstatus"
)

// The test binary also acts as the Router-managed runtime process: with
// fakeRuntimeEnv set, "<binary> serve --models FILE --uds PATH ..." serves the
// fake contract on the socket, every model answering decisions. Until the
// file fakeRuntimeHoldEnv names exists, its models report loading; the models
// fakeRuntimeFailEnv lists (comma separated) fail to load.
const (
	fakeRuntimeEnv     = "ROUTER_CMD_FAKE_RUNTIME"
	fakeRuntimeHoldEnv = "ROUTER_CMD_FAKE_RUNTIME_HOLD"
	fakeRuntimeFailEnv = "ROUTER_CMD_FAKE_RUNTIME_FAIL"
)

func TestMain(m *testing.M) {
	if os.Getenv(fakeRuntimeEnv) == "1" {
		serveFakeRuntime(os.Args[1:])
		return
	}
	os.Exit(m.Run())
}

func serveFakeRuntime(args []string) {
	socket, modelsFile := "", ""
	for index := 0; index+1 < len(args); index++ {
		switch args[index] {
		case "--uds":
			socket = args[index+1]
		case "--models":
			modelsFile = args[index+1]
		}
	}
	data, err := os.ReadFile(modelsFile)
	if err != nil {
		os.Exit(2)
	}
	var document struct {
		Models []struct {
			Name string `json:"name"`
		} `json:"models"`
	}
	if json.Unmarshal(data, &document) != nil {
		os.Exit(2)
	}
	listener, err := net.Listen("unix", socket)
	if err != nil {
		os.Exit(2)
	}
	failing := strings.Split(os.Getenv(fakeRuntimeFailEnv), ",")
	models := make([]runtimetest.Model, 0, len(document.Models))
	var held []string
	for _, entry := range document.Models {
		models = append(models, runtimetest.Model{ID: entry.Name})
		if !slices.Contains(failing, entry.Name) {
			held = append(held, entry.Name)
		}
	}
	fake := runtimetest.New(models...)
	for _, entry := range document.Models {
		if slices.Contains(failing, entry.Name) {
			fake.SetFailed(entry.Name, "fake weights are missing")
		}
	}
	if hold := os.Getenv(fakeRuntimeHoldEnv); hold != "" {
		for _, name := range held {
			fake.SetReady(name, false)
		}
		go func() {
			for {
				if _, statErr := os.Stat(hold); statErr == nil {
					for _, name := range held {
						fake.SetReady(name, true)
					}
					return
				}
				time.Sleep(20 * time.Millisecond)
			}
		}()
	}
	_ = http.Serve(listener, fake.Handler())
}

// useFakeRuntime makes this test binary the managed runtime command; its
// models load once the returned hold file exists.
func useFakeRuntime(t *testing.T) (hold string) {
	t.Helper()
	binary, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	runtimeDir, err := os.MkdirTemp("", "vsr-run-")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = os.RemoveAll(runtimeDir) })
	hold = filepath.Join(t.TempDir(), "models-loaded")
	t.Setenv(fakeRuntimeEnv, "1")
	t.Setenv(fakeRuntimeHoldEnv, hold)
	t.Setenv(modelservice.RuntimeCommandEnv, binary)
	t.Setenv(modelservice.RuntimeDirEnv, runtimeDir)
	t.Setenv(configsnapshot.HistoryDirEnv, t.TempDir())
	return hold
}

type readinessPorts struct {
	api, listener, extproc int
}

// readinessDocument routes to model-route when the decision model behind the
// "hard" signal answers (the fake answers Noul 0.8) and to default-route when
// the signal is unknown. deployments are the model_runtime deployments; the
// first one answers the signal.
func readinessDocument(ports readinessPorts, backend string, deployments ...string) string {
	var signals, decision, catalog strings.Builder
	for index, deployment := range deployments {
		fmt.Fprintf(&catalog, "      %s:\n        provider: model_runtime\n        artifact: /opt/vsr-test/%s\n        device: cpu\n        process: %s\n",
			deployment, deployment, deployment)
		fmt.Fprintf(&signals, "      - name: hard_%d\n        deployment: %s\n        question: {type: noul, instructions: \"Is this request hard?\"}\n        predicate: {gte: 0.5}\n",
			index, deployment)
	}
	if len(deployments) > 0 {
		decision.WriteString(`    - name: model-route
      priority: 200
      rules:
        operator: AND
        on_unknown: no_match
        conditions: [{type: decision, name: hard_0}]
      modelRefs: [{model: a}]
`)
	}
	document := fmt.Sprintf(`version: v0.3
listeners:
  - name: http
    address: 127.0.0.1
    port: %d
providers:
  defaults:
    model: a
  models:
    - name: a
      backend_refs: [{endpoint: %s, protocol: http, provider: vllm}]
routing:
  modelCards:
    - name: a
`, ports.listener, strings.TrimPrefix(backend, "http://"))
	if signals.Len() > 0 {
		document += "  signals:\n    decision:\n" + signals.String()
	}
	document += "  decisions:\n" + decision.String() + `    - name: default-route
      priority: 100
      rules: {operator: AND, conditions: []}
      modelRefs: [{model: a}]
global:
  services:
    management_api:
      bind_address: 127.0.0.1
      port: ` + fmt.Sprint(ports.api) + `
      remote_exposure: false
      auth:
        mode: disabled
    observability:
      tracing:
        enabled: false
  stores:
    semantic_cache:
      enabled: false
`
	if catalog.Len() > 0 {
		document += "  model_catalog:\n    deployments:\n" + catalog.String()
	}
	return document
}

func completionBackend(t *testing.T) *httptest.Server {
	t.Helper()
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"c","object":"chat.completion","model":"a","choices":[{"index":0,`+
			`"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],`+
			`"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`)
	}))
	t.Cleanup(backend.Close)
	return backend
}

// routerProcess is a Router run as main runs it; err is set once exited closes.
type routerProcess struct {
	exited chan struct{}
	err    error
}

// startRouterProcess runs the Router until the test ends.
func startRouterProcess(t *testing.T, configPath string, gateway config.GatewayMode, ports readinessPorts) *routerProcess {
	t.Helper()
	ctx, cancel := context.WithCancel(context.Background())
	process := &routerProcess{exited: make(chan struct{})}
	go func() {
		process.err = runRouterProcess(ctx, runtimeOptions{
			configPath: configPath, port: ports.extproc, apiPort: ports.api, apiBind: "127.0.0.1",
			enableAPI: true, gateway: gateway,
		})
		close(process.exited)
	}()
	t.Cleanup(func() {
		cancel()
		select {
		case <-process.exited:
		case <-time.After(30 * time.Second):
			t.Error("the Router did not stop")
		}
	})
	return process
}

func getJSON(t *testing.T, url string, into interface{}) int {
	t.Helper()
	request, err := http.NewRequestWithContext(context.Background(), http.MethodGet, url, nil)
	if err != nil {
		t.Fatal(err)
	}
	response, err := http.DefaultClient.Do(request)
	if err != nil {
		return 0
	}
	defer response.Body.Close()
	body, _ := io.ReadAll(response.Body)
	if into != nil {
		if err := json.Unmarshal(body, into); err != nil {
			t.Fatalf("GET %s: %v: %s", url, err, body)
		}
	}
	return response.StatusCode
}

// eventually polls check until it returns nil, failing the test with its last
// error after the deadline or when the Router exits.
func eventually(t *testing.T, process *routerProcess, within time.Duration, check func() error) {
	t.Helper()
	deadline := time.Now().Add(within)
	for {
		err := check()
		if err == nil {
			return
		}
		select {
		case <-process.exited:
			t.Fatalf("the Router exited (%v) while waiting: %v", process.err, err)
		default:
		}
		if time.Now().After(deadline) {
			t.Fatal(err)
		}
		time.Sleep(25 * time.Millisecond)
	}
}

type readyBody struct {
	Status        string   `json:"status"`
	Ready         bool     `json:"ready"`
	Phase         string   `json:"phase"`
	Message       string   `json:"message"`
	PendingModels []string `json:"pending_models"`
	ReadyModels   int      `json:"ready_models"`
	TotalModels   int      `json:"total_models"`
}

type previewBody struct {
	RoutingDecision string            `json:"routing_decision"`
	SignalErrors    map[string]string `json:"signal_errors"`
}

func previewRoute(t *testing.T, api string) (previewBody, error) {
	t.Helper()
	response, err := http.Post(api+"/api/v1/routing/preview", "application/json",
		strings.NewReader(`{"model":"vllm-sr/auto","messages":[{"role":"user","content":"Prove that there are infinitely many primes."}]}`))
	if err != nil {
		return previewBody{}, err
	}
	defer response.Body.Close()
	body, _ := io.ReadAll(response.Body)
	var preview previewBody
	if response.StatusCode != http.StatusOK || json.Unmarshal(body, &preview) != nil {
		return previewBody{}, fmt.Errorf("routing preview %d: %s", response.StatusCode, body)
	}
	return preview, nil
}

func refused(address string) bool {
	connection, err := net.DialTimeout("tcp", address, 100*time.Millisecond)
	if err != nil {
		return true
	}
	_ = connection.Close()
	return false
}

// While its Router-managed decision model loads, the Router is alive but not
// ready, names the deployment it waits for, and does not serve; once the model
// is ready, the first request routes on its answer. Both gateway modes.
func TestStartupWaitsForTheRouterManagedModelDeployments(t *testing.T) {
	for _, gateway := range []config.GatewayMode{config.GatewayStandalone, config.GatewayExtProc} {
		t.Run(string(gateway), func(t *testing.T) {
			hold := useFakeRuntime(t)
			ports := readinessPorts{api: freePort(t), listener: freePort(t), extproc: freePort(t)}
			configPath := filepath.Join(t.TempDir(), "config.yaml")
			if err := os.WriteFile(configPath, []byte(readinessDocument(ports, completionBackend(t).URL, "decider")), 0o600); err != nil {
				t.Fatal(err)
			}
			process := startRouterProcess(t, configPath, gateway, ports)
			api := fmt.Sprintf("http://127.0.0.1:%d", ports.api)
			served := fmt.Sprintf("127.0.0.1:%d", ports.listener)
			if gateway == config.GatewayExtProc {
				served = fmt.Sprintf("127.0.0.1:%d", ports.extproc)
			}

			var loading startupstatus.State
			eventually(t, process, 20*time.Second, func() error {
				if getJSON(t, api+"/startup-status", &loading) != http.StatusServiceUnavailable || loading.Phase != startupstatus.PhaseLoadingModelDeployments {
					return fmt.Errorf("startup does not report the deployment it waits for: %+v", loading)
				}
				return nil
			})
			deployments := loading.ModelDeployments
			if len(deployments) != 1 || deployments[0].Name != "decider" || deployments[0].Ready ||
				deployments[0].Artifact != "/opt/vsr-test/decider" || deployments[0].Process != "decider" ||
				loading.TotalModels != 1 || loading.ReadyModels != 0 || !slices.Equal(loading.PendingModels, []string{"decider"}) ||
				!strings.Contains(loading.Message, "decider") {
				t.Fatalf("startup status while the model loads: %+v", loading)
			}
			for range 10 {
				var ready readyBody
				if status := getJSON(t, api+"/ready", &ready); status != http.StatusServiceUnavailable || ready.Ready ||
					ready.Phase != startupstatus.PhaseLoadingModelDeployments || !slices.Equal(ready.PendingModels, []string{"decider"}) ||
					ready.TotalModels != 1 {
					t.Fatalf("/ready while the model loads: %d %+v", status, ready)
				}
				if status := getJSON(t, api+"/health", nil); status != http.StatusOK {
					t.Fatalf("/health is liveness and answers while the model loads, got %d", status)
				}
				if !refused(served) {
					t.Fatalf("%s serves %s before its model is ready", gateway, served)
				}
				time.Sleep(50 * time.Millisecond)
			}

			if err := os.WriteFile(hold, nil, 0o600); err != nil {
				t.Fatal(err)
			}
			var ready readyBody
			eventually(t, process, 20*time.Second, func() error {
				if status := getJSON(t, api+"/ready", &ready); status != http.StatusOK || !ready.Ready {
					return fmt.Errorf("/ready after the model loaded: %d %+v", status, ready)
				}
				return nil
			})
			if ready.TotalModels != 1 || ready.ReadyModels != 1 || len(ready.PendingModels) != 0 {
				t.Fatalf("/ready counts the ready deployment: %+v", ready)
			}
			var status startupstatus.State
			if getJSON(t, api+"/startup-status", &status); status.Phase != "ready" || len(status.ModelDeployments) != 1 ||
				!status.ModelDeployments[0].Ready || status.ModelDeployments[0].State != "ready" {
				t.Fatalf("startup status lists the ready deployment: %+v", status)
			}
			preview, err := previewRoute(t, api)
			if err != nil {
				t.Fatal(err)
			}
			if preview.RoutingDecision != "model-route" || len(preview.SignalErrors) != 0 {
				t.Fatalf("the first request after ready routes on the model's answer: %+v", preview)
			}
			if gateway == config.GatewayExtProc {
				requireGRPCServing(t, served)
				return
			}
			requireRoutedDecision(t, "http://"+served, "model-route")
		})
	}
}

func requireGRPCServing(t *testing.T, address string) {
	t.Helper()
	connection, err := grpc.NewClient(address, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatal(err)
	}
	defer connection.Close()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	response, err := healthpb.NewHealthClient(connection).Check(ctx, &healthpb.HealthCheckRequest{})
	if err != nil || response.GetStatus() != healthpb.HealthCheckResponse_SERVING {
		t.Fatalf("ext_proc gRPC health once the model is ready: %v %v", response, err)
	}
}

func requireRoutedDecision(t *testing.T, listener, decision string) {
	t.Helper()
	var ready readyBody
	if status := getJSON(t, listener+"/ready", &ready); status != http.StatusOK {
		t.Fatalf("the standalone listener's /ready once the model is ready: %d %+v", status, ready)
	}
	response, err := http.Post(listener+"/v1/chat/completions", "application/json",
		strings.NewReader(`{"model":"vllm-sr/auto","messages":[{"role":"user","content":"Prove that there are infinitely many primes."}]}`))
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()
	_, _ = io.Copy(io.Discard, response.Body)
	if response.StatusCode != http.StatusOK || response.Header.Get("x-vsr-selected-decision") != decision {
		t.Fatalf("routed request: %d decision %q, want %q", response.StatusCode, response.Header.Get("x-vsr-selected-decision"), decision)
	}
}

// A Router-managed deployment that cannot load stops the start with its
// reason, instead of serving on unknown answers.
func TestStartupFailsWhenARouterManagedModelCannotLoad(t *testing.T) {
	hold := useFakeRuntime(t)
	if err := os.WriteFile(hold, nil, 0o600); err != nil {
		t.Fatal(err)
	}
	t.Setenv(fakeRuntimeFailEnv, "decider")
	ports := readinessPorts{api: freePort(t), listener: freePort(t), extproc: freePort(t)}
	dir := t.TempDir()
	configPath := filepath.Join(dir, "config.yaml")
	if err := os.WriteFile(configPath, []byte(readinessDocument(ports, completionBackend(t).URL, "decider")), 0o600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("VLLM_SR_RUNTIME_STATUS_DIR", dir)
	process := startRouterProcess(t, configPath, config.GatewayStandalone, ports)
	select {
	case <-process.exited:
		if err := process.err; err == nil || !strings.Contains(err.Error(), `model_runtime deployment "decider"`) ||
			!strings.Contains(err.Error(), "fake weights are missing") {
			t.Fatalf("startup error = %v, want the deployment and its load failure", err)
		}
	case <-time.After(60 * time.Second):
		t.Fatal("startup kept waiting for a model that fails to load")
	}
	state, err := startupstatus.Load(filepath.Join(dir, "router-runtime.json"))
	if err != nil || state.Phase != "error" || state.Ready || !strings.Contains(state.Message, `"decider"`) {
		t.Fatalf("the startup status reports the failure: %+v %v", state, err)
	}
}

type configHashBody struct {
	ActivationStatus string `json:"activation_status"`
	LastRejection    *struct {
		Status string `json:"status"`
		Error  string `json:"error"`
	} `json:"last_rejection"`
}

// A reload that adds a Router-managed deployment keeps the Router ready and
// serving the previous configuration until the new model is ready; one whose
// new model fails to load is rejected, and the previous configuration serves on.
func TestReloadKeepsServingUntilAnAddedManagedModelIsReady(t *testing.T) {
	hold := useFakeRuntime(t)
	backend := completionBackend(t).URL
	ports := readinessPorts{api: freePort(t), listener: freePort(t), extproc: freePort(t)}
	configPath := filepath.Join(t.TempDir(), "config.yaml")
	if err := os.WriteFile(configPath, []byte(readinessDocument(ports, backend)), 0o600); err != nil {
		t.Fatal(err)
	}
	process := startRouterProcess(t, configPath, config.GatewayStandalone, ports)
	api := fmt.Sprintf("http://127.0.0.1:%d", ports.api)
	listener := fmt.Sprintf("http://127.0.0.1:%d", ports.listener)
	eventually(t, process, 20*time.Second, func() error {
		if status := getJSON(t, api+"/ready", nil); status != http.StatusOK {
			return fmt.Errorf("a configuration without managed models is ready at once, /ready %d", status)
		}
		return nil
	})

	if err := os.WriteFile(configPath, []byte(readinessDocument(ports, backend, "decider")), 0o600); err != nil {
		t.Fatal(err)
	}
	var hash configHashBody
	eventually(t, process, 20*time.Second, func() error {
		if getJSON(t, api+"/api/v1/config/hash", &hash); hash.ActivationStatus != "pending" {
			return fmt.Errorf("the reload is not waiting for its model: %+v", hash)
		}
		return nil
	})
	for range 10 {
		if status := getJSON(t, api+"/ready", nil); status != http.StatusOK {
			t.Fatalf("a serving Router stays ready while a reload's model loads, /ready %d", status)
		}
		requireRoutedDecision(t, listener, "default-route")
		time.Sleep(50 * time.Millisecond)
	}
	if err := os.WriteFile(hold, nil, 0o600); err != nil {
		t.Fatal(err)
	}
	eventually(t, process, 20*time.Second, func() error {
		if getJSON(t, api+"/api/v1/config/hash", &hash); hash.ActivationStatus != "active" {
			return fmt.Errorf("the reload did not activate once its model loaded: %+v", hash)
		}
		return nil
	})
	requireRoutedDecision(t, listener, "model-route")

	t.Setenv(fakeRuntimeFailEnv, "broken")
	if err := os.WriteFile(configPath, []byte(readinessDocument(ports, backend, "decider", "broken")), 0o600); err != nil {
		t.Fatal(err)
	}
	eventually(t, process, 60*time.Second, func() error {
		getJSON(t, api+"/api/v1/config/hash", &hash)
		if hash.ActivationStatus != "failed" || hash.LastRejection == nil || !strings.Contains(hash.LastRejection.Error, `model_runtime deployment "broken"`) {
			return fmt.Errorf("a reload whose model cannot load is rejected with the deployment: %+v", hash)
		}
		return nil
	})
	if status := getJSON(t, api+"/ready", nil); status != http.StatusOK {
		t.Fatalf("a rejected reload leaves the Router ready, /ready %d", status)
	}
	requireRoutedDecision(t, listener, "model-route")
}
