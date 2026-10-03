package modelservice

import (
	"context"
	"encoding/json"
	"errors"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice/runtimetest"
)

const fakeRuntimeEnv = "MODELSERVICE_FAKE_RUNTIME"

// TestMain lets the test binary act as a managed runtime process:
// <binary> serve --models FILE --uds PATH serves the fake contract on the socket.
func TestMain(m *testing.M) {
	if os.Getenv(fakeRuntimeEnv) == "1" {
		serveManagedFake(os.Args[1:])
		return
	}
	os.Exit(m.Run())
}

func serveManagedFake(args []string) {
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
		Models []modelEntry `json:"models"`
	}
	if json.Unmarshal(data, &document) != nil {
		os.Exit(2)
	}
	listener, err := net.Listen("unix", socket)
	if err != nil {
		os.Exit(2)
	}
	_ = http.Serve(listener, fakeModels(document.Models).Handler())
}

func sampleRequest(state string) Request {
	return Request{State: state, Questions: []Question{
		{ID: "reasoning", Type: "noul", Instructions: "Hard?"},
		{ID: "kind", Type: "choice", Instructions: "Kind?", Choices: []Choice{{Key: "code", Description: "Code"}, {Key: "math", Description: "Math"}}},
	}}
}

func TestClientDecideOverUnixSocket(t *testing.T) {
	socket := filepath.Join(t.TempDir(), "runtime.sock")
	listener, err := net.Listen("unix", socket)
	if err != nil {
		t.Fatal(err)
	}
	server := &http.Server{Handler: fakeModels([]modelEntry{{Name: "kai", Model: "vllm-sr/Decision-2.0-Kai-0.6B"}}).Handler(), ReadHeaderTimeout: time.Second}
	go func() { _ = server.Serve(listener) }()
	defer server.Close()
	client, err := NewClient("unix://" + socket)
	if err != nil {
		t.Fatal(err)
	}
	request := sampleRequest("text")
	request.Model = "kai"
	response, err := client.Decide(context.Background(), request)
	if err != nil {
		t.Fatal(err)
	}
	if got := response.Answers["reasoning"]; got.Type != "noul" || got.Noul != 0.8 {
		t.Fatalf("noul answer = %+v", got)
	}
	if got := response.Answers["kind"]; got.Choice != "code" || got.Probabilities["code"] != 0.7 {
		t.Fatalf("choice answer = %+v", got)
	}
}

func TestPlanProcessesGroupsByDeviceAndProcessKey(t *testing.T) {
	deployments := map[string]config.ModelDeployment{
		"domain":  {Provider: config.ModelRuntimeProvider, Artifact: "vllm-sr/Vela-1.0-Encoder-307M-Domain", Device: "cpu"},
		"pii":     {Provider: config.ModelRuntimeProvider, Artifact: "vllm-sr/Vela-1.0-Encoder-307M-PII", Device: "cpu"},
		"domain2": {Provider: config.ModelRuntimeProvider, Artifact: "vllm-sr/Vela-1.0-Encoder-307M-Domain", Device: "cpu"},
		"kai":     {Provider: config.ModelRuntimeProvider, Artifact: "vllm-sr/Decision-2.0-Kai-0.6B", Device: "rocm:0"},
		"guard":   {Provider: config.ModelRuntimeProvider, Artifact: "vllm-sr/Vela-1.0-Encoder-307M-Guard", Device: "cpu", Process: "safety"},
		"remote":  {Provider: config.ModelRuntimeProvider, Endpoint: "http://shared:8100", ServedName: "vela-domain"},
		"remote2": {Provider: config.ModelRuntimeProvider, Endpoint: "http://shared:8100"},
	}
	plans := planProcesses(deployments, []string{"vllm-sr-runtime"}, "")
	byName := map[string]*processPlan{}
	for _, plan := range plans {
		byName[plan.name] = plan
	}
	if len(plans) != 4 || byName["cpu"] == nil || byName["rocm:0"] == nil || byName["safety"] == nil || byName["attached"] == nil {
		t.Fatalf("plans = %+v", plans)
	}
	cpu := byName["cpu"]
	if len(cpu.models) != 2 || cpu.members["domain2"] != "domain" || cpu.members["pii"] != "pii" {
		t.Fatalf("identical models share one entry in a process: %+v %+v", cpu.models, cpu.members)
	}
	if attached := byName["attached"]; attached.members["remote"] != "vela-domain" || attached.members["remote2"] != "remote2" {
		t.Fatalf("attached members select by served name: %+v", attached.members)
	}
	again := planProcesses(deployments, []string{"vllm-sr-runtime"}, "")
	if again[0].key != plans[0].key {
		t.Fatal("the same composition must keep its key so generations share the process")
	}
	changed := planProcesses(deployments, []string{"vllm-sr-runtime"}, "/cache")
	for _, plan := range changed {
		if plan.name == "cpu" && plan.key == cpu.key {
			t.Fatal("a changed composition must start a new process")
		}
	}
}

func TestManagedCommandAndModelsFile(t *testing.T) {
	got := strings.Join(managedCommand([]string{"vllm-sr-runtime"}, "/run/x.sock", "/run/x.models.json", "/cache"), " ")
	if want := "vllm-sr-runtime serve --models /run/x.models.json --uds /run/x.sock --cache-dir /cache"; got != want {
		t.Fatalf("command\n got %s\nwant %s", got, want)
	}
	plan := planProcesses(map[string]config.ModelDeployment{
		"kai": {Provider: config.ModelRuntimeProvider, Artifact: "vllm-sr/Decision-2.0-Kai-0.6B", Revision: strings.Repeat("b", 40), Device: "cpu", Profile: "batching"},
	}, nil, "")[0]
	path := filepath.Join(t.TempDir(), "models.json")
	if err := plan.writeModelsFile(path); err != nil {
		t.Fatal(err)
	}
	info, _ := os.Stat(path)
	data, _ := os.ReadFile(path)
	if info.Mode().Perm() != 0o600 || !strings.Contains(string(data), `"profile": "batching"`) || !strings.Contains(string(data), `"name": "kai"`) {
		t.Fatalf("models file (%v): %s", info.Mode(), data)
	}
}

func runtimeConfig(deployments map[string]config.ModelDeployment) *config.RouterConfig {
	cfg := &config.RouterConfig{}
	cfg.ModelDeployments = deployments
	for name := range deployments {
		if name == "unused" {
			continue
		}
		cfg.DecisionRules = append(cfg.DecisionRules, config.DecisionSignalRule{
			Name: "hard_" + name, Deployment: name, Question: config.DecisionQuestion{Type: "noul", Instructions: "Hard?"},
		})
	}
	return cfg
}

func waitReady(t *testing.T, lease *Lease, name string) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	if _, err := lease.Card(ctx, name); err != nil {
		t.Fatalf("deployment %s never became ready: %v %+v", name, err, lease.Statuses())
	}
}

func TestLeaseFailsOpenUntilTheAttachedModelIsReady(t *testing.T) {
	runtime := fakeModels([]modelEntry{{Name: "decider", Model: "vllm-sr/Decision-2.0-Kai-0.6B"}})
	runtime.SetReady("decider", false)
	server := httptest.NewServer(runtime.Handler())
	defer server.Close()
	manager := NewManager()
	defer func() { _ = manager.Shutdown(context.Background()) }()
	lease, err := manager.Acquire(runtimeConfig(map[string]config.ModelDeployment{
		"decider": {Provider: config.ModelRuntimeProvider, Endpoint: server.URL},
		"unused":  {Provider: config.ModelRuntimeProvider, Artifact: "acme/unused"},
	}))
	if err != nil {
		t.Fatal(err)
	}
	if names := lease.Deployments(); len(names) != 1 || names[0] != "decider" {
		t.Fatalf("only referenced deployments are leased: %v", names)
	}
	if _, err := lease.Decide(context.Background(), "decider", sampleRequest("text")); !errors.Is(err, ErrUnavailable) {
		t.Fatalf("a not-ready deployment must fail open at once, got %v", err)
	}
	if runtime.Calls("decisions") != 0 {
		t.Fatal("a not-ready deployment must not be called")
	}
	short, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()
	if _, err := lease.Card(short, "decider"); !errors.Is(err, ErrUnavailable) {
		t.Fatalf("Card waits for readiness until its deadline, got %v", err)
	}
	runtime.SetReady("decider", true)
	waitReady(t, lease, "decider")
	if _, err := lease.Decide(context.Background(), "decider", sampleRequest("text")); err != nil {
		t.Fatal(err)
	}
	if _, err := lease.Decide(context.Background(), "missing", sampleRequest("text")); !errors.Is(err, ErrUnknownDeployment) {
		t.Fatalf("unknown deployment error = %v", err)
	}
}

func TestManagerSharesProcessesByCompositionAndSupervisesThem(t *testing.T) {
	binary, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	t.Setenv(fakeRuntimeEnv, "1")
	t.Setenv(RuntimeCommandEnv, binary)
	t.Setenv(RuntimeDirEnv, filepath.Join(t.TempDir(), "run"))
	manager := NewManager()
	defer func() { _ = manager.Shutdown(context.Background()) }()
	deployments := map[string]config.ModelDeployment{
		"kai": {Provider: config.ModelRuntimeProvider, Artifact: "vllm-sr/Decision-2.0-Kai-0.6B", Device: "cpu"},
		"eos": {Provider: config.ModelRuntimeProvider, Artifact: "vllm-sr/Decision-2.0-Eos-0.8B", Device: "cpu"},
	}
	if reconcileErr := manager.Reconcile(runtimeConfig(deployments)); reconcileErr != nil {
		t.Fatal(reconcileErr)
	}
	generation, err := manager.Acquire(runtimeConfig(deployments))
	if err != nil {
		t.Fatal(err)
	}
	waitReady(t, generation, "kai")
	waitReady(t, generation, "eos")
	statuses := generation.Statuses()
	if len(statuses) != 2 || statuses[0].Endpoint != statuses[1].Endpoint || !statuses[0].Managed || statuses[0].Process != "cpu" {
		t.Fatalf("two models on one device share one managed process: %+v", statuses)
	}
	info, err := os.Stat(filepath.Dir(strings.TrimPrefix(statuses[0].Endpoint, "unix://")))
	if err != nil || info.Mode().Perm() != 0o700 {
		t.Fatalf("socket directory must be private: %v %v", info, err)
	}
	if len(manager.groups) != 1 {
		t.Fatalf("leases of the same composition share the process: %d processes", len(manager.groups))
	}
	if _, err := generation.Decide(context.Background(), "eos", sampleRequest("text")); err != nil {
		t.Fatal(err)
	}
	// A new composition starts a new process; the old one serves until its last lease closes.
	changed := map[string]config.ModelDeployment{"kai": deployments["kai"]}
	if err := manager.Reconcile(runtimeConfig(changed)); err != nil {
		t.Fatal(err)
	}
	if len(manager.groups) != 2 {
		t.Fatalf("the previous generation keeps its process: %d processes", len(manager.groups))
	}
	if _, err := generation.Decide(context.Background(), "eos", sampleRequest("text")); err != nil {
		t.Fatalf("the previous generation must keep serving: %v", err)
	}
	_ = generation.Close()
	if len(manager.groups) != 1 {
		t.Fatalf("closing the last lease stops its process: %d processes", len(manager.groups))
	}
}

func TestCardFailsFastWhenTheRuntimeCommandCannotRun(t *testing.T) {
	t.Setenv(RuntimeCommandEnv, filepath.Join(t.TempDir(), "missing-runtime"))
	t.Setenv(RuntimeDirEnv, filepath.Join(t.TempDir(), "run"))
	manager := NewManager()
	defer func() { _ = manager.Shutdown(context.Background()) }()
	lease, err := manager.Acquire(runtimeConfig(map[string]config.ModelDeployment{
		"kai": {Provider: config.ModelRuntimeProvider, Artifact: "vllm-sr/Decision-2.0-Kai-0.6B"},
	}))
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	started := time.Now()
	if _, err := lease.Card(ctx, "kai"); !errors.Is(err, ErrUnavailable) || !strings.Contains(err.Error(), "cannot run") {
		t.Fatalf("a missing runtime command must fail preparation, got %v", err)
	}
	if time.Since(started) > 5*time.Second {
		t.Fatal("the failure must not wait for the deadline")
	}
}

func TestDefaultDeciderFailsOpenBeforeStartup(t *testing.T) {
	if _, err := (unavailable{}).Decide(context.Background(), "x", Request{}); !errors.Is(err, ErrUnavailable) {
		t.Fatal("the default decider must answer unavailable")
	}
	if _, err := NewManager().Decide(context.Background(), "x", Request{}); !errors.Is(err, ErrUnavailable) {
		t.Fatal("a manager without a published configuration must answer unavailable")
	}
}

func TestEncodeQuestionSendsNullForMissingDescriptions(t *testing.T) {
	encoded, err := json.Marshal(encodeQuestion(Question{
		Type: "choice", Instructions: "Pick",
		Choices: []Choice{{Key: "a", Description: "First"}, {Key: "b"}},
	}))
	if err != nil {
		t.Fatal(err)
	}
	var decoded struct {
		Choices []map[string]interface{} `json:"choices"`
	}
	if err := json.Unmarshal(encoded, &decoded); err != nil {
		t.Fatal(err)
	}
	if decoded.Choices[0]["description"] != "First" {
		t.Fatalf("described choice lost its description: %s", encoded)
	}
	if description := decoded.Choices[1]["description"]; description != nil {
		t.Fatalf("a choice without a description must send null, not %q: %s", description, encoded)
	}
}

// fakeModels serves models file entries: Vela encoders get a classify head
// (a token head for PII), everything else answers decisions.
func fakeModels(entries []modelEntry) *runtimetest.Runtime {
	models := make([]runtimetest.Model, 0, len(entries))
	for _, entry := range entries {
		model := runtimetest.Model{ID: entry.Name}
		switch {
		case strings.Contains(entry.Model, "Encoder-307M-PII"):
			model.Heads = []runtimetest.Head{{Name: "default", Kind: "token", Labels: []string{"O", "B-PERSON", "I-PERSON"}}}
		case strings.Contains(entry.Model, "Encoder-307M"):
			model.Heads = []runtimetest.Head{{Name: "default", Kind: "sequence", Labels: []string{"math", "law", "other"}}}
		}
		models = append(models, model)
	}
	return runtimetest.New(models...)
}
