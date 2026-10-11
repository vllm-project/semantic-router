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
	"strconv"
	"strings"
	"testing"
	"time"

	"gopkg.in/yaml.v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice/api"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice/runtimetest"
)

const fakeRuntimeEnv = "MODELSERVICE_FAKE_RUNTIME"

// fakeRuntimeFailOnceEnv names a marker file: the first fake process to start
// creates it and reports every model failed; later processes load normally.
// fakeRuntimeFailAlwaysEnv makes every fake process report its models failed.
// fakeRuntimeHoldEnv names a file: until it exists, a fake process reports its
// models loading.
const (
	fakeRuntimeFailOnceEnv   = "MODELSERVICE_FAKE_FAIL_ONCE"
	fakeRuntimeFailAlwaysEnv = "MODELSERVICE_FAKE_FAIL_ALWAYS"
	fakeRuntimeHoldEnv       = "MODELSERVICE_FAKE_HOLD"
)

// fakeAutoEnv is the device the fake's devices command reports for auto;
// unset, the command fails. fakeDevicesLogEnv names a file the command
// appends a line to on every call.
const (
	fakeAutoEnv       = "MODELSERVICE_FAKE_AUTO"
	fakeDevicesLogEnv = "MODELSERVICE_FAKE_DEVICES_LOG"
)

// TestMain lets the test binary act as a managed runtime process:
// <binary> serve --models FILE --uds PATH serves the fake contract on the
// socket, and <binary> devices answers the device query.
func TestMain(m *testing.M) {
	if os.Getenv(fakeRuntimeEnv) == "1" {
		if len(os.Args) == 2 && os.Args[1] == "devices" {
			fakeDevices()
			return
		}
		serveManagedFake(os.Args[1:])
		return
	}
	os.Exit(m.Run())
}

func fakeDevices() {
	if path := os.Getenv(fakeDevicesLogEnv); path != "" {
		file, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o600)
		if err != nil {
			os.Exit(2)
		}
		_, _ = file.WriteString("devices\n")
		_ = file.Close()
	}
	device := os.Getenv(fakeAutoEnv)
	if device == "" {
		_, _ = os.Stderr.WriteString("no accelerator plugin could be loaded\n")
		os.Exit(1)
	}
	data, _ := json.Marshal(map[string]interface{}{"auto": device, "devices": []string{device}})
	_, _ = os.Stdout.Write(append(data, '\n'))
}

func serveManagedFake(args []string) {
	socket, modelsFile := "", ""
	limits := api.ProcessLimits{MaxBundleTasks: DefaultBundleTasks, MaxRequestBytes: 8 << 20}
	for index := 0; index+1 < len(args); index++ {
		switch args[index] {
		case "--uds":
			socket = args[index+1]
		case "--models":
			modelsFile = args[index+1]
		case "--max-bundle-tasks":
			limits.MaxBundleTasks, _ = strconv.Atoi(args[index+1])
		case "--max-request-bytes":
			limits.MaxRequestBytes, _ = strconv.Atoi(args[index+1])
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
	fake := fakeModels(document.Models)
	fake.SetLimits(limits)
	failed := os.Getenv(fakeRuntimeFailAlwaysEnv) == "1"
	if marker := os.Getenv(fakeRuntimeFailOnceEnv); marker != "" {
		_, statErr := os.Stat(marker)
		failed = errors.Is(statErr, os.ErrNotExist) && os.WriteFile(marker, nil, 0o600) == nil
	}
	if failed {
		for _, entry := range document.Models {
			fake.SetFailed(entry.Name, "fake load failure")
		}
	}
	if hold := os.Getenv(fakeRuntimeHoldEnv); hold != "" {
		holdModels(fake, document.Models, hold)
	}
	_ = http.Serve(listener, fake.Handler())
}

// holdModels reports the models loading until the hold file exists.
func holdModels(fake *runtimetest.Runtime, models []modelEntry, hold string) {
	for _, entry := range models {
		fake.SetReady(entry.Name, false)
	}
	go func() {
		for {
			if _, err := os.Stat(hold); err == nil {
				for _, entry := range models {
					fake.SetReady(entry.Name, true)
				}
				return
			}
			time.Sleep(20 * time.Millisecond)
		}
	}()
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

func TestManagedCommandAndModelsFile(t *testing.T) {
	got := strings.Join(managedCommand([]string{"vllm-srun"}, "/run/x.sock", "/run/x.models.json", "/cache", 0), " ")
	if want := "vllm-srun serve --models /run/x.models.json --uds /run/x.sock --max-request-bytes 67108864 --max-bundle-tasks 1024 --cache-dir /cache"; got != want {
		t.Fatalf("command\n got %s\nwant %s", got, want)
	}
	if got := strings.Join(managedCommand([]string{"vllm-srun"}, "/run/x.sock", "/run/x.models.json", "", 4), " "); !strings.HasSuffix(got, "--max-bundle-tasks 1024 --threads 4") {
		t.Fatalf("a CPU share passes its thread count: %s", got)
	}
	plan := planProcesses(map[string]config.ModelDeployment{
		"kai": {Provider: config.ModelRuntimeProvider, Artifact: "vllm-sr/Decision-2.0-Kai-0.6B", Revision: strings.Repeat("b", 40), Device: "cpu", Profile: "batching"},
	}, nil, "", 0, "")[0]
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

func TestManagerKeepsIndependentProcessesAndSupervisesThem(t *testing.T) {
	binary, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	t.Setenv(fakeRuntimeEnv, "1")
	t.Setenv(RuntimeCommandEnv, binary)
	t.Setenv(RuntimeDirEnv, filepath.Join(t.TempDir(), "run"))
	t.Setenv(CPUThreadsEnv, "1")
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
	if len(statuses) != 2 || statuses[0].Endpoint == statuses[1].Endpoint || !statuses[0].Managed || statuses[0].Process == statuses[1].Process {
		t.Fatalf("each logical model owns an independent managed process: %+v", statuses)
	}
	info, err := os.Stat(filepath.Dir(strings.TrimPrefix(statuses[0].Endpoint, "unix://")))
	if err != nil || info.Mode().Perm() != 0o700 {
		t.Fatalf("socket directory must be private: %v %v", info, err)
	}
	if len(manager.groups) != 2 {
		t.Fatalf("leases reuse each independently owned process: %d processes", len(manager.groups))
	}
	if _, err := generation.Decide(context.Background(), "eos", sampleRequest("text")); err != nil {
		t.Fatal(err)
	}
	// Removing a different logical deployment retains kai's original process.
	initialKai := generation.members["kai"].group
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
	if manager.Published().members["kai"].group != initialKai {
		t.Fatal("consumer set change restarted unchanged deployment")
	}
	_ = generation.Close()
	if len(manager.groups) != 1 {
		t.Fatalf("closing the last lease stops its process: %d processes", len(manager.groups))
	}
}

// managedFakeLease serves one managed fake deployment, "kai", from this test binary.
func managedFakeLease(t *testing.T) *Lease {
	t.Helper()
	binary, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	t.Setenv(fakeRuntimeEnv, "1")
	t.Setenv(RuntimeCommandEnv, binary)
	t.Setenv(RuntimeDirEnv, filepath.Join(t.TempDir(), "run"))
	manager := NewManager()
	t.Cleanup(func() { _ = manager.Shutdown(context.Background()) })
	lease, err := manager.Acquire(runtimeConfig(map[string]config.ModelDeployment{
		"kai": {Provider: config.ModelRuntimeProvider, Artifact: "vllm-sr/Decision-2.0-Kai-0.6B", Device: "cpu"},
	}))
	if err != nil {
		t.Fatal(err)
	}
	return lease
}

// TestManagedProcessesTakeTheRouterLimits sends a body between the runtime's
// default 8 MiB bound and the managed one through a managed process, which
// was started with the router's --max-request-bytes and --max-bundle-tasks;
// the client learns the bundle cap from the process.
func TestManagedProcessesTakeTheRouterLimits(t *testing.T) {
	binary, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	t.Setenv(fakeRuntimeEnv, "1")
	t.Setenv(RuntimeCommandEnv, binary)
	t.Setenv(RuntimeDirEnv, filepath.Join(t.TempDir(), "run"))
	manager := NewManager()
	t.Cleanup(func() { _ = manager.Shutdown(context.Background()) })
	lease, err := manager.Acquire(runtimeConfig(map[string]config.ModelDeployment{
		"domain": {Provider: config.ModelRuntimeProvider, Artifact: "vllm-sr/Vela-1.0-Encoder-307M-Domain", Device: "cpu"},
	}))
	if err != nil {
		t.Fatal(err)
	}
	waitReady(t, lease, "domain")
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	response, err := lease.Classify(ctx, "domain", ClassifyRequest{Inputs: []ClassifyInput{{Text: strings.Repeat("a", 16<<20)}}})
	if err != nil || len(response.Results) != 1 {
		t.Fatalf("a 16 MiB request must fit a managed process: %v", err)
	}
	if limit := lease.members["domain"].group.client.bundleTasks.Load(); limit != managedBundleTasks {
		t.Fatalf("the client's bundle cap is %d, want the managed %d", limit, managedBundleTasks)
	}
}

func TestSupervisorRestartsAProcessWhoseEveryModelFailedToLoad(t *testing.T) {
	t.Setenv(fakeRuntimeFailOnceEnv, filepath.Join(t.TempDir(), "failed-once"))
	lease := managedFakeLease(t)
	waitReady(t, lease, "kai")
	if statuses := lease.Statuses(); len(statuses) != 1 || statuses[0].Restarts != 1 || !statuses[0].Ready {
		t.Fatalf("one recycle restarts the process: %+v", statuses)
	}
}

func TestCardFailsAfterRepeatedFailedLoads(t *testing.T) {
	t.Setenv(fakeRuntimeFailAlwaysEnv, "1")
	lease := managedFakeLease(t)
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	started := time.Now()
	if _, err := lease.Card(ctx, "kai"); !errors.Is(err, ErrUnavailable) || !strings.Contains(err.Error(), "failed to load 3 times") {
		t.Fatalf("repeated failed loads must fail preparation, got %v", err)
	}
	if time.Since(started) > 20*time.Second {
		t.Fatal("the failure must not wait for the deadline")
	}
	if statuses := lease.Statuses(); statuses[0].Restarts < 2 {
		t.Fatalf("the supervisor retried before giving up: %+v", statuses)
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

func TestCardRefusesARuntimeOfAnotherContractMajor(t *testing.T) {
	for _, version := range []string{"3.0.0", ""} {
		runtime := fakeModels([]modelEntry{{Name: "decider", Model: "vllm-sr/Decision-2.0-Kai-0.6B"}})
		runtime.SetAPIVersion(version)
		server := httptest.NewServer(runtime.Handler())
		manager := NewManager()
		lease, err := manager.Acquire(runtimeConfig(map[string]config.ModelDeployment{
			"decider": {Provider: config.ModelRuntimeProvider, Endpoint: server.URL},
		}))
		if err != nil {
			t.Fatal(err)
		}
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		started := time.Now()
		_, err = lease.Card(ctx, "decider")
		cancel()
		if !errors.Is(err, ErrUnavailable) || !strings.Contains(err.Error(), "this router speaks "+RuntimeAPIMajor+".x") {
			t.Fatalf("api_version %q must be refused, got %v", version, err)
		}
		if time.Since(started) > 5*time.Second {
			t.Fatal("the refusal must not wait for the deadline")
		}
		if statuses := lease.Statuses(); statuses[0].Ready || statuses[0].State != "incompatible" {
			t.Fatalf("the deployment reports the incompatible contract: %+v", statuses)
		}
		_ = manager.Shutdown(context.Background())
		server.Close()
	}
}

func TestRuntimeAPIMajorIsTheContractsMajor(t *testing.T) {
	data, err := os.ReadFile(filepath.Join("..", "..", "..", "model-runtime", "vllm_srun", "api", "openapi.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	var spec struct {
		Info struct {
			Version string `yaml:"version"`
		} `yaml:"info"`
	}
	if err := yaml.Unmarshal(data, &spec); err != nil {
		t.Fatal(err)
	}
	if major, _, _ := strings.Cut(spec.Info.Version, "."); major != RuntimeAPIMajor {
		t.Fatalf("openapi.yaml info.version %s: RuntimeAPIMajor is %s; regenerate the client and move the constant together", spec.Info.Version, RuntimeAPIMajor)
	}
	if major, _, _ := strings.Cut(runtimetest.APIVersion, "."); major != RuntimeAPIMajor {
		t.Fatalf("the fake serves %s", runtimetest.APIVersion)
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
