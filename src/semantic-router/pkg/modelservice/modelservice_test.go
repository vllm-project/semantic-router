package modelservice

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

const fakeRuntimeEnv = "MODELSERVICE_FAKE_RUNTIME"

// TestMain lets the test binary act as a managed runtime process:
// <binary> serve <artifact> --uds <path> ... serves the fake contract on the socket.
func TestMain(m *testing.M) {
	if os.Getenv(fakeRuntimeEnv) == "1" {
		serveFakeRuntime(os.Args[1:])
		return
	}
	os.Exit(m.Run())
}

func serveFakeRuntime(args []string) {
	socket := ""
	for index := 0; index+1 < len(args); index++ {
		if args[index] == "--uds" {
			socket = args[index+1]
		}
	}
	listener, err := net.Listen("unix", socket)
	if err != nil {
		os.Exit(2)
	}
	_ = http.Serve(listener, fakeRuntime(&atomic.Int64{}, nil))
}

// fakeRuntime answers like the runtime: noul P(true)=0.8, choice picks the
// second option, score expects level 1.5; question "fail" returns a per-question error.
func fakeRuntime(calls *atomic.Int64, ready *atomic.Bool) http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/health", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if ready != nil && !ready.Load() {
			w.WriteHeader(http.StatusServiceUnavailable)
			_, _ = w.Write([]byte(`{"status":"loading","reason":"weights","model":null}`))
			return
		}
		_, _ = w.Write([]byte(`{"status":"ready","reason":null,"model":"tiny"}`))
	})
	mux.HandleFunc("/v1/decisions", func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		var body struct {
			State     string                     `json:"state"`
			Questions map[string]json.RawMessage `json:"questions"`
			Options   map[string]interface{}     `json:"options"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			w.WriteHeader(http.StatusBadRequest)
			_, _ = w.Write([]byte(`{"error":{"code":"invalid_request","message":"bad json"}}`))
			return
		}
		if body.State == "overload" {
			w.WriteHeader(http.StatusTooManyRequests)
			_, _ = w.Write([]byte(`{"error":{"code":"overloaded","message":"busy"}}`))
			return
		}
		if body.State == "slow" {
			time.Sleep(300 * time.Millisecond)
		}
		answers := map[string]interface{}{}
		for id, raw := range body.Questions {
			var question struct {
				Type    string `json:"type"`
				Choices []struct {
					Key string `json:"key"`
				} `json:"choices"`
				Levels []string `json:"levels"`
			}
			_ = json.Unmarshal(raw, &question)
			switch {
			case id == "fail":
				answers[id] = map[string]interface{}{"type": question.Type, "error": "max_length_exceeded"}
			case question.Type == "noul":
				answers[id] = map[string]interface{}{"type": "noul", "noul": 0.8}
			case question.Type == "score":
				answers[id] = map[string]interface{}{"type": "score", "score": 1.5, "confidence": 0.4, "probabilities": map[string]float64{"0": 0.1, "1": 0.3, "2": 0.6}}
			default:
				probabilities := map[string]float64{}
				for index, choice := range question.Choices {
					probabilities[choice.Key] = 0.1
					if index == 1 {
						probabilities[choice.Key] = 1 - 0.1*float64(len(question.Choices)-1)
					}
				}
				answers[id] = map[string]interface{}{"type": "choice", "choice": question.Choices[1].Key, "confidence": 0.7, "probabilities": probabilities}
			}
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"model": "tiny", "answers": answers, "usage": map[string]int{"input_tokens": 10, "output_tokens": 0},
		})
	})
	return mux
}

func sampleRequest(state string) Request {
	return Request{State: state, Questions: []Question{
		{ID: "reasoning", Type: "noul", Instructions: "Hard?"},
		{ID: "kind", Type: "choice", Instructions: "Kind?", Choices: []Choice{{Key: "code", Description: "Code"}, {Key: "math", Description: "Math"}}},
		{ID: "difficulty", Type: "score", Instructions: "How hard?", Levels: []string{"easy", "medium", "hard"}},
		{ID: "fail", Type: "noul", Instructions: "x"},
	}}
}

func TestClientDecideOverTCP(t *testing.T) {
	server := httptest.NewServer(fakeRuntime(&atomic.Int64{}, nil))
	defer server.Close()
	client, err := NewClient(server.URL)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	response, err := client.Decide(ctx, sampleRequest("text"))
	if err != nil {
		t.Fatal(err)
	}
	if response.InputTokens != 10 {
		t.Fatalf("unexpected response metadata: %+v", response)
	}
	if got := response.Answers["reasoning"]; got.Type != "noul" || got.Noul != 0.8 {
		t.Fatalf("noul answer = %+v", got)
	}
	if got := response.Answers["kind"]; got.Choice != "math" || got.Probabilities["math"] != 0.9 {
		t.Fatalf("choice answer = %+v", got)
	}
	if got := response.Answers["difficulty"]; got.Score != 1.5 {
		t.Fatalf("score answer = %+v", got)
	}
	if got := response.Answers["fail"]; got.Error != "max_length_exceeded" {
		t.Fatalf("per-question error = %+v", got)
	}
}

func TestClientDecideOverUnixSocket(t *testing.T) {
	socket := filepath.Join(t.TempDir(), "runtime.sock")
	listener, err := net.Listen("unix", socket)
	if err != nil {
		t.Fatal(err)
	}
	server := &http.Server{Handler: fakeRuntime(&atomic.Int64{}, nil), ReadHeaderTimeout: time.Second}
	go func() { _ = server.Serve(listener) }()
	defer server.Close()
	client, err := NewClient("unix://" + socket)
	if err != nil {
		t.Fatal(err)
	}
	ready, state, err := client.Ready(context.Background())
	if err != nil || !ready || state != "ready" {
		t.Fatalf("Ready() = %v %q %v", ready, state, err)
	}
	if _, err := client.Decide(context.Background(), sampleRequest("text")); err != nil {
		t.Fatal(err)
	}
}

func TestClientErrorMapping(t *testing.T) {
	server := httptest.NewServer(fakeRuntime(&atomic.Int64{}, nil))
	defer server.Close()
	client, _ := NewClient(server.URL)
	if _, err := client.Decide(context.Background(), sampleRequest("overload")); !errors.Is(err, ErrOverloaded) {
		t.Fatalf("overload error = %v", err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	_, err := client.Decide(ctx, sampleRequest("slow"))
	if ErrorReason(err) != "timeout" {
		t.Fatalf("slow call error = %v (%s)", err, ErrorReason(err))
	}
}

func runtimeConfig(deployment config.ModelDeployment) *config.RouterConfig {
	cfg := &config.RouterConfig{}
	cfg.ModelDeployments = map[string]config.ModelDeployment{
		"decider": deployment,
		"unused":  {Provider: config.ModelRuntimeProvider, Artifact: "acme/unused"},
	}
	cfg.DecisionRules = []config.DecisionSignalRule{{
		Name: "hard", Deployment: "decider", Question: config.DecisionQuestion{Type: "noul", Instructions: "Hard?"},
	}}
	return cfg
}

func waitReady(t *testing.T, manager *Manager, name string) {
	t.Helper()
	deadline := time.Now().Add(20 * time.Second)
	for time.Now().Before(deadline) {
		for _, status := range manager.Statuses() {
			if status.Name == name && status.Ready {
				return
			}
		}
		time.Sleep(50 * time.Millisecond)
	}
	t.Fatalf("deployment %s never became ready: %+v", name, manager.Statuses())
}

func TestManagerAttachFailsOpenUntilReady(t *testing.T) {
	ready := &atomic.Bool{}
	calls := &atomic.Int64{}
	server := httptest.NewServer(fakeRuntime(calls, ready))
	defer server.Close()
	manager := NewManager()
	defer func() { _ = manager.Shutdown(context.Background()) }()
	if err := manager.Reconcile(runtimeConfig(config.ModelDeployment{Provider: config.ModelRuntimeProvider, Endpoint: server.URL})); err != nil {
		t.Fatal(err)
	}
	statuses := manager.Statuses()
	if len(statuses) != 1 || statuses[0].Name != "decider" || statuses[0].Managed {
		t.Fatalf("only the referenced deployment should start, attached: %+v", statuses)
	}
	if _, err := manager.Decide(context.Background(), "decider", sampleRequest("text")); !errors.Is(err, ErrUnavailable) {
		t.Fatalf("not-ready deployment must fail open at once, got %v", err)
	}
	if calls.Load() != 0 {
		t.Fatal("a not-ready deployment must not be called")
	}
	ready.Store(true)
	waitReady(t, manager, "decider")
	if _, err := manager.Decide(context.Background(), "decider", sampleRequest("text")); err != nil {
		t.Fatal(err)
	}
	if _, err := manager.Decide(context.Background(), "missing", sampleRequest("text")); !errors.Is(err, ErrUnknownDeployment) {
		t.Fatalf("unknown deployment error = %v", err)
	}
}

func TestManagerSupervisesAndRestartsManagedRuntime(t *testing.T) {
	binary, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	t.Setenv(fakeRuntimeEnv, "1")
	t.Setenv(RuntimeCommandEnv, binary)
	t.Setenv(RuntimeDirEnv, filepath.Join(t.TempDir(), "run"))
	manager := NewManager()
	defer func() { _ = manager.Shutdown(context.Background()) }()
	deployment := config.ModelDeployment{Provider: config.ModelRuntimeProvider, Artifact: "vllm-sr/Decision-2.0-Kai-0.6B"}
	if reconcileErr := manager.Reconcile(runtimeConfig(deployment)); reconcileErr != nil {
		t.Fatal(reconcileErr)
	}
	waitReady(t, manager, "decider")
	status := manager.Statuses()[0]
	if !status.Managed || !strings.HasPrefix(status.Endpoint, "unix://") {
		t.Fatalf("managed deployment status = %+v", status)
	}
	info, err := os.Stat(filepath.Dir(strings.TrimPrefix(status.Endpoint, "unix://")))
	if err != nil || info.Mode().Perm() != 0o700 {
		t.Fatalf("socket directory must be private: %v %v", info.Mode(), err)
	}
	if _, err := manager.Decide(context.Background(), "decider", sampleRequest("text")); err != nil {
		t.Fatal(err)
	}
	// A configuration change restarts the runtime; removing every reference stops it.
	deployment.Profile = "batching"
	if err := manager.Reconcile(runtimeConfig(deployment)); err != nil {
		t.Fatal(err)
	}
	waitReady(t, manager, "decider")
	if err := manager.Reconcile(&config.RouterConfig{}); err != nil {
		t.Fatal(err)
	}
	if len(manager.Statuses()) != 0 {
		t.Fatalf("unreferenced deployments must stop: %+v", manager.Statuses())
	}
}

func TestManagedCommand(t *testing.T) {
	deployment := config.ModelDeployment{
		Provider: config.ModelRuntimeProvider, Artifact: "vllm-sr/Decision-2.0-Eos-0.8B",
		Revision: strings.Repeat("b", 40), Device: "rocm:1", Profile: "shared_context",
	}
	got := strings.Join(managedCommand([]string{"vllm-sr-runtime"}, deployment, "/run/x.sock", "/cache"), " ")
	want := fmt.Sprintf("vllm-sr-runtime serve vllm-sr/Decision-2.0-Eos-0.8B --uds /run/x.sock --device rocm:1 --profile shared_context --revision %s --cache-dir /cache", strings.Repeat("b", 40))
	if got != want {
		t.Fatalf("command\n got %s\nwant %s", got, want)
	}
}

func TestDefaultDeciderFailsOpenBeforeStartup(t *testing.T) {
	if _, err := (unavailable{}).Decide(context.Background(), "x", Request{}); !errors.Is(err, ErrUnavailable) {
		t.Fatal("the default decider must answer unavailable")
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
