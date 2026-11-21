package classification

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
)

// A real System One client and independent classifier generations exercise
// both the generation-local result cache and the shared guard window cache.
func TestSharedGuardWindowsRemoteIdentityChange(t *testing.T) {
	for _, change := range []string{"model", "endpoint"} {
		t.Run(change, func(t *testing.T) {
			for _, shared := range []bool{false, true} {
				name := "local"
				if shared {
					name = "shared"
				}
				t.Run(name, func(t *testing.T) {
					_, store := sharedVerdictTestServer(t)
					if !shared {
						store = nil
					}
					var calls atomic.Int64
					upstream := remoteVerdictTestServer(t, &calls)
					first := remoteVerdictTestClassifier(t, store, upstream.URL, "guard-v1")
					text := scanProse(4_000)
					baseline := evaluateTurn(first, text)
					require.Empty(t, baseline.SignalErrors)
					require.False(t, guarded(baseline))
					require.Positive(t, calls.Load())

					before := calls.Load()
					same := remoteVerdictTestClassifier(t, store, upstream.URL, "guard-v1")
					got := evaluateTurn(same, text)
					require.Empty(t, got.SignalErrors)
					require.Equal(t, baseline.SignalValues, got.SignalValues)
					require.Equal(t, baseline.ScanCoverage, got.ScanCoverage)
					if shared {
						require.Equal(t, before, calls.Load(), "the same remote model can share completed windows")
					} else {
						require.Greater(t, calls.Load(), before)
					}

					before = calls.Load()
					endpoint, model := upstream.URL, "guard-v2"
					if change == "endpoint" {
						endpoint, model = upstream.URL+"/strict", "guard-v1"
					}
					changed := remoteVerdictTestClassifier(t, store, endpoint, model)
					got = evaluateTurn(changed, text)
					require.Empty(t, got.SignalErrors)
					require.True(t, guarded(got), "a new model must not inherit an old model's benign verdict")
					require.Greater(t, calls.Load(), before, "the changed model must read the input")
				})
			}
		})
	}
}

func TestGuardWindowsRequireRemoteIdentity(t *testing.T) {
	for _, missing := range []string{"model", "endpoint"} {
		t.Run(missing, func(t *testing.T) {
			deployment := config.ModelDeployment{Provider: config.SystemOneProvider, Model: "guard-v1", Endpoint: "https://guard.example/v1/systemone"}
			if missing == "model" {
				deployment.Model = ""
			} else {
				deployment.Endpoint = ""
			}
			cfg := &config.RouterConfig{}
			cfg.ModelDeployments = map[string]config.ModelDeployment{"primary": deployment}
			classifier := &Classifier{Config: cfg}
			scan := &guardScan{
				deployment: "primary", text: "input",
				rules:       []askedRule{{rule: config.SystemOneRule{Stage: config.SystemOneStageRequest}}},
				windowState: func(text string) modelservice.Request { return modelservice.Request{State: text} },
			}
			scan.verdicts = classifier.newScanVerdictKeys(scan)
			require.Nil(t, scan.verdictKey(scanWindow{end: len(scan.text)}), "unknown remote identities cannot retain or reuse window verdicts")
		})
	}
}

func remoteVerdictTestClassifier(t *testing.T, store *config.GuardVerdictStoreConfig, endpoint, model string, configure ...func(*config.ModelDeployment)) *Classifier {
	t.Helper()
	cfg := verdictTestConfig(t, store, endpoint, model, configure...)
	remote, err := modelservice.NewRemoteSystemOne(cfg)
	require.NoError(t, err)
	t.Cleanup(func() { _ = remote.Close() })
	classifier, err := buildClassifier(cfg, RecipeRuntimeOptions{SystemOne: remote})
	require.NoError(t, err)
	t.Cleanup(func() { _ = classifier.Close() })
	require.NoError(t, classifier.InitializeRuntime())
	return classifier
}

func verdictTestConfig(t *testing.T, store *config.GuardVerdictStoreConfig, endpoint, model string, configure ...func(*config.ModelDeployment)) *config.RouterConfig {
	t.Helper()
	document := strings.NewReplacer(
		"TIMEOUT", "20000", "RULE\n", "",
		"      safety:\n        - name: unsafe_request\n          predicate: {gte: 0.9}\n", "",
		"            - {type: safety, name: unsafe_request}\n", "",
	).Replace(refusingGuardConfig)
	cfg, err := config.ParseYAMLBytes([]byte(document))
	require.NoError(t, err)
	deployment := cfg.ModelDeployments["primary"]
	deployment.Provider, deployment.Endpoint, deployment.Model = config.SystemOneProvider, endpoint, model
	deployment.Schema = "vllm-sr/4.0"
	for _, apply := range configure {
		apply(&deployment)
	}
	cfg.ModelDeployments["primary"] = deployment
	cfg.GuardVerdicts = store
	return cfg
}

func remoteVerdictTestServer(t *testing.T, calls *atomic.Int64, refuse ...func(*http.Request) bool) *httptest.Server {
	t.Helper()
	server := httptest.NewServer(remoteVerdictTestHandler(calls, refuse...))
	t.Cleanup(server.Close)
	return server
}

func remoteVerdictTestHandler(calls *atomic.Int64, refuse ...func(*http.Request) bool) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.NotFound(w, r)
			return
		}
		var request struct {
			Model     string                     `json:"model"`
			Questions map[string]json.RawMessage `json:"questions"`
			States    map[string]struct {
				Questions map[string]json.RawMessage `json:"questions"`
			} `json:"states"`
		}
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		blocked := request.Model == "guard-v2" || r.URL.Path == "/strict"
		for _, check := range refuse {
			blocked = blocked || check(r)
		}
		answer := func(questions map[string]json.RawMessage) map[string]any {
			answers := make(map[string]any, len(questions))
			for id := range questions {
				choice := "writing"
				probabilities := map[string]float64{"chat": 0, "writing": 1}
				if strings.Contains(id, "jailbreak") {
					calls.Add(1)
					choice, probabilities = "benign", map[string]float64{"benign": 1, "jailbreak": 0}
					if blocked {
						choice, probabilities = "jailbreak", map[string]float64{"benign": 0, "jailbreak": 1}
					}
				}
				answers[id] = map[string]any{"type": "choice", "choice": choice, "probabilities": probabilities, "confidence": 1, "input_coverage": "complete"}
			}
			return map[string]any{"model": request.Model, "answers": answers}
		}
		response := answer(request.Questions)
		if request.States != nil {
			states := make(map[string]any, len(request.States))
			for id, state := range request.States {
				states[id] = answer(state.Questions)
			}
			response = map[string]any{"model": request.Model, "states": states}
		}
		_ = json.NewEncoder(w).Encode(response)
	})
}
