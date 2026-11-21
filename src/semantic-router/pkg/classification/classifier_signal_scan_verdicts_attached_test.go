package classification

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/serving"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice/api"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice/runtimetest"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/presets"
)

func TestSharedGuardWindowsAttachedIdentity(t *testing.T) {
	for _, change := range []string{"endpoint", "model", "headers", "static_auth", "replica_endpoint", "replica_model"} {
		t.Run(change, func(t *testing.T) {
			server, store := sharedVerdictTestServer(t)
			var calls atomic.Int64
			upstream := httptest.NewServer(attachedVerdictTestHandler(&calls))
			t.Cleanup(upstream.Close)
			text := scanProse(4_000)
			var classifiers []*Classifier
			for _, strict := range []bool{false, true} {
				t.Setenv("ATTACHED_GUARD_KEY", map[bool]string{false: "lenient", true: "strict"}[strict])
				classifier := attachedVerdictTestClassifier(t, store, upstream.URL, func(d *config.ModelDeployment) {
					if strings.HasPrefix(change, "replica_") {
						d.Artifact = "example/guard"
						d.Replicas = []config.ModelReplica{{Endpoint: d.Endpoint, Model: d.Model}}
						d.Endpoint, d.Model = "", ""
					}
					if change == "static_auth" {
						d.Auth = &config.DeploymentAuth{APIKeyEnv: "ATTACHED_GUARD_KEY"}
					}
					if strict {
						switch change {
						case "endpoint":
							d.Endpoint += "/strict"
						case "model":
							d.Model = "guard-v2"
						case "headers":
							d.Headers = map[string]string{"X-Policy": "strict"}
						case "replica_endpoint":
							d.Replicas[0].Endpoint += "/strict"
						case "replica_model":
							d.Replicas[0].Model = "guard-v2"
						}
					}
				})
				classifiers = append(classifiers, classifier)
				before := calls.Load()
				got := evaluateTurn(classifier, text)
				require.Empty(t, got.SignalErrors)
				require.Equal(t, strict, guarded(got), "attached policy changes must not reuse a benign verdict")
				require.Greater(t, calls.Load(), before)
			}
			if change == "static_auth" {
				for _, classifier := range classifiers {
					require.Empty(t, classifier.windowVerdicts.entries)
				}
				require.Empty(t, server.Keys())
			}
		})
	}
}

func TestSharedGuardWindowsEffectivePrecision(t *testing.T) {
	_, store := sharedVerdictTestServer(t)
	const model = "guard-v1"
	fake := runtimetest.New(runtimetest.Model{
		ID: model, Repo: "example/guard", Revision: "v1", Family: presets.FamilyVela3D3,
		MaxInputTokens: 32_768, Dtype: "fp32",
	})
	upstream := httptest.NewServer(fake.Handler())
	t.Cleanup(upstream.Close)
	text := scanProse(4_000)
	// Each replica starts with an empty local cache and the same deployment.
	scanOnNewReplica := func() int {
		t.Helper()
		classifier := attachedVerdictTestClassifier(t, store, upstream.URL, func(d *config.ModelDeployment) { d.Dtype = "auto" })
		before := len(guardStates(fake))
		got := evaluateTurn(classifier, text)
		require.Empty(t, got.SignalErrors)
		require.Equal(t, scanCoverageFull, got.ScanCoverage["jailbreak:prompt_attack"])
		return len(guardStates(fake)) - before
	}
	windows := scanOnNewReplica()
	require.Positive(t, windows)
	require.Zero(t, scanOnNewReplica(), "unchanged fp32 windows should be reused")
	fake.SetDtype(model, "bf16/fp32-readout")
	require.Equal(t, windows, scanOnNewReplica(), "a change in reported precision must rescan every window")
	require.Zero(t, scanOnNewReplica(), "unchanged bf16 windows should be reused")
}

func attachedVerdictTestClassifier(t *testing.T, store *config.GuardVerdictStoreConfig, endpoint string, configure ...func(*config.ModelDeployment)) *Classifier {
	t.Helper()
	cfg := verdictTestConfig(t, store, endpoint, "guard-v1", func(d *config.ModelDeployment) {
		d.Provider, d.Schema = config.ModelRuntimeProvider, ""
		for _, apply := range configure {
			apply(d)
		}
	})
	manager := modelservice.NewManager()
	t.Cleanup(func() { _ = manager.Shutdown(context.Background()) })
	lease, err := manager.AcquireDeployments(cfg.ModelDeployments)
	require.NoError(t, err)
	t.Cleanup(func() { _ = lease.Close() })
	runtime := serving.New(lease, nil)
	primeAttachedDeploymentCards(t, runtime, cfg)
	classifier, err := buildClassifier(cfg, RecipeRuntimeOptions{Runtime: runtime})
	require.NoError(t, err)
	t.Cleanup(func() { _ = classifier.Close() })
	require.NoError(t, classifier.InitializeRuntime())
	return classifier
}

// All models report the same artifact identity. Only the connection and served
// model select the guard policy, through the real runtime lease and HTTP client.
func attachedVerdictTestHandler(calls *atomic.Int64) http.Handler {
	model := func(id string, strict bool) runtimetest.Model {
		return runtimetest.Model{
			ID: id, Family: presets.FamilyVela3D3, MaxInputTokens: 32_768,
			ModelSHA256: strings.Repeat("a", 64),
			Answer: func(state, id string, question api.Question) (api.Answer, bool) {
				if strings.Contains(id, "jailbreak") {
					calls.Add(1)
				}
				return scoredAnswer(func(string) float64 {
					if strict {
						return 1
					}
					return 0
				})(state, id, question)
			},
		}
	}
	lenient := runtimetest.New(model("guard-v1", false), model("guard-v2", true)).Handler()
	strict := runtimetest.New(model("guard-v1", true), model("guard-v2", true)).Handler()
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		blocked := r.Header.Get("X-Policy") == "strict" || r.Header.Get("Authorization") == "Bearer strict" ||
			strings.HasPrefix(r.URL.Path, "/strict/") || (r.TLS != nil && r.TLS.ServerName == "strict.example")
		if blocked {
			r.URL.Path = strings.TrimPrefix(r.URL.Path, "/strict")
			strict.ServeHTTP(w, r)
			return
		}
		lenient.ServeHTTP(w, r)
	})
}
