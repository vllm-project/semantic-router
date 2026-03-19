package modelresearch

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	modelinventory "github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelinventory"
)

func TestStartCampaignBlocksWithoutAMDOrCPUDryRun(t *testing.T) {
	t.Parallel()

	server := newResearchTestServer(t)
	defer server.Close()

	manager, err := NewManager(ManagerConfig{
		BaseDir:             t.TempDir(),
		RepoRoot:            "/repo",
		PythonPath:          "python3",
		DefaultAPIBase:      server.URL,
		DefaultRequestModel: "MoM",
		Platform:            "cpu",
	})
	if err != nil {
		t.Fatalf("NewManager() error = %v", err)
	}

	campaign, err := manager.StartCampaign(CreateCampaignRequest{
		Name:         "blocked-campaign",
		GoalTemplate: GoalImproveAccuracy,
		Target:       "feedback",
		Budget:       Budget{MaxTrials: 1},
	})
	if err != nil {
		t.Fatalf("StartCampaign() error = %v", err)
	}
	if campaign.Status != StatusBlocked {
		t.Fatalf("campaign status = %s, want %s", campaign.Status, StatusBlocked)
	}
	if !strings.Contains(campaign.LastError, "AMD platform") {
		t.Fatalf("campaign last_error = %q, want AMD platform message", campaign.LastError)
	}
}

func TestCampaignLoopSelectsBestTrialAndPersistsConfigFragment(t *testing.T) {
	t.Parallel()

	server := newResearchTestServer(t)
	defer server.Close()

	manager, err := NewManager(ManagerConfig{
		BaseDir:             t.TempDir(),
		RepoRoot:            "/repo",
		PythonPath:          "python3",
		DefaultAPIBase:      server.URL,
		DefaultRequestModel: "MoM",
		Platform:            "amd",
		CommandRunner:       fakeCommandRunner(t),
	})
	if err != nil {
		t.Fatalf("NewManager() error = %v", err)
	}

	campaign, err := manager.StartCampaign(CreateCampaignRequest{
		Name:               "feedback-optimizer",
		GoalTemplate:       GoalImproveAccuracy,
		Target:             "feedback",
		Budget:             Budget{MaxTrials: 2},
		SuccessThresholdPP: 0.5,
		Overrides: Overrides{
			RequestModelOverride: "MoM",
		},
	})
	if err != nil {
		t.Fatalf("StartCampaign() error = %v", err)
	}

	finalCampaign := waitForCampaignTerminal(t, manager, campaign.ID)
	if finalCampaign.Status != StatusCompleted {
		t.Fatalf("campaign status = %s, want %s", finalCampaign.Status, StatusCompleted)
	}
	if finalCampaign.BestTrial == nil || finalCampaign.BestTrial.Eval == nil {
		t.Fatalf("expected best trial metrics, got %+v", finalCampaign.BestTrial)
	}
	if finalCampaign.BestTrial.Name != "trial-02" {
		t.Fatalf("best trial = %q, want trial-02", finalCampaign.BestTrial.Name)
	}
	if finalCampaign.BestTrial.Eval.ImprovementPP <= 0 {
		t.Fatalf("improvement = %.2f, want positive", finalCampaign.BestTrial.Eval.ImprovementPP)
	}
	if finalCampaign.ConfigFragmentPath == "" {
		t.Fatalf("expected config fragment path")
	}
	if _, err := os.Stat(finalCampaign.ConfigFragmentPath); err != nil {
		t.Fatalf("config fragment stat error = %v", err)
	}
	if len(finalCampaign.Trials) != 2 {
		t.Fatalf("trials = %d, want 2", len(finalCampaign.Trials))
	}
}

func TestCampaignOverridesRemainScopedToThatCampaign(t *testing.T) {
	t.Parallel()

	server := newResearchTestServer(t)
	defer server.Close()

	manager, err := NewManager(ManagerConfig{
		BaseDir:             t.TempDir(),
		RepoRoot:            "/repo",
		PythonPath:          "python3",
		DefaultAPIBase:      server.URL,
		DefaultRequestModel: "MoM",
		Platform:            "cpu",
	})
	if err != nil {
		t.Fatalf("NewManager() error = %v", err)
	}

	overrideCampaign, err := manager.StartCampaign(CreateCampaignRequest{
		Name:         "override-campaign",
		GoalTemplate: GoalImproveAccuracy,
		Target:       "feedback",
		Budget:       Budget{MaxTrials: 1},
		Overrides: Overrides{
			APIBaseOverride:      server.URL,
			RequestModelOverride: "feedback-shadow",
		},
	})
	if err != nil {
		t.Fatalf("StartCampaign() with overrides error = %v", err)
	}

	defaultCampaign, err := manager.StartCampaign(CreateCampaignRequest{
		Name:         "default-campaign",
		GoalTemplate: GoalImproveAccuracy,
		Target:       "feedback",
		Budget:       Budget{MaxTrials: 1},
	})
	if err != nil {
		t.Fatalf("StartCampaign() default error = %v", err)
	}

	if overrideCampaign.RequestModel != "feedback-shadow" {
		t.Fatalf("override request model = %q, want feedback-shadow", overrideCampaign.RequestModel)
	}
	if defaultCampaign.RequestModel != "MoM" {
		t.Fatalf("default request model = %q, want MoM", defaultCampaign.RequestModel)
	}
	if overrideCampaign.APIBase != server.URL {
		t.Fatalf("override api base = %q, want %q", overrideCampaign.APIBase, server.URL)
	}
	if defaultCampaign.APIBase != server.URL {
		t.Fatalf("default api base = %q, want %q", defaultCampaign.APIBase, server.URL)
	}
}

func newResearchTestServer(t *testing.T) *httptest.Server {
	t.Helper()

	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/v1/models":
			_ = json.NewEncoder(w).Encode(map[string]any{
				"data": []map[string]any{
					{"id": "MoM"},
					{"id": "feedback-shadow"},
				},
			})
		case "/info/models":
			_ = json.NewEncoder(w).Encode(modelinventory.ModelsInfoResponse{
				Models: []modelinventory.ModelInfo{
					{
						Name:      "feedback_detector",
						Type:      "feedback_detection",
						Loaded:    true,
						State:     "ready",
						ModelPath: "models/mmbert-feedback-detector-merged",
					},
				},
				Summary: modelinventory.ModelsInfoSummary{
					Ready:        true,
					Phase:        "ready",
					LoadedModels: 1,
					TotalModels:  1,
				},
			})
		default:
			http.NotFound(w, r)
		}
	}))
}

func fakeCommandRunner(t *testing.T) commandRunner {
	t.Helper()

	return func(_ context.Context, spec commandSpec, _ func(stream, line string)) error {
		args := spec.Args
		switch {
		case containsArg(args, "mom_collection_eval.py"):
			outputDir := argValue(args, "--output_dir")
			if err := os.MkdirAll(outputDir, 0o755); err != nil {
				return err
			}
			accuracy := 0.80
			switch {
			case strings.Contains(outputDir, "trial-01"):
				accuracy = 0.81
			case strings.Contains(outputDir, "trial-02"):
				accuracy = 0.84
			}
			modelName := argValue(args, "--model")
			payload := map[string]any{
				"accuracy":  accuracy,
				"f1":        accuracy - 0.01,
				"precision": accuracy - 0.02,
				"recall":    accuracy - 0.015,
				"latency": map[string]any{
					"avg_ms": 12.5,
				},
			}
			data, _ := json.Marshal(payload)
			return os.WriteFile(filepath.Join(outputDir, modelName+"_results.json"), data, 0o644)
		case containsArg(args, "signal_eval.py"):
			outputPath := argValue(args, "--output")
			if err := os.MkdirAll(filepath.Dir(outputPath), 0o755); err != nil {
				return err
			}
			data, _ := json.Marshal(map[string]any{"accuracy": 0.78})
			return os.WriteFile(outputPath, data, 0o644)
		case containsArg(args, "train_feedback_detector.py"):
			outputDir := argValue(args, "--output_dir") + "_lora"
			return os.MkdirAll(outputDir, 0o755)
		default:
			return nil
		}
	}
}

func waitForCampaignTerminal(t *testing.T, manager *Manager, campaignID string) *Campaign {
	t.Helper()

	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		campaign := manager.GetCampaign(campaignID)
		if campaign != nil {
			switch campaign.Status {
			case StatusCompleted, StatusFailed, StatusStopped, StatusBlocked:
				return campaign
			}
		}
		time.Sleep(20 * time.Millisecond)
	}
	t.Fatalf("campaign %s did not reach terminal state", campaignID)
	return nil
}

func containsArg(args []string, suffix string) bool {
	for _, arg := range args {
		if strings.HasSuffix(arg, suffix) {
			return true
		}
	}
	return false
}

func argValue(args []string, name string) string {
	for index, arg := range args {
		if arg == name && index+1 < len(args) {
			return args[index+1]
		}
	}
	return ""
}
