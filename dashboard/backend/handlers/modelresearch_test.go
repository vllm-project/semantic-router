package handlers

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/dashboard/backend/modelresearch"
)

func TestModelResearchStreamEvents_ReplaysBlockedCampaignAndCompletes(t *testing.T) {
	t.Parallel()

	manager, err := modelresearch.NewManager(modelresearch.ManagerConfig{
		BaseDir:             t.TempDir(),
		DefaultRequestModel: "MoM",
		Platform:            "cpu",
	})
	if err != nil {
		t.Fatalf("NewManager() error = %v", err)
	}

	campaign, err := manager.StartCampaign(modelresearch.CreateCampaignRequest{
		Name:         "blocked-campaign",
		GoalTemplate: modelresearch.GoalImproveAccuracy,
		Target:       "feedback",
		Budget:       modelresearch.Budget{MaxTrials: 1},
	})
	if err != nil {
		t.Fatalf("StartCampaign() error = %v", err)
	}
	if campaign.Status != modelresearch.StatusBlocked {
		t.Fatalf("campaign status = %s, want %s", campaign.Status, modelresearch.StatusBlocked)
	}

	handler := NewModelResearchHandler(manager)
	req := httptest.NewRequest(http.MethodGet, "/api/model-research/campaigns/"+campaign.ID+"/events", nil)
	recorder := httptest.NewRecorder()

	handler.StreamEventsHandler().ServeHTTP(recorder, req)

	if recorder.Code != http.StatusOK {
		t.Fatalf("StreamEventsHandler() status = %d, want %d", recorder.Code, http.StatusOK)
	}

	body := recorder.Body.String()
	for _, expected := range []string{
		"event: connected",
		"Campaign created",
		"AMD platform is required unless CPU dry run is explicitly enabled",
		"event: completed",
	} {
		if !strings.Contains(body, expected) {
			t.Fatalf("expected SSE payload to contain %q, got %q", expected, body)
		}
	}
}
