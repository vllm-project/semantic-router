package evaluation

import (
	"testing"

	"github.com/vllm-project/semantic-router/dashboard/backend/models"
)

func TestGetAvailableDatasets_IncludesSignalAndSystemDimensions(t *testing.T) {
	t.Parallel()
	datasets := GetAvailableDatasets()

	if len(datasets[string(models.DimensionDomain)]) == 0 {
		t.Error("expected domain datasets")
	}
	if len(datasets[string(models.DimensionFactCheck)]) == 0 {
		t.Error("expected fact_check datasets")
	}
	if len(datasets[string(models.DimensionUserFeedback)]) == 0 {
		t.Error("expected user_feedback datasets")
	}
	accuracySets := datasets[string(models.DimensionAccuracy)]
	if len(accuracySets) == 0 {
		t.Fatal("expected accuracy (system) datasets")
	}
	found := false
	for _, d := range accuracySets {
		if d.Name == "mmlu-pro" && d.Level == models.LevelMoM {
			found = true
			break
		}
	}
	if !found {
		t.Errorf("expected mmlu-pro dataset for accuracy (mom level); got %v", accuracySets)
	}
}
