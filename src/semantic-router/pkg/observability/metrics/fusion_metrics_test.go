package metrics

import (
	"testing"

	"github.com/prometheus/client_golang/prometheus/testutil"
)

func TestRecordFusionRequest(t *testing.T) {
	before := testutil.ToFloat64(FusionRequests.WithLabelValues("decA", "success"))
	RecordFusionRequest("decA", "success")
	after := testutil.ToFloat64(FusionRequests.WithLabelValues("decA", "success"))
	if after-before != 1 {
		t.Fatalf("FusionRequests success: got delta %v, want 1", after-before)
	}
}

func TestRecordFusionRequestEmptyDecisionDefaults(t *testing.T) {
	before := testutil.ToFloat64(FusionRequests.WithLabelValues("unknown", "error"))
	RecordFusionRequest("", "error")
	after := testutil.ToFloat64(FusionRequests.WithLabelValues("unknown", "error"))
	if after-before != 1 {
		t.Fatalf("FusionRequests empty decision: got delta %v, want 1 on 'unknown'", after-before)
	}
}

func TestRecordFusionPanelModel(t *testing.T) {
	beforeOK := testutil.ToFloat64(FusionPanelModels.WithLabelValues("qwen3:8b", "success"))
	beforeFail := testutil.ToFloat64(FusionPanelModels.WithLabelValues("gemma3:12b", "failed"))
	RecordFusionPanelModel("qwen3:8b", "success")
	RecordFusionPanelModel("gemma3:12b", "failed")
	if d := testutil.ToFloat64(FusionPanelModels.WithLabelValues("qwen3:8b", "success")) - beforeOK; d != 1 {
		t.Fatalf("panel success delta: got %v, want 1", d)
	}
	if d := testutil.ToFloat64(FusionPanelModels.WithLabelValues("gemma3:12b", "failed")) - beforeFail; d != 1 {
		t.Fatalf("panel failed delta: got %v, want 1", d)
	}
}

func TestRecordFusionGroundingDroppedSkipsZero(t *testing.T) {
	before := testutil.ToFloat64(FusionGroundingDropped.WithLabelValues("filter"))
	RecordFusionGroundingDropped("filter", 0) // no-op
	if d := testutil.ToFloat64(FusionGroundingDropped.WithLabelValues("filter")) - before; d != 0 {
		t.Fatalf("dropped count=0 should be a no-op, got delta %v", d)
	}
	RecordFusionGroundingDropped("filter", 2)
	if d := testutil.ToFloat64(FusionGroundingDropped.WithLabelValues("filter")) - before; d != 2 {
		t.Fatalf("dropped count=2 delta: got %v, want 2", d)
	}
}

func TestRecordFusionRequestTokensClampsNonPositive(t *testing.T) {
	beforeP := testutil.ToFloat64(FusionRequestTokens.WithLabelValues("decT", "prompt"))
	beforeC := testutil.ToFloat64(FusionRequestTokens.WithLabelValues("decT", "completion"))
	RecordFusionRequestTokens("decT", 100, -5) // completion negative -> skipped
	if d := testutil.ToFloat64(FusionRequestTokens.WithLabelValues("decT", "prompt")) - beforeP; d != 100 {
		t.Fatalf("prompt tokens delta: got %v, want 100", d)
	}
	if d := testutil.ToFloat64(FusionRequestTokens.WithLabelValues("decT", "completion")) - beforeC; d != 0 {
		t.Fatalf("negative completion tokens should be skipped, got delta %v", d)
	}
}

// Smoke: histogram observers must not panic and must record a sample.
func TestRecordFusionHistogramsNoPanic(t *testing.T) {
	RecordFusionRequestDuration("decH", 12.5)
	RecordFusionStageDuration("panel", 3.2)
	RecordFusionStageDuration("", 0.1) // empty stage -> unknown
	RecordFusionGroundingScore("panel", "weight", 0.63)
	RecordFusionGroundingScore("", "", 0.0)
}
