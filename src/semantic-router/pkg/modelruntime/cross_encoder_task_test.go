package modelruntime

import "testing"

func TestCrossEncoderTaskSkippedWhenUnset(t *testing.T) {
	t.Setenv("SR_CROSS_ENCODER_MODEL_PATH", "")
	if tasks := crossEncoderTask("router"); tasks != nil {
		t.Fatalf("expected no cross-encoder task when SR_CROSS_ENCODER_MODEL_PATH is unset, got %d", len(tasks))
	}
}

func TestCrossEncoderTaskBuiltWhenConfigured(t *testing.T) {
	t.Setenv("SR_CROSS_ENCODER_MODEL_PATH", "/tmp/cross-encoder")
	tasks := crossEncoderTask("router")
	if len(tasks) != 1 {
		t.Fatalf("expected exactly one cross-encoder task, got %d", len(tasks))
	}
	if tasks[0].Name != "router.rerank.cross_encoder" {
		t.Fatalf("unexpected task name %q", tasks[0].Name)
	}
	if !tasks[0].BestEffort {
		t.Fatalf("expected cross-encoder task to be best-effort so a load failure stays non-fatal")
	}
}
