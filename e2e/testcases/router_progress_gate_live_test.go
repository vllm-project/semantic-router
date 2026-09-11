package testcases

import (
	"context"
	"os"
	"testing"
	"time"
)

// Runs the maintained profile assertion against an already running stack.
func TestProgressGateLive(t *testing.T) {
	public, management := os.Getenv("GATE_E2E_INFERENCE_URL"), os.Getenv("GATE_E2E_MANAGEMENT_URL")
	if public == "" || management == "" {
		t.Skip("set GATE_E2E_INFERENCE_URL and GATE_E2E_MANAGEMENT_URL for the progress-gate fixture")
	}
	mode := os.Getenv("GATE_E2E_MODE")
	if mode == "" {
		mode = "enforce"
	}
	if mode != "observe" && mode != "enforce" {
		t.Fatalf("unsupported GATE_E2E_MODE %q", mode)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
	defer cancel()
	if err := runProgressGateVertical(ctx, public, management, os.Getenv("GATE_E2E_MANAGEMENT_TOKEN"), mode); err != nil {
		t.Fatal(err)
	}
}
