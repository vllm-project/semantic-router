package handlers

import (
	"strings"
	"testing"
)

func TestDetectRouterRuntimeStatusDownloadingModels(t *testing.T) {
	logContent := strings.Join([]string{
		`{"msg":"Installing required models..."}`,
		`{"msg":"models/mmbert32k-intent-classifier-merged (ready)"}`,
		`{"msg":"✗ models/mmbert32k-jailbreak-detector-merged (need download)"}`,
		`{"msg":"Downloading model: models/mmbert32k-jailbreak-detector-merged"}`,
	}, "\n")

	runtime := detectRouterRuntimeStatus(logContent, true)
	if runtime == nil {
		t.Fatalf("expected router runtime status")
	}
	if runtime.Phase != "downloading_models" {
		t.Fatalf("expected downloading_models phase, got %q", runtime.Phase)
	}
	if runtime.Ready {
		t.Fatalf("expected runtime to be non-ready during download")
	}
	if runtime.ReadyModels != 1 || runtime.TotalModels != 2 {
		t.Fatalf("expected ready/total counts 1/2, got %d/%d", runtime.ReadyModels, runtime.TotalModels)
	}
	if runtime.DownloadingModel != "models/mmbert32k-jailbreak-detector-merged" {
		t.Fatalf("unexpected downloading model: %q", runtime.DownloadingModel)
	}
}

func TestDetectRouterRuntimeStatusInitializingModels(t *testing.T) {
	logContent := strings.Join([]string{
		`{"msg":"Installing required models..."}`,
		`{"msg":"All required models are ready"}`,
		`{"msg":"Initializing embedding models: qwen3=\"models/qwen\" useCPU=true"}`,
	}, "\n")

	runtime := detectRouterRuntimeStatus(logContent, false)
	if runtime == nil {
		t.Fatalf("expected router runtime status")
	}
	if runtime.Phase != "initializing_models" {
		t.Fatalf("expected initializing_models phase, got %q", runtime.Phase)
	}
	if runtime.Ready {
		t.Fatalf("expected runtime to be non-ready while initializing models")
	}
}

func TestDetectRouterRuntimeStatusReady(t *testing.T) {
	logContent := strings.Join([]string{
		`{"msg":"Installing required models..."}`,
		`{"msg":"models/mmbert32k-jailbreak-detector-merged (ready)"}`,
		`{"msg":"All required models are ready"}`,
		`{"msg":"Unified embedding models initialized successfully"}`,
		`{"msg":"Starting insecure LLM Router ExtProc server on port 50051..."}`,
	}, "\n")

	runtime := detectRouterRuntimeStatus(logContent, true)
	if runtime == nil {
		t.Fatalf("expected router runtime status")
	}
	if runtime.Phase != "ready" {
		t.Fatalf("expected ready phase, got %q", runtime.Phase)
	}
	if !runtime.Ready {
		t.Fatalf("expected runtime to be ready")
	}
}

func TestDetectRouterRuntimeStatusStructuredEvents(t *testing.T) {
	logContent := strings.Join([]string{
		`{"event":"required_models_check_started","component":"router"}`,
		`{"msg":"models/mmbert32k-jailbreak-detector-merged (ready)"}`,
		`{"event":"required_models_ready","component":"router"}`,
		`{"event":"embedding_models_init_started","component":"router"}`,
		`{"event":"embedding_models_initialized","component":"router"}`,
		`{"event":"startup_complete","component":"router"}`,
	}, "\n")

	runtime := detectRouterRuntimeStatus(logContent, true)
	if runtime == nil {
		t.Fatalf("expected router runtime status")
	}
	if runtime.Phase != "ready" {
		t.Fatalf("expected ready phase, got %q", runtime.Phase)
	}
	if !runtime.Ready {
		t.Fatalf("expected runtime to be ready")
	}
}
