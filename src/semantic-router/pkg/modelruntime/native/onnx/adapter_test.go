package onnx

import (
	"context"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/native"
)

func TestAdapter_UnsupportedLifecycle(t *testing.T) {
	adapter := NewAdapter()
	ctx := context.Background()

	t.Run("LoadModel returns error", func(t *testing.T) {
		req := native.LoadRequest{ModelRef: "test-model"}
		_, err := adapter.LoadModel(ctx, req)
		if err == nil {
			t.Fatal("Expected error from LoadModel, got nil")
		}
		if err.Error() != "onnx native adapter lifecycle is not yet wired (Phase 3)" {
			t.Errorf("Unexpected error message: %v", err)
		}
	})

	t.Run("UnloadModel returns error", func(t *testing.T) {
		handle := &onnxHandle{id: "test-model"}
		err := adapter.UnloadModel(ctx, handle)
		if err == nil {
			t.Fatal("Expected error from UnloadModel, got nil")
		}
		if err.Error() != "onnx native adapter lifecycle is not yet wired (Phase 3)" {
			t.Errorf("Unexpected error message: %v", err)
		}
	})

	t.Run("Inference returns error", func(t *testing.T) {
		handle := &onnxHandle{id: "test-model"}
		req := native.InferenceRequest{}
		_, err := adapter.Inference(ctx, handle, req)
		if err == nil {
			t.Fatal("Expected error from Inference, got nil")
		}
		if err.Error() != "onnx native adapter inference is not yet wired (Phase 3)" {
			t.Errorf("Unexpected error message: %v", err)
		}
	})
}
