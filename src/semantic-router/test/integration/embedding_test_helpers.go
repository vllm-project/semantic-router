//go:build integration
// +build integration

// Package integration contains integration tests for the semantic router.
// These tests require real ML models and test the full stack:
// Go → CGO → Rust → Candle → GPU/CPU
//
// Run with: make test-embedding
// or: go test -tags=integration -v ./test/integration
package integration

import (
	"testing"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
)

// testModelsDir points to the models directory for integration tests
const testModelsDir = "../../../../models"

// modelsInitialized tracks if embedding models have been initialized
var modelsInitialized bool

// initEmbeddingModelsWithFallback initializes embedding models with graceful fallback
// for gated models like Gemma that may not be available in CI.
// Skips initialization if models are already initialized (for test reuse).
func initEmbeddingModelsWithFallback(t *testing.T, modelsDir string) {
	t.Helper()

	// Skip if already initialized (models persist across tests)
	if modelsInitialized {
		t.Log("✓ Embedding models already initialized, skipping re-initialization")
		return
	}

	qwen := modelsDir + "/Qwen3-Embedding-0.6B"
	gemma := modelsDir + "/embeddinggemma-300m"

	// First attempt
	if err := candle_binding.InitEmbeddingModels(qwen, gemma, true); err == nil {
		t.Log("✓ Embedding models loaded (Qwen3 + Gemma)")
		modelsInitialized = true
		return
	}

	// Fallback: use only Qwen3 (pass empty string for Gemma)
	t.Log("⚠️  Gemma unavailable, falling back to Qwen3-only mode")

	if err := candle_binding.InitEmbeddingModels(qwen, "", true); err != nil {
		t.Fatalf("❌ Embedding initialization failed with fallback: %v", err)
	}

	t.Log("✓ Fallback successful: Qwen3-only mode")
	modelsInitialized = true
}
