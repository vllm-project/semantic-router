//go:build !windows && cgo && (amd64 || arm64)

package native

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/operatingpoint"
)

func TestOwnedOperatingPointSharesResourcesNotRecipePolicy(t *testing.T) {
	path := nativeHeadlessFullFixture(t, 1)
	cfgFile := filepath.Join(path, "config.json")
	raw, err := os.ReadFile(cfgFile)
	if err != nil {
		t.Fatal(err)
	}
	var metadata map[string]any
	if err = json.Unmarshal(raw, &metadata); err != nil {
		t.Fatal(err)
	}
	metadata["problem_type"] = "multi_label_classification"
	raw, _ = json.Marshal(metadata)
	if err = os.WriteFile(cfgFile, raw, 0o600); err != nil {
		t.Fatal(err)
	}
	hashes := map[string]string{}
	for _, name := range []string{"config.json", "model.safetensors", "tokenizer.json"} {
		payload, readErr := os.ReadFile(filepath.Join(path, name))
		if readErr != nil {
			t.Fatal(readErr)
		}
		sum := sha256.Sum256(payload)
		hashes[name] = hex.EncodeToString(sum[:])
	}
	zero := 0
	pad := uint32(0)
	d := operatingpoint.Definition{Version: 2, ModelWeightsSHA256: hashes["model.safetensors"], ModelConfigSHA256: hashes["config.json"], TokenizerSHA256: hashes["tokenizer.json"], ScoreType: "independent_sigmoid", Labels: []string{"safe", "unsafe"}, Thresholds: []float32{.5, .5}, Comparison: "score >= threshold", Executions: []operatingpoint.Execution{{Provider: "candle", Precision: "float32", WeightsFile: "model.safetensors"}}, Input: operatingpoint.InputPolicy{Strategy: "overlapping_content_windows", WindowTokens: 4, ContentTokens: 4, Overlap: 1, Stride: 3, MaxDocumentTokens: 32, Aggregation: "per-label maximum sigmoid over all covering windows", Positions: "reset for each window", PaddingSide: "right", PadTokenID: &pad, PaddingAttentionMask: &zero, SpecialPrefixIDs: []uint32{}, SpecialSuffixIDs: []uint32{}, ReferenceWindowBatchSize: 4, BatchOrder: "ascending actual window token count, stable original order on ties", Tokenization: "Tokenize once without truncation; slice original content token IDs and restore the tokenizer special-token envelope for each window.", Overflow: "reject"}}
	data, _ := json.Marshal(d)
	sum := sha256.Sum256(data)
	if err = os.WriteFile(filepath.Join(path, "point.json"), data, 0o600); err != nil {
		t.Fatal(err)
	}
	spec := config.ResolvedModelBinding{Recipe: "one", Name: "classifier.risk", Binding: config.ModelBinding{Deployment: "encoder", Adapter: "modernbert", Contract: config.RemoteClassifierContractLabelScores, OperatingPoint: &config.OperatingPointReference{Path: "point.json", SHA256: hex.EncodeToString(sum[:])}}, Deployment: config.ModelDeployment{Provider: "candle", Device: "cpu", Artifact: path, Precision: "native", Input: config.ModelInputBudget{MaxTokens: 32, Overflow: "reject"}}}
	runtime := New(nil)
	ctx := context.Background()
	first, err := runtime.OperatingPoint(ctx, spec, d.Labels)
	if err != nil {
		t.Fatal(err)
	}
	defer first.Close()
	otherSpec := spec
	otherSpec.Recipe = "two"
	other, err := runtime.OperatingPoint(ctx, otherSpec, d.Labels)
	if err != nil {
		t.Fatal(err)
	}
	defer other.Close()
	result, err := first.Score(ctx, "one", strings.Repeat("hello ", 8))
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Windows) != 3 || result.Windows[2] != [2]int{6, 8} || result.Input.OriginalTokens != 8 || result.Input.Truncated || result.Scores[0]+result.Scores[1] <= 1 {
		t.Fatalf("actual owned scores/odd tail altered: %+v", result)
	}
	if _, err = first.Score(ctx, "two", "hello"); !errors.Is(err, binding.ErrCapability) {
		t.Fatalf("foreign recipe admitted: %v", err)
	}
	if _, err = first.Score(ctx, "one", strings.Repeat("hello ", 33)); !errors.Is(err, binding.ErrInputLimit) {
		t.Fatalf("overflow was not rejected: %v", err)
	}
	if err = first.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err = first.Score(ctx, "one", "hello"); !errors.Is(err, binding.ErrClosed) {
		t.Fatalf("closed model accepted: %v", err)
	}
	if _, err = other.Score(ctx, "two", "hello"); err != nil {
		t.Fatal("sibling close removed live owner", err)
	}
	// A newly self-consistent sidecar must not legitimize the old cached owner
	// after files at the same path change inside this generation.
	metadata["reference_compile"] = false
	raw, _ = json.Marshal(metadata)
	if err = os.WriteFile(cfgFile, raw, 0o600); err != nil {
		t.Fatal(err)
	}
	updated, err := operatingpoint.BindArtifact(ctx, data, path)
	if err != nil {
		t.Fatal(err)
	}
	if err = os.WriteFile(filepath.Join(path, "point.json"), updated, 0o600); err != nil {
		t.Fatal(err)
	}
	updatedHash := sha256.Sum256(updated)
	newRef := *spec.Binding.OperatingPoint
	newRef.SHA256 = hex.EncodeToString(updatedHash[:])
	spec.Binding.OperatingPoint = &newRef
	if handle, bindErr := runtime.OperatingPoint(ctx, spec, d.Labels); bindErr == nil {
		handle.Close()
		t.Fatal("new policy reused an owner from before replacement")
	}
	if _, err = other.Score(ctx, "two", "hello"); err != nil {
		t.Fatal("replacement invalidated existing immutable owner", err)
	}
	next, err := New(runtime.Pool).OperatingPoint(ctx, spec, d.Labels)
	if err != nil {
		t.Fatal("new generation could not admit replacement", err)
	}
	defer next.Close()
}
