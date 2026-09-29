//go:build openvino && !windows && cgo

package native

import (
	"context"
	"errors"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
)

// This test runs exported IRs and their actual tokenizers. No loader/inference
// hook is replaced. The fixture root is deliberately opt-in on OpenVINO hosts.
func TestOwnedOpenVINORuntimeIntegration(t *testing.T) {
	root := os.Getenv("SEMANTIC_ROUTER_OPENVINO_TEST_ARTIFACTS")
	if root == "" {
		t.Skip("requires real OpenVINO IR qualification fixtures")
	}
	ctx := context.Background()
	pool := binding.NewPool()
	current := New(pool)
	next := New(pool)
	spec := func(name, contract string) config.ResolvedModelBinding {
		return config.ResolvedModelBinding{Recipe: "first", Name: "qualified", Binding: config.ModelBinding{Deployment: name, Adapter: "bert", Contract: contract}, Deployment: config.ModelDeployment{Provider: "openvino", Device: "CPU", Precision: "native", Artifact: filepath.Join(root, name), Input: config.ModelInputBudget{MaxTokens: 16, Overflow: "reject"}}}
	}
	t.Run("embedding", func(t *testing.T) {
		a := spec("embedding_a", "embedding.v1")
		first, err := current.Embedding(ctx, a, 0, 0)
		if err != nil {
			t.Fatal(err)
		}
		defer first.Close()
		a.Recipe = "second"
		peer, err := next.Embedding(ctx, a, 0, 0)
		if err != nil {
			t.Fatal(err)
		}
		defer peer.Close()
		var firstEngine, peerEngine io.Closer
		if err = first.resource.Use(ctx, func(value io.Closer) error { firstEngine = value; return nil }); err != nil {
			t.Fatal(err)
		}
		if err = peer.resource.Use(ctx, func(value io.Closer) error { peerEngine = value; return nil }); err != nil {
			t.Fatal(err)
		}
		if firstEngine != peerEngine {
			t.Fatal("same graph/execution created another physical model")
		}
		firstVector, err := first.Embed(ctx, "alpha beta")
		if err != nil {
			t.Fatal(err)
		}
		other, err := current.Embedding(ctx, spec("embedding_b", "embedding.v1"), 0, 0)
		if err != nil {
			t.Fatal(err)
		}
		defer other.Close()
		otherVector, err := other.Embed(ctx, "alpha beta")
		if err != nil {
			t.Fatal(err)
		}
		if reflect.DeepEqual(firstVector, otherVector) {
			t.Fatalf("independent tokenizer/IR outputs conflated: %v", firstVector)
		}
		if _, err = first.EmbedWithOptions(ctx, "alpha", embedding.Options{Layer: 1}); !errors.Is(err, binding.ErrCapability) {
			t.Fatalf("ignored layer: %v", err)
		}
		if _, err = first.EmbedWithOptions(ctx, "alpha", embedding.Options{Dimension: 2}); !errors.Is(err, binding.ErrCapability) {
			t.Fatalf("ignored crop: %v", err)
		}
		if _, err = first.Windows(ctx, "alpha", 16); !errors.Is(err, binding.ErrCapability) {
			t.Fatalf("invented window support: %v", err)
		}
		long := strings.Repeat("alpha ", 80) // exceeds the fixture's declared 64-token capacity
		if _, err = first.Embed(ctx, long); !errors.Is(err, binding.ErrInputLimit) {
			t.Fatalf("reject=%v", err)
		}
		a.Deployment.Input.Overflow = "truncate"
		truncated, err := current.Embedding(ctx, a, 0, 0)
		if err != nil {
			t.Fatal(err)
		}
		defer truncated.Close()
		result, err := truncated.text.Call(ctx, string(a.Recipe), embedding.TextRequest{Text: long})
		if err != nil || result.Input == nil || result.Input.OriginalTokens != 82 || result.Input.ProcessedTokens != 16 || !result.Input.Truncated {
			t.Fatalf("real token usage=%+v err=%v", result.Input, err)
		}
		if err = first.Close(); err != nil {
			t.Fatal(err)
		}
		if _, err = first.Embed(ctx, "alpha"); !errors.Is(err, binding.ErrClosed) {
			t.Fatalf("closed owner callable: %v", err)
		}
		got, err := peer.Embed(ctx, "alpha beta")
		if err != nil || !reflect.DeepEqual(got, firstVector) {
			t.Fatalf("new generation lost shared model: %v %v", got, err)
		}
	})
	t.Run("classifier", func(t *testing.T) {
		a := spec("classifier_a", config.RemoteClassifierContractLabelDistribution)
		first, err := current.Sequence(ctx, a)
		if err != nil {
			t.Fatal(err)
		}
		defer first.Close()
		a.Recipe = "second"
		peer, err := next.Sequence(ctx, a)
		if err != nil {
			t.Fatal(err)
		}
		defer peer.Close()
		if current.PreparedBindings()[0].ResourceID == "" || current.PreparedBindings()[0].ResourceID != next.PreparedBindings()[0].ResourceID {
			t.Fatal("missing real resource identity")
		}
		original, err := first.Call(ctx, "first", "alpha beta")
		if err != nil {
			t.Fatal(err)
		}
		if _, err = peer.Call(ctx, "first", "alpha"); !errors.Is(err, binding.ErrCapability) {
			t.Fatalf("foreign recipe scope accepted: %v", err)
		}
		other, err := current.Sequence(ctx, spec("classifier_b", config.RemoteClassifierContractLabelDistribution))
		if err != nil {
			t.Fatal(err)
		}
		defer other.Close()
		different, err := other.Call(ctx, "first", "alpha beta")
		if err != nil || reflect.DeepEqual(original.Probabilities, different.Probabilities) {
			t.Fatalf("distinct classifier output=%v err=%v", different, err)
		}
		long := strings.Repeat("alpha ", 80)
		if _, err = peer.Call(ctx, "second", long); !errors.Is(err, binding.ErrInputLimit) {
			t.Fatalf("reject=%v", err)
		}
		a.Deployment.Input.Overflow = "truncate"
		truncated, err := current.Sequence(ctx, a)
		if err != nil {
			t.Fatal(err)
		}
		defer truncated.Close()
		result, err := truncated.Call(ctx, "second", long)
		if err != nil || result.Input == nil || result.Input.OriginalTokens != 82 || result.Input.ProcessedTokens != 16 || !result.Input.Truncated {
			t.Fatalf("real token usage=%+v err=%v", result.Input, err)
		}
		if err = first.Close(); err != nil {
			t.Fatal(err)
		}
		result, err = peer.Call(ctx, "second", "alpha beta")
		if err != nil || !reflect.DeepEqual(result.Probabilities, original.Probabilities) {
			t.Fatalf("shared classifier lifetime: %v %v", result, err)
		}
	})
}
