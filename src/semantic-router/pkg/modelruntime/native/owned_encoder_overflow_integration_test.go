//go:build !windows && cgo && (amd64 || arm64)

package native

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
)

const encoderOverflowBudget = 32768

// These small, real ONNX graphs execute all input IDs without requiring a
// checkpoint download. Their tokenizer adds CLS/SEP, so the 32K budget must
// include special tokens and preserve SEP when truncating the content prefix.
func TestORTOwnedEncoder32KOverflowPolicies(t *testing.T) {
	if os.Getenv("ORT_DYLIB_PATH") == "" {
		t.Skip("requires the real ONNX Runtime library")
	}
	for _, kind := range []string{"sequence", "embedding"} {
		t.Run(kind, func(t *testing.T) {
			spec := encoderOverflowFixture(t, kind)
			runtime := New(nil)
			prefix := strings.Repeat("hello ", encoderOverflowBudget-3) + "world"
			overflow := prefix + " 秘密"

			truncate := prepareOverflowEncoder(t, runtime, spec, kind)
			for _, input := range []struct {
				name, text string
				tokens     int
			}{
				{"short", "hello world", 4},
				{"exact_budget", prefix, encoderOverflowBudget},
			} {
				t.Run(input.name, func(t *testing.T) {
					_, usage, err := truncate.call("primary", input.text)
					require.NoError(t, err)
					require.Equal(t, &tasks.InputUsage{OriginalTokens: input.tokens, ProcessedTokens: input.tokens}, usage)
				})
			}
			reference, _, err := truncate.call("primary", prefix)
			require.NoError(t, err)
			actual, usage, err := truncate.call("primary", overflow)
			require.NoError(t, err)
			require.Equal(t, &tasks.InputUsage{OriginalTokens: encoderOverflowBudget + 1, ProcessedTokens: encoderOverflowBudget, Truncated: true}, usage)
			// Equal metadata alone would also pass if the native graph still saw
			// the discarded token. Check the actual probabilities/vector too.
			require.Equal(t, reference, actual)

			rejectSpec := spec
			rejectSpec.Recipe = "explicit-reject"
			rejectSpec.Deployment.Input.Overflow = "reject"
			reject := prepareOverflowEncoder(t, runtime, rejectSpec, kind)
			rejectedReference, fullUsage, err := reject.call("explicit-reject", prefix)
			require.NoError(t, err)
			require.Equal(t, reference, rejectedReference)
			require.Equal(t, &tasks.InputUsage{OriginalTokens: encoderOverflowBudget, ProcessedTokens: encoderOverflowBudget}, fullUsage)
			_, _, err = reject.call("explicit-reject", overflow)
			require.ErrorIs(t, err, binding.ErrInputLimit)

			// Another recipe may own the same physical graph and truncation
			// policy, but neither handle grants access to the other's recipe.
			peerSpec := spec
			peerSpec.Recipe = "secondary"
			peer := prepareOverflowEncoder(t, runtime, peerSpec, kind)
			_, _, err = truncate.call("secondary", overflow)
			require.ErrorIs(t, err, binding.ErrCapability)
			_, _, err = peer.call("primary", overflow)
			require.ErrorIs(t, err, binding.ErrCapability)
			require.NoError(t, truncate.close())
			_, _, err = truncate.call("primary", prefix)
			require.ErrorIs(t, err, binding.ErrClosed)
			peerOutput, peerUsage, err := peer.call("secondary", overflow)
			require.NoError(t, err)
			require.Equal(t, actual, peerOutput)
			require.Equal(t, usage, peerUsage)
		})
	}
}

type overflowEncoder struct {
	call  func(recipe, text string) ([]float32, *tasks.InputUsage, error)
	close func() error
}

func prepareOverflowEncoder(t *testing.T, runtime *Runtime, spec config.ResolvedModelBinding, kind string) overflowEncoder {
	t.Helper()
	ctx := context.Background()
	var result overflowEncoder
	var capability binding.Capability
	if kind == "sequence" {
		handle, err := runtime.Sequence(ctx, spec)
		require.NoError(t, err)
		capability = handle.Capability()
		result.close = handle.Close
		result.call = func(recipe, text string) ([]float32, *tasks.InputUsage, error) {
			output, err := handle.Call(ctx, recipe, text)
			return output.Probabilities, output.Input, err
		}
	} else {
		provider, err := runtime.Embedding(ctx, spec, 3, 0)
		require.NoError(t, err)
		capability = provider.text.Capability()
		result.close = provider.Close
		result.call = func(recipe, text string) ([]float32, *tasks.InputUsage, error) {
			output, err := provider.text.Call(ctx, recipe, embedding.TextRequest{Text: text, Options: embedding.Options{Dimension: 3}})
			return output.Embedding, output.Input, err
		}
	}
	t.Cleanup(func() { require.NoError(t, result.close()) })
	require.Equal(t, "ort", capability.Provider)
	require.Equal(t, "cpu", capability.Device)
	require.Equal(t, binding.Limits{ModelTokens: encoderOverflowBudget, TaskTokens: encoderOverflowBudget, DeploymentTokens: encoderOverflowBudget, Overflow: spec.Deployment.Input.Overflow}, capability.Limits)
	return result
}

func encoderOverflowFixture(t *testing.T, kind string) config.ResolvedModelBinding {
	t.Helper()
	directory := t.TempDir()
	fixture := filepath.Join("..", "..", "..", "..", "..", "onnx-binding", "instance", "testdata", kind)
	for _, name := range []string{"model.onnx", "config.json", "tokenizer.json"} {
		data, err := os.ReadFile(filepath.Join(fixture, name))
		require.NoError(t, err)
		if name != "model.onnx" {
			var document map[string]any
			require.NoError(t, json.Unmarshal(data, &document))
			if name == "config.json" {
				document["max_position_embeddings"] = encoderOverflowBudget
			} else {
				document["post_processor"] = map[string]any{"type": "BertProcessing", "sep": []any{"test", 4}, "cls": []any{"[UNK]", 0}}
			}
			data, err = json.Marshal(document)
			require.NoError(t, err)
		}
		require.NoError(t, os.WriteFile(filepath.Join(directory, name), data, 0o600))
	}
	contract := config.RemoteClassifierContractLabelDistribution
	if kind == "embedding" {
		contract = "embedding.v1"
	}
	return config.ResolvedModelBinding{
		Recipe: "primary", Name: kind,
		Binding: config.ModelBinding{Deployment: kind + "-encoder", Adapter: "mmbert", Contract: contract, Head: "model.onnx"},
		Deployment: config.ModelDeployment{
			Artifact: directory, Provider: "ort", Device: "cpu", Precision: "native",
			Input: config.ModelInputBudget{MaxTokens: encoderOverflowBudget, Overflow: "truncate"},
		},
	}
}
