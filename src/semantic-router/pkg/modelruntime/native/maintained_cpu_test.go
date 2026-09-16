package native

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"regexp"
	"testing"
	"time"

	ort "github.com/vllm-project/semantic-router/onnx-binding/instance"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
)

// This opt-in correctness probe loads maintained weights through both providers
// in one process. It deliberately has no latency or throughput acceptance gate.
// Paths and full repository revisions must be supplied together. Reports contain
// complete observed vectors, including on numerical comparison failures.
func TestMaintainedCPUSameProcess(t *testing.T) {
	intentPath, embeddingPath := os.Getenv("CORE_NATIVE_INTENT_PATH"), os.Getenv("CORE_NATIVE_EMBEDDING_PATH")
	if intentPath == "" && embeddingPath == "" {
		t.Skip("set CORE_NATIVE_INTENT_PATH and CORE_NATIVE_EMBEDDING_PATH to pinned maintained artifacts")
	}
	intentRevision, embeddingRevision := os.Getenv("CORE_NATIVE_INTENT_REVISION"), os.Getenv("CORE_NATIVE_EMBEDDING_REVISION")
	for name, value := range map[string]string{"intent": intentRevision, "embedding": embeddingRevision} {
		if !regexp.MustCompile(`^[0-9a-f]{40}$`).MatchString(value) {
			t.Fatalf("%s revision must be a full pinned repository commit", name)
		}
	}
	if intentPath == "" || embeddingPath == "" {
		t.Fatal("both artifact paths are required")
	}
	report := map[string]any{
		"intent_revision": intentRevision, "embedding_revision": embeddingRevision,
		"intent_path": intentPath, "embedding_path": embeddingPath,
		"protocol":   "same graph/head/tokenizer; CPU float32; embedding layer 6 dimension 256",
		"tolerances": map[string]float64{"probability_max_abs": 1e-3, "probability_sum_error": 1e-3, "embedding_cosine_min": .999, "embedding_max_abs": 2e-3, "embedding_norm_difference": 1e-3},
	}
	reportDir := os.Getenv("CORE_NATIVE_REPORT_DIR")
	if reportDir == "" {
		reportDir = t.TempDir()
	}
	saveReport := func() {
		dir := reportDir
		if err := os.MkdirAll(dir, 0o700); err != nil {
			t.Error(err)
			return
		}
		data, err := json.MarshalIndent(report, "", "  ")
		if err == nil {
			err = os.WriteFile(filepath.Join(dir, "maintained-cpu-results.json"), data, 0o600)
		}
		if err != nil {
			t.Error(err)
		}
	}
	t.Cleanup(saveReport)
	saveReport()
	ctx := context.Background()
	runtime := New(binding.NewPool())
	for name, path := range map[string]string{"intent": intentPath, "embedding": embeddingPath} {
		revision, err := runtime.artifactRevision(ctx, path)
		if err != nil {
			t.Fatal(err)
		}
		report[name+"_contents_sha256"] = revision
	}
	texts := []string{"Explain how a compiler optimizes a program.", "如何计算两个向量之间的余弦相似度？"}
	report["inputs"] = texts
	sequenceResults := map[string][]tasks.LabelDistribution{}
	embeddingResults := map[string][]tasks.EmbeddingResult{}
	labels := map[string][]string{}
	parent := t
	for _, provider := range []string{"candle", "ort"} {
		t.Run(provider, func(t *testing.T) {
			// Cleanup is registered on the parent so Candle models remain loaded
			// while ORT models execute in this same process.
			providerReport := map[string]any{}
			report[provider] = providerReport
			spec := maintainedSpec(provider, intentPath, intentRevision, "label_distribution.v1")
			first, err := runtime.Sequence(ctx, spec)
			if err != nil {
				providerReport["load_error"] = err.Error()
				t.Fatal(err)
			}
			defer first.Close()
			aliasSpec, secondSpec := spec, spec
			aliasSpec.Recipe, aliasSpec.Name = "alias", "shared-intent"
			secondSpec.Name, secondSpec.Deployment.Input.MaxTokens = "independent-intent", 256
			alias, err := runtime.Sequence(ctx, aliasSpec)
			if err != nil {
				t.Fatal(err)
			}
			// Keep aliases alive across both provider subtests.
			parent.Cleanup(func() { _ = alias.Close() })
			second, err := runtime.Sequence(ctx, secondSpec)
			if err != nil {
				t.Fatal(err)
			}
			parent.Cleanup(func() { _ = second.Close() })
			addresses := make([]string, 3)
			sequenceEvidence := make([]any, 3)
			for i, candidate := range []config.ResolvedModelBinding{spec, aliasSpec, secondSpec} {
				resource := maintainedSequenceResource(t, runtime, candidate)
				addresses[i] = maintainedAddress(t, resource)
				sequenceEvidence[i] = maintainedNativeEvidence(t, resource)
				_ = resource.Close()
			}
			providerReport["sequence_resource_addresses"] = addresses
			providerReport["sequence_native_evidence"] = sequenceEvidence
			if addresses[0] != addresses[1] || addresses[0] == addresses[2] {
				t.Fatal("sequence alias or independent physical ownership mismatch")
			}
			capability := first.Capability()
			providerReport["sequence_capability"] = capability
			labels[provider] = capability.Labels
			if capability.Provider != provider || capability.Device != "cpu" || capability.Limits.EffectiveTokens() != 512 || len(capability.Labels) < 2 {
				t.Fatalf("unexpected actual sequence capability: %+v", capability)
			}
			for _, text := range texts {
				result, callErr := first.Call(ctx, "primary", text)
				if callErr != nil {
					t.Fatal(callErr)
				}
				sequenceResults[provider] = append(sequenceResults[provider], result)
				maintainedUsage(t, result.Input, 512)
				if len(result.Probabilities) != len(capability.Labels) {
					t.Fatal("incomplete probability distribution")
				}
			}
			providerReport["sequence_results"] = sequenceResults[provider]
			saveReport()
			if _, err = first.Call(ctx, "foreign", texts[0]); !errors.Is(err, binding.ErrCapability) {
				t.Fatalf("foreign recipe admitted: %v", err)
			}
			cancelled, cancel := context.WithCancel(ctx)
			cancel()
			if _, err = first.Call(cancelled, "primary", texts[0]); !errors.Is(err, context.Canceled) {
				t.Fatalf("cancelled call admitted: %v", err)
			}
			candidate := spec
			candidate.Deployment.Artifact = filepath.Join(t.TempDir(), "missing-candidate")
			if failed, prepareErr := runtime.Sequence(ctx, candidate); prepareErr == nil {
				_ = failed.Close()
				t.Fatal("missing candidate unexpectedly loaded")
			}
			if _, err = first.Call(ctx, "primary", texts[0]); err != nil {
				t.Fatalf("failed candidate broke original binding: %v", err)
			}
			resource := maintainedSequenceResource(t, runtime, spec)
			providerReport["sequence_native_after_calls"] = maintainedNativeEvidence(t, resource)
			providerReport["cancelled_native_forward"] = maintainedDrain(t, resource, spec, texts[0])
			if err = first.Close(); err != nil {
				t.Fatal(err)
			}
			if _, err = first.Call(ctx, "primary", texts[0]); !errors.Is(err, binding.ErrClosed) {
				t.Fatalf("closed binding remains callable: %v", err)
			}
			for _, handle := range []*binding.Resolved[string, tasks.LabelDistribution]{alias, second} {
				if _, err = handle.Call(ctx, handle.Identity().Recipe, texts[0]); err != nil {
					t.Fatalf("independent binding did not survive close: %v", err)
				}
			}
			providerReport["sequence_ownership_checks_passed"] = true
			saveReport()

			embedSpec := maintainedSpec(provider, embeddingPath, embeddingRevision, "embedding.v1")
			firstEmbedding, err := runtime.Embedding(ctx, embedSpec, 256, 6)
			if err != nil {
				providerReport["embedding_load_error"] = err.Error()
				t.Fatal(err)
			}
			defer firstEmbedding.Close()
			aliasEmbedSpec, secondEmbedSpec := embedSpec, embedSpec
			aliasEmbedSpec.Recipe, aliasEmbedSpec.Name = "alias", "shared-embedding"
			secondEmbedSpec.Name, secondEmbedSpec.Deployment.Input.MaxTokens = "independent-embedding", 256
			aliasEmbedding, err := runtime.Embedding(ctx, aliasEmbedSpec, 256, 6)
			if err != nil {
				t.Fatal(err)
			}
			parent.Cleanup(func() { _ = aliasEmbedding.Close() })
			secondEmbedding, err := runtime.Embedding(ctx, secondEmbedSpec, 256, 6)
			if err != nil {
				t.Fatal(err)
			}
			parent.Cleanup(func() { _ = secondEmbedding.Close() })
			embeddingAddresses := []string{maintainedAddress(t, firstEmbedding.resource), maintainedAddress(t, aliasEmbedding.resource), maintainedAddress(t, secondEmbedding.resource)}
			providerReport["embedding_resource_addresses"] = embeddingAddresses
			if embeddingAddresses[0] != embeddingAddresses[1] || embeddingAddresses[0] == embeddingAddresses[2] {
				t.Fatal("embedding alias or independent physical ownership mismatch")
			}
			providerReport["embedding_capability"] = firstEmbedding.text.Capability()
			semantics := firstEmbedding.text.Capability().Embedding
			if semantics == nil || semantics.Dimension != 256 || semantics.Layer != 6 || semantics.Normalization != "l2" || !reflect.DeepEqual(semantics.Modalities, []string{"text"}) {
				t.Fatalf("unexpected actual embedding semantics: %+v", semantics)
			}
			for _, text := range texts {
				result, callErr := firstEmbedding.text.Call(ctx, "primary", embedding.TextRequest{Text: text, Options: embedding.Options{Dimension: 256, Layer: 6}})
				if callErr != nil {
					t.Fatal(callErr)
				}
				embeddingResults[provider] = append(embeddingResults[provider], result)
				maintainedUsage(t, result.Input, 512)
			}
			providerReport["embedding_results"] = embeddingResults[provider]
			providerReport["embedding_native_after_calls"] = maintainedNativeEvidence(t, firstEmbedding.resource)
			saveReport()
			if _, err = firstEmbedding.text.Call(ctx, "foreign", embedding.TextRequest{Text: texts[0]}); !errors.Is(err, binding.ErrCapability) {
				t.Fatalf("foreign recipe embedding admitted: %v", err)
			}
			candidate = embedSpec
			candidate.Deployment.Artifact = filepath.Join(t.TempDir(), "missing-embedding-candidate")
			if failed, prepareErr := runtime.Embedding(ctx, candidate, 256, 6); prepareErr == nil {
				_ = failed.Close()
				t.Fatal("missing embedding candidate unexpectedly loaded")
			}
			if _, err = firstEmbedding.Embed(ctx, texts[0]); err != nil {
				t.Fatalf("failed candidate broke original embedding: %v", err)
			}
			if _, err = firstEmbedding.Embed(cancelled, texts[0]); !errors.Is(err, context.Canceled) {
				t.Fatalf("cancelled embedding call admitted: %v", err)
			}
			if err = firstEmbedding.Close(); err != nil {
				t.Fatal(err)
			}
			for _, handle := range []*EmbeddingProvider{aliasEmbedding, secondEmbedding} {
				if _, err = handle.Embed(ctx, texts[0]); err != nil {
					t.Fatalf("independent embedding did not survive close: %v", err)
				}
			}
			providerReport["embedding_ownership_checks_passed"] = true
			saveReport()
		})
	}
	if len(sequenceResults["candle"]) == len(texts) && len(sequenceResults["ort"]) == len(texts) {
		if !reflect.DeepEqual(labels["candle"], labels["ort"]) {
			t.Error("providers expose different ordered labels")
		}
		for i := range texts {
			a, b := sequenceResults["candle"][i].Probabilities, sequenceResults["ort"][i].Probabilities
			metrics := maintainedVectorMetrics(a, b)
			report[fmt.Sprintf("sequence_comparison_%d", i)] = metrics
			if len(a) != len(b) || metrics["max_abs"] > 1e-3 || math.Abs(metrics["sum_a"]-1) > 1e-3 || math.Abs(metrics["sum_b"]-1) > 1e-3 || maintainedArgmax(a) != maintainedArgmax(b) {
				t.Errorf("sequence %d exceeds frozen tolerance: %+v", i, metrics)
			}
		}
	}
	if len(embeddingResults["candle"]) == len(texts) && len(embeddingResults["ort"]) == len(texts) {
		for i := range texts {
			a, b := embeddingResults["candle"][i].Embedding, embeddingResults["ort"][i].Embedding
			metrics := maintainedVectorMetrics(a, b)
			report[fmt.Sprintf("embedding_comparison_%d", i)] = metrics
			if len(a) != 256 || len(b) != 256 || metrics["cosine"] < .999 || metrics["max_abs"] > 2e-3 || math.Abs(metrics["norm_a"]-metrics["norm_b"]) > 1e-3 {
				t.Errorf("embedding %d exceeds frozen tolerance: %+v", i, metrics)
			}
		}
	}
}

func maintainedSpec(provider, path, revision, contract string) config.ResolvedModelBinding {
	precision := "fp32"
	if provider == "ort" {
		precision = "native"
	}
	return config.ResolvedModelBinding{Recipe: "primary", Name: contract, Binding: config.ModelBinding{Deployment: provider + "-cpu", Contract: contract, Adapter: "mmbert"}, Deployment: config.ModelDeployment{Artifact: path, Revision: revision, Provider: provider, Device: "cpu", Precision: precision, Input: config.ModelInputBudget{MaxTokens: 512, Overflow: "truncate"}}}
}

func maintainedSequenceResource(t *testing.T, runtime *Runtime, spec config.ResolvedModelBinding) *binding.Resource {
	t.Helper()
	var resource *binding.Resource
	var err error
	if spec.Deployment.Provider == "candle" {
		resource, err = runtime.candleResource(context.Background(), spec, false)
	} else {
		resource, err = runtime.ortResource(context.Background(), spec, "sequence", func(options ort.Options) (io.Closer, error) { return ort.LoadSequenceClassifier(options) })
	}
	if err != nil {
		t.Fatal(err)
	}
	return resource
}

func maintainedAddress(t *testing.T, resource *binding.Resource) string {
	t.Helper()
	var address string
	if err := resource.Use(context.Background(), func(value io.Closer) error { address = fmt.Sprintf("%p", value); return nil }); err != nil {
		t.Fatal(err)
	}
	return address
}

func maintainedNativeEvidence(t *testing.T, resource *binding.Resource) any {
	t.Helper()
	var evidence any
	err := resource.Use(context.Background(), func(value io.Closer) error {
		var err error
		switch model := value.(type) {
		case *candleBackbone:
			if model.encoder != nil {
				evidence, err = model.encoder.Info()
			} else {
				evidence, err = model.sequence.Info()
			}
		case *ort.SequenceClassifier:
			evidence, err = model.Info()
		case *embeddingEngine:
			if model.candle != nil {
				evidence, err = model.candle.Info()
			} else {
				evidence, err = model.ort.Info()
			}
		default:
			return fmt.Errorf("unexpected maintained resource type %T", value)
		}
		return err
	})
	if err != nil {
		t.Fatal(err)
	}
	return evidence
}

func maintainedUsage(t *testing.T, usage *tasks.InputUsage, budget int) {
	t.Helper()
	if usage == nil || usage.OriginalTokens <= 0 || usage.ProcessedTokens <= 0 || usage.ProcessedTokens > budget || usage.Truncated {
		t.Errorf("missing or inconsistent actual input counters: %+v", usage)
	}
}

func maintainedDrain(t *testing.T, resource *binding.Resource, spec config.ResolvedModelBinding, text string) any {
	t.Helper()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	entered, forwarded, release := make(chan struct{}), make(chan struct{}), make(chan struct{})
	callDone, closeDone := make(chan error, 1), make(chan error, 1)
	closeStarted := make(chan struct{})
	var observed any
	go func() {
		callDone <- resource.Use(ctx, func(value io.Closer) error {
			close(entered)
			var err error
			if spec.Deployment.Provider == "candle" {
				backbone := value.(*candleBackbone)
				if backbone.encoder != nil {
					path := spec.Binding.Head
					if path == "" {
						path = spec.Deployment.Artifact
					}
					model, bindErr := backbone.encoder.BindSequenceHead(path)
					if bindErr != nil {
						close(forwarded)
						<-release
						return bindErr
					}
					defer model.Close()
					observed, err = model.Classify(text)
				} else {
					observed, err = backbone.sequence.Classify(text)
				}
			} else {
				observed, err = value.(*ort.SequenceClassifier).Classify(text)
			}
			close(forwarded)
			<-release
			return err
		})
	}()
	<-entered
	cancel()
	go func() { close(closeStarted); closeDone <- resource.Close() }()
	<-closeStarted
	<-forwarded
	select {
	case err := <-closeDone:
		t.Errorf("Close returned before native Use drained: %v", err)
	case <-time.After(25 * time.Millisecond):
	}
	close(release)
	if err := <-callDone; !errors.Is(err, context.Canceled) {
		t.Errorf("native call cancellation result: %v", err)
	}
	select {
	case err := <-closeDone:
		if err != nil {
			t.Error(err)
		}
	case <-time.After(10 * time.Second):
		t.Error("Close did not finish after native Use drained")
	}
	return observed
}

func maintainedArgmax(values []float32) int {
	best := 0
	for i := range values {
		if values[i] > values[best] {
			best = i
		}
	}
	return best
}

func maintainedVectorMetrics(a, b []float32) map[string]float64 {
	out := map[string]float64{}
	for _, value := range a {
		out["sum_a"] += float64(value)
		out["norm_a"] += float64(value) * float64(value)
	}
	for _, value := range b {
		out["sum_b"] += float64(value)
		out["norm_b"] += float64(value) * float64(value)
	}
	var dot float64
	for i := 0; i < len(a) && i < len(b); i++ {
		out["max_abs"] = math.Max(out["max_abs"], math.Abs(float64(a[i])-float64(b[i])))
		dot += float64(a[i]) * float64(b[i])
	}
	out["norm_a"], out["norm_b"] = math.Sqrt(out["norm_a"]), math.Sqrt(out["norm_b"])
	if denominator := out["norm_a"] * out["norm_b"]; denominator > 0 {
		out["cosine"] = dot / denominator
	}
	return out
}
