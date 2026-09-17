package operatingpoint

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func digest(data []byte) string { sum := sha256.Sum256(data); return hex.EncodeToString(sum[:]) }
func fixtureSpec(t *testing.T, mutate func(map[string]any, map[string]any)) config.ResolvedModelBinding {
	t.Helper()
	root := t.TempDir()
	d := exampleDefinition()
	metadata := map[string]any{"problem_type": "multi_label_classification", "classifier_pooling": "mean", "max_position_embeddings": 10, "pad_token_id": 0, "id2label": map[string]string{"0": "one", "1": "two"}, "label2id": map[string]int{"one": 0, "two": 1}}
	tokenizer := map[string]any{"post_processor": map[string]any{"type": "TemplateProcessing", "single": []any{map[string]any{"SpecialToken": map[string]any{"id": "<bos>", "type_id": 0}}, map[string]any{"Sequence": map[string]any{"id": "A", "type_id": 0}}, map[string]any{"SpecialToken": map[string]any{"id": "<eos>", "type_id": 0}}}, "special_tokens": map[string]any{"<bos>": map[string]any{"ids": []int{2}}, "<eos>": map[string]any{"ids": []int{1}}}}}
	if mutate != nil {
		mutate(metadata, tokenizer)
	}
	cfg, _ := json.Marshal(metadata)
	tok, _ := json.Marshal(tokenizer)
	weights := []byte("fake weights; this test verifies identity, not model loading")
	d.ModelWeightsSHA256 = digest(weights)
	d.ModelConfigSHA256 = digest(cfg)
	d.TokenizerSHA256 = digest(tok)
	policy, _ := json.Marshal(d)
	for name, data := range map[string][]byte{"model.safetensors": weights, "config.json": cfg, "tokenizer.json": tok, "point.json": policy} {
		if err := os.WriteFile(filepath.Join(root, name), data, 0o600); err != nil {
			t.Fatal(err)
		}
	}
	return config.ResolvedModelBinding{Recipe: "one", Name: "classifier.risk", Binding: config.ModelBinding{Contract: config.RemoteClassifierContractLabelScores, Adapter: "modernbert", OperatingPoint: &config.OperatingPointReference{Path: "point.json", SHA256: digest(policy)}}, Deployment: config.ModelDeployment{Provider: "candle", Artifact: root, Device: "cpu", Precision: "native", Input: config.ModelInputBudget{MaxTokens: 10, Overflow: "reject"}}}
}

func TestLoadBindsActualFilesAndMetadata(t *testing.T) {
	ctx := context.Background()
	spec := fixtureSpec(t, nil)
	p, err := Load(ctx, spec, []string{"one", "two"})
	if err != nil {
		t.Fatal(err)
	}
	if _, err = Load(ctx, spec, []string{"two", "one"}); err == nil {
		t.Fatal("rule reordered labels")
	}
	for _, name := range []string{"model.safetensors", "config.json", "tokenizer.json", "point.json"} {
		t.Run(name, func(t *testing.T) {
			candidate := fixtureSpec(t, nil)
			path := filepath.Join(candidate.Deployment.Artifact, name)
			data, _ := os.ReadFile(path)
			if err = os.WriteFile(path, append(data, ' '), 0o600); err != nil {
				t.Fatal(err)
			}
			if _, err = Load(ctx, candidate, []string{"one", "two"}); err == nil {
				t.Fatal("replacement accepted")
			}
		})
	}
	if err = os.WriteFile(filepath.Join(spec.Deployment.Artifact, "model.safetensors"), []byte("changed after prepare"), 0o600); err != nil {
		t.Fatal(err)
	}
	if p.VerifyArtifacts(ctx, spec.Deployment.Artifact) == nil {
		t.Fatal("post-load replacement accepted")
	}
	for name, mutate := range map[string]func(map[string]any, map[string]any){"softmax": func(m, t map[string]any) { m["problem_type"] = "single_label_classification" }, "unknown pooling": func(m, t map[string]any) { m["classifier_pooling"] = "max" }, "missing label": func(m, t map[string]any) { m["label2id"] = map[string]int{"one": 0} }, "short capacity": func(m, t map[string]any) { m["max_position_embeddings"] = 9 }, "envelope": func(m, t map[string]any) { t["post_processor"] = nil }} {
		t.Run(name, func(t *testing.T) {
			if _, err = Load(ctx, fixtureSpec(t, mutate), []string{"one", "two"}); err == nil {
				t.Fatal("incompatible metadata accepted despite self-consistent hashes")
			}
		})
	}
	spec = fixtureSpec(t, nil)
	spec.Binding.Head = "other-head"
	if _, err = Load(ctx, spec, []string{"one", "two"}); err == nil {
		t.Fatal("unbound head accepted")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err = Load(ctx, fixtureSpec(t, nil), []string{"one", "two"}); err == nil {
		t.Fatal("cancelled preparation accepted")
	}
}

func TestExportPreservesScorePolicyAndRejectsChangedWeights(t *testing.T) {
	spec := fixtureSpec(t, nil)
	source, err := os.ReadFile(filepath.Join(spec.Deployment.Artifact, "point.json"))
	if err != nil {
		t.Fatal(err)
	}
	var original map[string]json.RawMessage
	if err = json.Unmarshal(source, &original); err != nil {
		t.Fatal(err)
	}
	original["version"] = json.RawMessage("1")
	delete(original, "model_config_sha256")
	delete(original, "tokenizer_sha256")
	delete(original, "executions")
	source, err = json.Marshal(original)
	if err != nil {
		t.Fatal(err)
	}
	got, err := BindArtifact(context.Background(), source, spec.Deployment.Artifact)
	if err != nil {
		t.Fatal(err)
	}
	var exported map[string]json.RawMessage
	if err = json.Unmarshal(got, &exported); err != nil {
		t.Fatal(err)
	}
	for key, raw := range original {
		if key == "version" {
			continue
		}
		var wantValue, gotValue any
		if err = json.Unmarshal(raw, &wantValue); err != nil {
			t.Fatal(err)
		}
		if err = json.Unmarshal(exported[key], &gotValue); err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(wantValue, gotValue) {
			t.Fatalf("score policy field %s changed", key)
		}
	}
	spec.Binding.OperatingPoint.SHA256 = digest(got)
	if err = os.WriteFile(filepath.Join(spec.Deployment.Artifact, "point.json"), got, 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err = Load(context.Background(), spec, []string{"one", "two"}); err != nil {
		t.Fatal("export is not readable by the actual preparation path", err)
	}
	if err = os.WriteFile(filepath.Join(spec.Deployment.Artifact, "model.safetensors"), []byte("different checkpoint"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err = BindArtifact(context.Background(), source, spec.Deployment.Artifact); err == nil {
		t.Fatal("writer silently rebound thresholds to different weights")
	}
}
