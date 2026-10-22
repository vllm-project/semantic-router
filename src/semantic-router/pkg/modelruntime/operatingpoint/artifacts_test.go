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
	return config.ResolvedModelBinding{Recipe: "one", Name: "classifier.risk", Binding: config.ModelBinding{Contract: config.RemoteClassifierContractLabelScores, Adapter: "modernbert", OperatingPoint: &config.OperatingPointReference{Path: "point.json", SHA256: digest(policy)}}, Deployment: config.ModelDeployment{Provider: config.ModelRuntimeProvider, Artifact: root, Device: "cpu", Input: config.ModelInputBudget{MaxTokens: 10, Overflow: "reject"}}}
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
	if _, err = Decode(got, digest(got)); err != nil {
		t.Fatal("export is not a readable operating point", err)
	}
	if err = os.WriteFile(filepath.Join(spec.Deployment.Artifact, "model.safetensors"), []byte("different checkpoint"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err = BindArtifact(context.Background(), source, spec.Deployment.Artifact); err == nil {
		t.Fatal("writer silently rebound thresholds to different weights")
	}
}
