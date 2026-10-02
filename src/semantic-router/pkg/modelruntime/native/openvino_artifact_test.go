package native

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
)

func openvinoMetadataFixture(t *testing.T) (string, map[string]any) {
	t.Helper()
	dir := t.TempDir()
	metadata := map[string]any{"max_position_embeddings": 64, "pad_token_id": 0, "id2label": map[string]string{"0": "no", "1": "yes"}}
	writeOVJSON(t, dir, "config.json", metadata)
	writeOVJSON(t, dir, "tokenizer.json", map[string]any{"post_processor": map[string]any{"type": "TemplateProcessing", "single": []any{map[string]any{"SpecialToken": map[string]any{"id": "CLS"}}, map[string]any{"Sequence": map[string]any{"id": "A"}}, map[string]any{"SpecialToken": map[string]any{"id": "SEP"}}}, "special_tokens": map[string]any{"CLS": map[string]any{"ids": []int{91}}, "SEP": map[string]any{"ids": []int{93, 94}}}}})
	if err := os.WriteFile(filepath.Join(dir, "openvino_model.xml"), []byte("fixture metadata only"), 0o600); err != nil {
		t.Fatal(err)
	}
	graph := `<net><layers><layer type="WordpieceTokenizer"/><layer type="RaggedToDense"><data m_pad_max_length="false"/></layer></layers></net>`
	if err := os.WriteFile(filepath.Join(dir, "openvino_tokenizer.xml"), []byte(graph), 0o600); err != nil {
		t.Fatal(err)
	}
	return dir, metadata
}

func writeOVJSON(t *testing.T, dir, name string, value any) {
	t.Helper()
	data, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	if err = os.WriteFile(filepath.Join(dir, name), data, 0o600); err != nil {
		t.Fatal(err)
	}
}

func TestOpenVINOArtifactBudgetsAndTokenizerEnvelope(t *testing.T) {
	dir, metadata := openvinoMetadataFixture(t)
	spec := config.ResolvedModelBinding{Deployment: config.ModelDeployment{Provider: "openvino", Artifact: dir, Input: config.ModelInputBudget{MaxTokens: 32, Overflow: "truncate"}}}
	a, err := readOpenVINOArtifact(spec)
	if err != nil || a.ModelTokens != 64 || a.MaxTokens != 32 || a.Device != "CPU" || a.PadTokenID != 0 || !reflect.DeepEqual(a.EndTokenIDs, []int{93, 94}) || !reflect.DeepEqual(a.Labels, []string{"no", "yes"}) {
		t.Fatalf("artifact=%+v err=%v", a, err)
	}
	writeOVJSON(t, dir, "tokenizer_config.json", map[string]any{"model_max_length": 24})
	if _, err = readOpenVINOArtifact(spec); !errors.Is(err, binding.ErrCapability) {
		t.Fatalf("accepted export budget overflow: %v", err)
	}
	spec.Deployment.Input.MaxTokens = 16
	a, err = readOpenVINOArtifact(spec)
	if err != nil || a.ModelTokens != 24 {
		t.Fatalf("tokenizer cap=%+v %v", a, err)
	}
	delete(metadata, "pad_token_id")
	writeOVJSON(t, dir, "config.json", metadata)
	if _, err = readOpenVINOArtifact(spec); !errors.Is(err, binding.ErrCapability) {
		t.Fatalf("guessed padding token: %v", err)
	}
}

func TestOpenVINOSuffixUsesDeclaredTokensOnly(t *testing.T) {
	for _, kind := range []string{"BertProcessing", "RobertaProcessing"} {
		got := openVINOSuffixTokens(map[string]any{"type": kind, "cls": []any{"start", float64(99)}, "sep": []any{"end", float64(78)}})
		if !reflect.DeepEqual(got, []int{78}) {
			t.Fatalf("%s: %v", kind, got)
		}
	}
	if got := openVINOSuffixTokens(map[string]any{"type": "ByteLevel"}); len(got) != 0 {
		t.Fatalf("invented suffix: %v", got)
	}
}

func TestOpenVINOTokenizerIRRejectsHiddenTruncation(t *testing.T) {
	dir, _ := openvinoMetadataFixture(t)
	graph := filepath.Join(dir, "openvino_model.xml")
	for _, operator := range []string{"Minimum", "Clamp", "Slice", "StridedSlice", "Loop", "UnknownTruncation"} {
		content := `<net><layers><layer type="` + operator + `"/></layers></net>`
		if err := os.WriteFile(filepath.Join(dir, "openvino_tokenizer.xml"), []byte(content), 0o600); err != nil {
			t.Fatal(err)
		}
		if err := validateOpenVINOTokenizerIR(graph); !errors.Is(err, binding.ErrCapability) {
			t.Fatalf("accepted %s: %v", operator, err)
		}
	}
	for _, attributes := range []string{`m_pad_max_length="true"`, ``} {
		content := `<net><layers><layer type="RaggedToDense"><data ` + attributes + `/></layer></layers></net>`
		if err := os.WriteFile(filepath.Join(dir, "openvino_tokenizer.xml"), []byte(content), 0o600); err != nil {
			t.Fatal(err)
		}
		if err := validateOpenVINOTokenizerIR(graph); !errors.Is(err, binding.ErrCapability) {
			t.Fatalf("accepted fixed/unknown dense output: %v", err)
		}
	}
}

func TestOpenVINOQualifiedFixtureMetadata(t *testing.T) {
	root := os.Getenv("SEMANTIC_ROUTER_OPENVINO_TEST_ARTIFACTS")
	if root == "" {
		t.Skip("requires qualified exported IR fixture")
	}
	for _, name := range []string{"embedding_a", "embedding_b", "classifier_a", "classifier_b"} {
		a, err := readOpenVINOArtifact(config.ResolvedModelBinding{Deployment: config.ModelDeployment{Provider: "openvino", Artifact: filepath.Join(root, name), Input: config.ModelInputBudget{MaxTokens: 16, Overflow: "truncate"}}})
		if err != nil || a.MaxTokens != 16 || a.ModelTokens != 64 || a.PadTokenID != 0 || !reflect.DeepEqual(a.EndTokenIDs, []int{2}) {
			t.Fatalf("%s metadata=%+v err=%v", name, a, err)
		}
	}
}
