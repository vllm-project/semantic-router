package v1alpha1

import (
	"path/filepath"
	"reflect"
	"testing"

	"k8s.io/apiextensions-apiserver/pkg/apiserver/schema"
	"k8s.io/apiextensions-apiserver/pkg/apiserver/schema/pruning"
	kubejson "k8s.io/apimachinery/pkg/util/json"
	"sigs.k8s.io/yaml"
)

// EmbeddingGemma has no model runtime family, so the CRD no longer declares
// embedding_models.gemma_model_path. The API server prunes the key with the
// generated schema: under strict field validation (kubectl's default) the
// pruned path is an unknown-field error, and a resource stored before the
// upgrade loses the key when it is read, as if it had never been set.
func TestCRDPrunesTheRetiredGemmaModelPath(t *testing.T) {
	const cr = `
apiVersion: vllm.ai/v1alpha1
kind: SemanticRouter
metadata: {name: r}
spec:
  config:
    embedding_models:
      mmbert_model_path: models/Vela-1.0-Encoder-307M-Embedding
      gemma_model_path: models/embeddinggemma-300m
`
	for _, relative := range []string{"config/crd/bases", "bundle/manifests"} {
		structural := loadCRDStructural(t, filepath.Join("..", "..", relative, "vllm.ai_semanticrouters.yaml"))
		data, err := yaml.YAMLToJSON([]byte(cr))
		if err != nil {
			t.Fatal(err)
		}
		var obj map[string]interface{}
		if err := kubejson.Unmarshal(data, &obj); err != nil {
			t.Fatal(err)
		}
		unknown := pruning.PruneWithOptions(obj, structural, true, schema.UnknownFieldPathOptions{TrackUnknownFieldPaths: true})
		if want := []string{"spec.config.embedding_models.gemma_model_path"}; !reflect.DeepEqual(unknown, want) {
			t.Fatalf("%s: unknown fields %v, want %v", relative, unknown, want)
		}
		embedding := obj["spec"].(map[string]interface{})["config"].(map[string]interface{})["embedding_models"].(map[string]interface{})
		if _, kept := embedding["gemma_model_path"]; kept || embedding["mmbert_model_path"] != "models/Vela-1.0-Encoder-307M-Embedding" {
			t.Fatalf("%s: embedding_models after pruning = %v", relative, embedding)
		}
	}
}
