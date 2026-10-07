package v1alpha1

import (
	"path/filepath"
	"testing"

	"k8s.io/apiextensions-apiserver/pkg/apiserver/schema"
	"k8s.io/apiextensions-apiserver/pkg/apiserver/schema/pruning"
	kubejson "k8s.io/apimachinery/pkg/util/json"
	"sigs.k8s.io/yaml"
)

// A decision's reliability and fallback blocks must survive the API server's
// pruning in both generated CRD copies; without them in the schema the server
// drops the blocks and the Router serves the decision without them.
func TestCRDKeepsDecisionReliabilityAndFallback(t *testing.T) {
	const cr = `
apiVersion: vllm.ai/v1alpha1
kind: SemanticRouter
metadata: {name: r}
spec:
  config:
    decisions:
      - name: long_report
        rules: {operator: AND, conditions: [{type: keyword, name: long_report}]}
        modelRefs: [{model: small}, {model: large}]
        reliability:
          total_timeout: 600s
          per_try_timeout: 300s
          retry_count: 1
          retry_on: reset
          retriable_status_codes: [429]
        fallback:
          enabled: true
          max_attempts: 2
          total_timeout: 900s
          per_attempt_timeout: 450s
          retryable_status_codes: [502, 503]
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
		if unknown := pruning.PruneWithOptions(obj, structural, true, schema.UnknownFieldPathOptions{TrackUnknownFieldPaths: true}); len(unknown) > 0 {
			t.Fatalf("%s prunes %v", relative, unknown)
		}
		decision := obj["spec"].(map[string]interface{})["config"].(map[string]interface{})["decisions"].([]interface{})[0].(map[string]interface{})
		reliability, _ := decision["reliability"].(map[string]interface{})
		fallback, _ := decision["fallback"].(map[string]interface{})
		if len(reliability) != 5 || len(fallback) != 5 {
			t.Fatalf("%s: reliability %v, fallback %v after pruning", relative, reliability, fallback)
		}
	}
}
