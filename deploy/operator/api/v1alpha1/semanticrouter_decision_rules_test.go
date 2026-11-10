package v1alpha1

import (
	"path/filepath"
	"reflect"
	"testing"

	"k8s.io/apiextensions-apiserver/pkg/apiserver/schema"
	"k8s.io/apiextensions-apiserver/pkg/apiserver/schema/pruning"
	"sigs.k8s.io/yaml"
)

// A decision's conditions carry what a Router condition carries: a label, a
// predicate, on_error, and nested condition trees. Without them in the schema
// the API server prunes them and the Router matches a different rule.
func TestCRDKeepsDecisionConditionTrees(t *testing.T) {
	const rules = `
operator: AND
conditions:
  - {type: complexity, name: needs_reasoning, label: hard}
  - {type: systemone, name: risk, label: RISKY, predicate: {gte: 0.7}, on_error: no_match}
  - operator: OR
    conditions:
      - {type: keyword, name: urgent}
      - {operator: NOT, conditions: [{type: domain, name: other}]}
`
	for _, relative := range []string{"config/crd/bases", "bundle/manifests"} {
		structural := loadCRDStructural(t, filepath.Join("..", "..", relative, "vllm.ai_semanticrouters.yaml"))
		var want, written map[string]interface{}
		if err := yaml.Unmarshal([]byte(rules), &want); err != nil {
			t.Fatal(err)
		}
		if err := yaml.Unmarshal([]byte(rules), &written); err != nil {
			t.Fatal(err)
		}
		decision := map[string]interface{}{"name": "nested", "modelRefs": []interface{}{map[string]interface{}{"model": "small"}}, "rules": written}
		obj := map[string]interface{}{
			"apiVersion": "vllm.ai/v1alpha1", "kind": "SemanticRouter", "metadata": map[string]interface{}{"name": "r"},
			"spec": map[string]interface{}{"config": map[string]interface{}{"decisions": []interface{}{decision}}},
		}
		if unknown := pruning.PruneWithOptions(obj, structural, true, schema.UnknownFieldPathOptions{TrackUnknownFieldPaths: true}); len(unknown) > 0 {
			t.Fatalf("%s prunes %v", relative, unknown)
		}
		if !reflect.DeepEqual(decision["rules"], want) {
			t.Fatalf("%s: rules %v after pruning, want %v", relative, decision["rules"], want)
		}
	}
}
