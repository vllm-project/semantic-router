package k8s

import (
	"encoding/json"
	"reflect"
	"testing"

	"gopkg.in/yaml.v2"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/apis/vllm.ai/v1alpha1"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestRoutePrototypeScoringPreservesOverrideAndOwnership(t *testing.T) {
	disabled := false
	cases := []struct {
		object string
		want   *config.PrototypeScoringConfig
	}{
		{``, nil},
		{`,"prototypeScoring":{}`, &config.PrototypeScoringConfig{}},
		{
			`,"prototypeScoring":{"enabled":false,"clusterSimilarityThreshold":0.8,"maxPrototypes":12,"bestWeight":0.75,"topM":2,"marginThreshold":0.1}`,
			&config.PrototypeScoringConfig{Enabled: &disabled, ClusterSimilarityThreshold: 0.8, MaxPrototypes: 12, BestWeight: 0.75, TopM: 2, MarginThreshold: 0.1},
		},
	}
	for _, tc := range cases {
		var signal v1alpha1.EmbeddingSignal
		if err := json.Unmarshal([]byte(`{"name":"intent","threshold":0.8,"candidates":["sample"]`+tc.object+`}`), &signal); err != nil {
			t.Fatal(err)
		}
		yamlBytes, err := yaml.Marshal(signal)
		if err != nil {
			t.Fatal(err)
		}
		var fromYAML v1alpha1.EmbeddingSignal
		if decodeErr := yaml.Unmarshal(yamlBytes, &fromYAML); decodeErr != nil {
			t.Fatal(decodeErr)
		}
		if !reflect.DeepEqual(fromYAML, signal) {
			t.Fatal("JSON/YAML field names changed the prototype override")
		}
		route := &v1alpha1.IntelligentRoute{Spec: v1alpha1.IntelligentRouteSpec{Signals: v1alpha1.Signals{Embeddings: []v1alpha1.EmbeddingSignal{signal}}}}
		got := convertSignals(route.DeepCopy().Spec.Signals).Embeddings[0].PrototypeScoring
		if !reflect.DeepEqual(got, tc.want) {
			t.Fatalf("override = %+v, want %+v", got, tc.want)
		}
		if got != nil && got.Enabled != nil {
			*got.Enabled = true
			if *signal.PrototypeScoring.Enabled {
				t.Fatal("converted override shares the API object's boolean")
			}
		}
	}
}
