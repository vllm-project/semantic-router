package controllers

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	apiextensions "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	"k8s.io/apiextensions-apiserver/pkg/apiserver/schema"
	"k8s.io/apiextensions-apiserver/pkg/apiserver/schema/pruning"
	"sigs.k8s.io/yaml"

	vllmv1alpha1 "github.com/vllm-project/semantic-router/operator/api/v1alpha1"
	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestOperatorPrototypeScoringPreservesOverridePresence(t *testing.T) {
	cases := []struct {
		name string
		json string
		want *routerconfig.PrototypeScoringConfig
	}{
		{name: "inherit", json: `{"complexity_rules":[{"name":"rule"}]}`},
		{name: "empty", json: `{"complexity_rules":[{"name":"rule","prototype_scoring":{}}]}`, want: &routerconfig.PrototypeScoringConfig{}},
		{name: "disabled", json: `{"complexity_rules":[{"name":"rule","prototype_scoring":{"enabled":false}}]}`, want: &routerconfig.PrototypeScoringConfig{Enabled: prototypeBool(false)}},
		{name: "complete", json: `{"complexity_rules":[{"name":"rule","prototype_scoring":{"enabled":false,"cluster_similarity_threshold":"0.84","max_prototypes":3,"best_weight":"0.65","top_m":1,"margin_threshold":"0.04"}}]}`, want: &routerconfig.PrototypeScoringConfig{Enabled: prototypeBool(false), ClusterSimilarityThreshold: .84, MaxPrototypes: 3, BestWeight: .65, TopM: 1, MarginThreshold: .04}},
	}
	for _, test := range cases {
		t.Run(test.name, func(t *testing.T) {
			var spec vllmv1alpha1.ConfigSpec
			if err := json.Unmarshal([]byte(test.json), &spec); err != nil {
				t.Fatal(err)
			}
			canonical := &routerconfig.CanonicalConfig{}
			if err := (&SemanticRouterReconciler{}).applyOperatorRouting(canonical, spec); err != nil {
				t.Fatal(err)
			}
			if len(canonical.Routing.Signals.Complexity) != 1 || !reflect.DeepEqual(canonical.Routing.Signals.Complexity[0].PrototypeScoring, test.want) {
				t.Fatalf("override changed during translation: %+v", canonical.Routing.Signals.Complexity)
			}
		})
	}
}

func TestOperatorPrototypeScoringDeepCopy(t *testing.T) {
	original := &vllmv1alpha1.ConfigSpec{ComplexityRules: []vllmv1alpha1.ComplexityRulesConfig{{
		Name: "rule", PrototypeScoring: &vllmv1alpha1.PrototypeScoringConfig{Enabled: prototypeBool(false)},
	}}}
	copy := original.DeepCopy()
	*copy.ComplexityRules[0].PrototypeScoring.Enabled = true
	if *original.ComplexityRules[0].PrototypeScoring.Enabled {
		t.Fatal("deep copy aliases the original override flag")
	}
}

func TestOperatorPrototypeScoringCRDPreservesFields(t *testing.T) {
	for _, path := range []string{"../config/crd/bases/vllm.ai_semanticrouters.yaml", "../bundle/manifests/vllm.ai_semanticrouters.yaml"} {
		t.Run(filepath.Base(filepath.Dir(path)), func(t *testing.T) {
			data, readErr := os.ReadFile(path)
			if readErr != nil {
				t.Fatal(readErr)
			}
			var crd apiextensionsv1.CustomResourceDefinition
			if err := yaml.Unmarshal(data, &crd); err != nil {
				t.Fatal(err)
			}
			v1 := crd.Spec.Versions[0].Schema.OpenAPIV3Schema
			configSchema := v1.Properties["spec"].Properties["config"]
			properties := configSchema.Properties["complexity_rules"].Items.Schema.Properties["prototype_scoring"]
			if len(properties.Properties) != 6 || properties.Default != nil {
				t.Fatalf("missing fields or injected override default: %+v", properties)
			}
			for name, property := range properties.Properties {
				if property.Default != nil {
					t.Fatalf("operator injected a default for %s", name)
				}
			}
			var internal apiextensions.JSONSchemaProps
			if err := apiextensionsv1.Convert_v1_JSONSchemaProps_To_apiextensions_JSONSchemaProps(v1, &internal, nil); err != nil {
				t.Fatal(err)
			}
			structural, err := schema.NewStructural(&internal)
			if err != nil {
				t.Fatal(err)
			}
			var object map[string]interface{}
			if err := json.Unmarshal([]byte(`{"spec":{"config":{"complexity_rules":[{"name":"rule","prototype_scoring":{"enabled":false,"cluster_similarity_threshold":"0.84","max_prototypes":3,"best_weight":"0.65","top_m":1,"margin_threshold":"0.04"}}]}}}`), &object); err != nil {
				t.Fatal(err)
			}
			before, _ := json.Marshal(object)
			pruning.Prune(object, structural, false)
			after, _ := json.Marshal(object)
			if string(before) != string(after) {
				t.Fatalf("CRD pruned prototype scoring fields: %s", after)
			}
		})
	}
}

func prototypeBool(value bool) *bool { return &value }
