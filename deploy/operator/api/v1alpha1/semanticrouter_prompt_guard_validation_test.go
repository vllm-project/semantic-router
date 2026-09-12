package v1alpha1

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	"k8s.io/apiextensions-apiserver/pkg/apiserver/schema/pruning"
	"sigs.k8s.io/yaml"
)

func TestPromptGuardContextAdmission(t *testing.T) {
	cases := []struct {
		name, config string
		wantError    bool
	}{
		{"omitted", `{}`, false},
		{"nullable", `{"window":null}`, false},
		{"explicit whole input", `{"max_sequence_length":32768}`, false},
		{"local window", `{"variant":"mmbert32k","max_sequence_length":32768,"window":{"size":128,"overlap":63}}`, false},
		{"zero budget retains default", `{"max_sequence_length":0,"window":{"size":512}}`, false},
		{"omitted budget retains default", `{"window":{"size":512}}`, false},
		{"legacy candle", `{"variant":"candle"}`, false},
		{"remote without budget", `{"protocol":"http_classify"}`, false},
		{"negative budget", `{"max_sequence_length":-1}`, true},
		{"remote budget", `{"protocol":"http_chat","max_sequence_length":32768}`, true},
		{"candle budget", `{"variant":"candle","max_sequence_length":32768}`, true},
		{"remote window", `{"protocol":"http_classify","window":{"size":128}}`, true},
		{"candle window", `{"variant":"candle","window":{"size":128}}`, true},
		{"missing size", `{"window":{}}`, true},
		{"zero size", `{"window":{"size":0}}`, true},
		{"negative size", `{"window":{"size":-1}}`, true},
		{"exceeds implicit budget", `{"window":{"size":513}}`, true},
		{"exceeds explicit budget", `{"max_sequence_length":64,"window":{"size":128}}`, true},
		{"negative overlap", `{"window":{"size":128,"overlap":-1}}`, true},
		{"overlap equals size", `{"window":{"size":128,"overlap":128}}`, true},
		{"empty positive label", `{"window":{"size":128},"positive_labels":[""]}`, true},
		{"duplicate positive label", `{"window":{"size":128},"positive_labels":["attack","attack"]}`, true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var cfg PromptGuardConfig
			if err := json.Unmarshal([]byte(tc.config), &cfg); err != nil {
				t.Fatal(err)
			}
			sr := &SemanticRouter{Spec: SemanticRouterSpec{Config: ConfigSpec{PromptGuard: &cfg}}}
			for _, update := range []bool{false, true} {
				var err error
				if update {
					_, err = sr.ValidateUpdate(context.Background(), &SemanticRouter{}, sr)
				} else {
					_, err = sr.ValidateCreate(context.Background(), sr)
				}
				if (err != nil) != tc.wantError {
					t.Fatalf("update=%v error=%v, wantError=%v", update, err, tc.wantError)
				}
			}
		})
	}
}

func TestPromptGuardWindowCELAndPruning(t *testing.T) {
	structural, validator := loadCRDValidator(t)
	cases := []struct {
		config string
		refuse bool
	}{
		{`{window: null}`, false},
		{`{window: {size: 128, overlap: 63}, max_sequence_length: 32768}`, false},
		{`{window: {size: 512}, max_sequence_length: 0}`, false},
		{`{variant: candle}`, false},
		{`{protocol: http_classify, window: {size: 128}}`, true},
		{`{variant: candle, max_sequence_length: 1024}`, true},
		{`{window: {size: 513}}`, true},
		{`{window: {size: 128}, max_sequence_length: 64}`, true},
		{`{window: {size: 128, overlap: 128}}`, true},
	}
	for _, tc := range cases {
		cr := "apiVersion: vllm.ai/v1alpha1\nkind: SemanticRouter\nmetadata: {name: test}\nspec:\n  config:\n    prompt_guard: " + tc.config
		if errs := celErrors(t, structural, validator, cr); (len(errs) > 0) != tc.refuse {
			t.Errorf("%s: errors=%v, refuse=%v", tc.config, errs, tc.refuse)
		}
	}
	var obj map[string]interface{}
	if err := yaml.Unmarshal([]byte(`spec: {config: {prompt_guard: {max_sequence_length: 32768, window: {size: 128, overlap: 63}}}}`), &obj); err != nil {
		t.Fatal(err)
	}
	pruning.Prune(obj, structural, true)
	data, err := json.Marshal(obj)
	if err != nil {
		t.Fatal(err)
	}
	var sr SemanticRouter
	if err := json.Unmarshal(data, &sr); err != nil {
		t.Fatal(err)
	}
	cfg := sr.Spec.Config.PromptGuard
	if cfg.MaxSequenceLength != 32768 || cfg.Window == nil || cfg.Window.Size != 128 || cfg.Window.Overlap != 63 {
		t.Fatalf("API server pruned explicit context settings: %+v", cfg)
	}
	copied := sr.DeepCopy()
	copied.Spec.Config.PromptGuard.Window.Size = 256
	if cfg.Window.Size != 128 {
		t.Fatal("DeepCopy shares a mutable window with its source")
	}
}

func TestGeneratedPromptGuardContextSchemasAgree(t *testing.T) {
	var previous *apiextensionsv1.JSONSchemaProps
	for _, relative := range []string{"config/crd/bases", "bundle/manifests"} {
		data, err := os.ReadFile(filepath.Join("..", "..", relative, "vllm.ai_semanticrouters.yaml"))
		if err != nil {
			t.Fatal(err)
		}
		var crd apiextensionsv1.CustomResourceDefinition
		if err := yaml.Unmarshal(data, &crd); err != nil {
			t.Fatal(err)
		}
		guard := crd.Spec.Versions[0].Schema.OpenAPIV3Schema.Properties["spec"].Properties["config"].Properties["prompt_guard"]
		window := guard.Properties["window"]
		budget := guard.Properties["max_sequence_length"]
		if !window.Nullable || window.Type != "object" || budget.Type != "integer" || budget.Minimum == nil || *budget.Minimum != 0 {
			t.Fatalf("%s lost nullable window or nonnegative budget", relative)
		}
		size := window.Properties["size"]
		overlap := window.Properties["overlap"]
		if size.Minimum == nil || *size.Minimum != 1 || overlap.Minimum == nil || *overlap.Minimum != 0 || strings.Join(window.Required, ",") != "size" {
			t.Fatalf("%s lost window field constraints", relative)
		}
		if previous != nil && !reflect.DeepEqual(*previous, guard) {
			t.Fatal("installed CRD and OLM bundle disagree on PromptGuard")
		}
		previous = &guard
	}
}
