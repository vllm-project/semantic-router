package config

import (
	"reflect"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
)

func TestMLModelSelectionConfigDropsFeatureWeights(t *testing.T) {
	yamlContent := []byte(`
type: kmeans
num_clusters: 8
efficiency_weight: 0.2
feature_weights:
  quality: 0.9
`)
	var cfg MLModelSelectionConfig
	if err := yaml.Unmarshal(yamlContent, &cfg); err != nil {
		t.Fatalf("yaml.Unmarshal failed: %v", err)
	}
	if cfg.Type != "kmeans" || cfg.NumClusters != 8 || cfg.EfficiencyWeight == nil || *cfg.EfficiencyWeight != 0.2 {
		t.Fatalf("unexpected parsed config: %+v", cfg)
	}
	typ := reflect.TypeOf(cfg)
	for i := 0; i < typ.NumField(); i++ {
		if strings.HasPrefix(typ.Field(i).Tag.Get("yaml"), "feature_weights") {
			t.Fatalf("feature_weights must not be a config field")
		}
	}
}
