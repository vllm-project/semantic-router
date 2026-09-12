package config

import "testing"

func TestModelBindingProjectionKeepsCanonicalSourceImmutable(t *testing.T) {
	cfg := &RouterConfig{}
	cfg.CategoryModel.ModelID = "models/default"
	cfg.CategoryModel.Variant = "modernbert"
	cfg.ModelDeployments = map[string]ModelDeployment{"private": {Provider: "candle", Artifact: "/mounted/private", Revision: "pin"}}
	cfg.ModelBindings = map[string]ModelBinding{"domain_classifier": {Deployment: "private", Contract: "label_distribution.v1", Adapter: "mmbert32k", MappingPath: "/mounted/maps/domain.json"}}
	cfg.RoutingScope = "private"
	plan, err := CompileModelBindings(cfg)
	if err != nil {
		t.Fatal(err)
	}
	projected, err := ProjectRecipeModelBindings(cfg, plan, "private")
	if err != nil {
		t.Fatal(err)
	}
	if projected.CategoryModel.ModelID != "/mounted/private" || projected.CategoryMappingPath != "/mounted/maps/domain.json" || projected.CategoryModel.Variant != "" {
		t.Fatalf("incorrect projection: %#v", projected.CategoryModel)
	}
	if cfg.CategoryModel.ModelID != "models/default" || cfg.CategoryModel.Variant != "modernbert" || cfg.CategoryMappingPath != "" {
		t.Fatal("source mutated")
	}
	if _, err := ProjectRecipeModelBindings(cfg, plan, "other"); err == nil {
		t.Fatal("foreign recipe binding resolved")
	}
}
