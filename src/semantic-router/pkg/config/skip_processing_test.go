/*
Copyright 2025 vLLM Semantic Router.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package config

import (
	"testing"

	"gopkg.in/yaml.v2"
)

func TestSkipProcessingConfigDefaultsAreDisabled(t *testing.T) {
	var cfg SkipProcessingConfig
	if cfg.IsEnabled() {
		t.Fatal("zero-value SkipProcessingConfig should report disabled by default")
	}

	router := &RouterConfig{}
	if router.SkipProcessing.IsEnabled() {
		t.Fatal("zero-value RouterConfig.SkipProcessing should report disabled by default")
	}
}

func TestSkipProcessingConfigIsEnabledRespectsExplicitFlag(t *testing.T) {
	enabled := SkipProcessingConfig{Enabled: true}
	if !enabled.IsEnabled() {
		t.Fatal("explicit enabled flag should report enabled")
	}

	disabled := SkipProcessingConfig{Enabled: false}
	if disabled.IsEnabled() {
		t.Fatal("explicit disabled flag should report disabled")
	}
}

func TestCanonicalRouterGlobalSkipProcessingRoundTrips(t *testing.T) {
	const yamlInput = `
router:
  skip_processing:
    enabled: true
services: {}
stores: {}
integrations: {}
model_catalog:
  embeddings: {}
  system: {}
  modules: {}
`
	var global CanonicalGlobal
	if err := yaml.Unmarshal([]byte(yamlInput), &global); err != nil {
		t.Fatalf("failed to unmarshal canonical global: %v", err)
	}
	if !global.Router.SkipProcessing.IsEnabled() {
		t.Fatal("expected canonical global to surface skip_processing.enabled=true")
	}

	cfg := &RouterConfig{}
	if err := applyCanonicalGlobal(cfg, &global); err != nil {
		t.Fatalf("applyCanonicalGlobal failed: %v", err)
	}
	if !cfg.SkipProcessing.IsEnabled() {
		t.Fatal("applyCanonicalGlobal should propagate skip_processing.enabled to runtime config")
	}

	exported := CanonicalGlobalFromRouterConfig(cfg)
	if exported == nil {
		t.Fatal("expected non-nil canonical global on export")
	}
	if !exported.Router.SkipProcessing.IsEnabled() {
		t.Fatal("CanonicalGlobalFromRouterConfig should round-trip skip_processing.enabled=true")
	}
}
