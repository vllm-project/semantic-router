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

func TestHandoffDefaultsDisabled(t *testing.T) {
	cfg := DefaultGlobalConfig()
	if cfg.Handoff.IsEnabled() {
		t.Fatal("handoff must be disabled by default")
	}
}

func TestCanonicalHandoffConfigRoundTrip(t *testing.T) {
	yamlInput := `
router:
  handoff:
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

	cfg := &RouterConfig{}
	if err := applyCanonicalGlobal(cfg, &global); err != nil {
		t.Fatalf("applyCanonicalGlobal failed: %v", err)
	}
	if !cfg.Handoff.IsEnabled() {
		t.Fatal("expected global.router.handoff.enabled=true in runtime config")
	}

	exported := CanonicalGlobalFromRouterConfig(cfg)
	if exported == nil || !exported.Router.Handoff.IsEnabled() {
		t.Fatal("expected handoff config to survive canonical export")
	}
}
