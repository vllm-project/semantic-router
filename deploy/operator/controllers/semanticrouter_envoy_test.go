/*
Copyright 2026 vLLM Semantic Router Contributors.

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

package controllers

import (
	"testing"

	"gopkg.in/yaml.v3"
)

const externalProcessorType = "type.googleapis.com/envoy.extensions.filters.http.ext_proc.v3.ExternalProcessor"

func TestGenerateEnvoyConfigFailsClosedWhenExtProcIsUnavailable(t *testing.T) {
	r := &SemanticRouterReconciler{}

	var config map[string]any
	if err := yaml.Unmarshal([]byte(r.generateEnvoyConfig()), &config); err != nil {
		t.Fatalf("failed to parse generated Envoy config: %v", err)
	}

	extProcConfigs := findExternalProcessorConfigs(config)
	if len(extProcConfigs) != 1 {
		t.Fatalf("expected one ExtProc filter, got %d", len(extProcConfigs))
	}

	failOpen, present := extProcConfigs[0]["failure_mode_allow"]
	if !present {
		t.Fatal("generated ExtProc filter must explicitly set failure_mode_allow")
	}
	if failOpen != false {
		t.Fatalf("generated ExtProc filter must fail closed, got failure_mode_allow=%#v", failOpen)
	}
}

func findExternalProcessorConfigs(value any) []map[string]any {
	var matches []map[string]any
	var visit func(any)
	visit = func(current any) {
		switch typed := current.(type) {
		case map[string]any:
			if typed["@type"] == externalProcessorType {
				matches = append(matches, typed)
			}
			for _, child := range typed {
				visit(child)
			}
		case []any:
			for _, child := range typed {
				visit(child)
			}
		}
	}
	visit(value)
	return matches
}
