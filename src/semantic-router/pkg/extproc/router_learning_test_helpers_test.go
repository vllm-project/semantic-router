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

package extproc

import (
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func extprocIntPtr(v int) *int { return &v }

func extprocFloat64Ptr(v float64) *float64 { return &v }

func extprocBoolPtr(v bool) *bool { return &v }

func stringPtr(v string) *string { return &v }

func routerLearningTestConfig(scope string) *config.RouterConfig {
	return &config.RouterConfig{
		BackendModels: config.BackendModels{
			DefaultModel: "cheap",
			ModelConfig: map[string]config.ModelParams{
				"cheap":    {},
				"frontier": {},
			},
		},
		RouterLearning: config.RouterLearningConfig{
			Enabled: true,
			Protection: config.RouterLearningProtectionConfig{
				Enabled: extprocBoolPtr(true),
				Scope:   scope,
				Identity: config.RouterLearningIdentityConfig{
					Headers: config.RouterLearningIdentityHeadersConfig{
						Session:      stringPtr("x-session-id"),
						Conversation: stringPtr("x-conversation-id"),
					},
				},
			},
		},
	}
}

func routerLearningAdaptationTestConfig() *config.RouterConfig {
	disabled := false
	return &config.RouterConfig{
		BackendModels: config.BackendModels{
			DefaultModel: "cheap",
			ModelConfig: map[string]config.ModelParams{
				"cheap":    {},
				"frontier": {},
			},
		},
		RouterLearning: config.RouterLearningConfig{
			Enabled: true,
			Adaptation: config.RouterLearningAdaptationConfig{
				CandidateSet: config.RouterLearningCandidateSetDecision,
			},
			Protection: config.RouterLearningProtectionConfig{
				Enabled: &disabled,
			},
		},
	}
}

func routerLearningProtectionOnlyTestConfig(scope string) *config.RouterConfig {
	disabled := false
	cfg := routerLearningTestConfig(scope)
	cfg.RouterLearning.Adaptation.Enabled = &disabled
	return cfg
}

func routerLearningRequestContext(sessionID string, conversationID string) *RequestContext {
	return &RequestContext{
		Headers: map[string]string{
			"x-session-id":      sessionID,
			"x-conversation-id": conversationID,
		},
		SessionID: sessionID,
	}
}

func learningIdentityPart(t *testing.T, policy *routerLearningPolicy, name string) map[string]interface{} {
	t.Helper()
	policyMap := policy.ToMap()
	identity, ok := policyMap["identity"].(map[string]interface{})
	if !ok {
		t.Fatalf("expected identity diagnostics in learning policy, got %#v", policy)
	}
	part, ok := identity[name].(map[string]interface{})
	if !ok {
		t.Fatalf("expected %s identity diagnostics, got %#v", name, identity)
	}
	return part
}

func assertModelRefs(t *testing.T, refs []config.ModelRef, want []string) {
	t.Helper()
	got := make([]string, 0, len(refs))
	for _, ref := range refs {
		got = append(got, ref.Model)
	}
	if strings.Join(got, ",") != strings.Join(want, ",") {
		t.Fatalf("expected model refs %v, got %v", want, got)
	}
}
