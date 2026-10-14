//go:build !windows && cgo

package apiserver

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/services"
)

func actionPreviewRouterConfig() *config.RouterConfig {
	cfg := &config.RouterConfig{}
	for _, action := range config.SupportedActions() {
		cfg.ActionRules = append(cfg.ActionRules, config.ActionRule{Name: action})
		cfg.Decisions = append(cfg.Decisions, config.Decision{
			Name:     action + "-route",
			Priority: 10,
			Rules:    config.RuleNode{Type: config.SignalTypeAction, Name: action},
		})
	}
	return cfg
}

func TestPreviewRoutesEachActionToItsDecision(t *testing.T) {
	cfg := actionPreviewRouterConfig()
	registry := routerruntime.NewRegistry(cfg)
	service := newContextEvalServer(t, cfg).classificationSvc.(*services.ClassificationService)
	t.Cleanup(func() {
		if err := service.Close(); err != nil {
			t.Error(err)
		}
	})
	registry.SetClassificationService(service)
	server := httptest.NewServer((&ClassificationAPIServer{config: cfg, runtimeRegistry: registry}).setupRoutes())
	t.Cleanup(server.Close)

	user := func(text string) []map[string]string {
		return []map[string]string{{"role": "user", "content": text}}
	}
	tests := []struct {
		action   string
		messages []map[string]string
	}{
		{action: config.ActionGenerate, messages: user("Write a function that merges two sorted lists.")},
		{action: config.ActionExplain, messages: user("What does this regex match?")},
		{action: config.ActionFix, messages: user("Fix the null pointer exception in the handler.")},
		{action: config.ActionRefactor, messages: user("Rename getUserData to fetchUser everywhere.")},
		{action: config.ActionTest, messages: user("Add a test for the empty-input case.")},
		{action: config.ActionOther, messages: []map[string]string{
			{"role": "user", "content": "Fix the failing test in parser_test.go."},
			{"role": "assistant", "content": "Done, the test passes now."},
			{"role": "user", "content": "thanks!"},
		}},
	}
	for _, tt := range tests {
		t.Run(tt.action, func(t *testing.T) {
			body, err := json.Marshal(map[string]interface{}{"messages": tt.messages})
			if err != nil {
				t.Fatal(err)
			}
			response, err := server.Client().Post(server.URL+apiRoutingPreviewPath, "application/json", strings.NewReader(string(body)))
			if err != nil {
				t.Fatal(err)
			}
			defer response.Body.Close()
			var result services.EvalResponse
			if err := json.NewDecoder(response.Body).Decode(&result); err != nil {
				t.Fatal(err)
			}
			if response.StatusCode != http.StatusOK || result.DecisionResult == nil ||
				result.DecisionResult.DecisionName != tt.action+"-route" {
				t.Fatalf("preview = %d %+v, want decision %s-route", response.StatusCode, result.DecisionResult, tt.action)
			}
			if matched := result.DecisionResult.MatchedSignals.Action; len(matched) != 1 || matched[0] != tt.action {
				t.Fatalf("matched actions = %v, want only %q", matched, tt.action)
			}
			if score := result.SignalValues["action:"+tt.action]; score != 1 {
				t.Fatalf("signal values = %v, want action:%s at 1", result.SignalValues, tt.action)
			}
		})
	}
}
