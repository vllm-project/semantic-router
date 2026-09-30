package testcases

import (
	"context"
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

func TestDomainClassificationUsesIsolatedUpstreamPrediction(t *testing.T) {
	for _, surface := range []struct {
		profile, model, recipe, path string
		valid                        bool
	}{
		{"envoy-ai-gateway", "e2e-domain", "e2e-domain", "upstream", true},
		{"envoy-ai-gateway", "e2e-domain", "e2e-domain", "cache", false},
		{"envoy-ai-gateway", "e2e-domain", "mom", "upstream", false},
		{"envoy-ai-gateway", "e2e-domain", "e2e-domain", "fast_response", false},
		{"ml-model-selection", "MoM", "", "upstream", true},
		{"production-stack", "MoM", "", "upstream", true},
	} {
		t.Run(surface.profile+"/"+surface.recipe+"/"+surface.path, func(t *testing.T) {
			fixture, fixtureErr := domainFixtureForProfile(surface.profile)
			if fixtureErr != nil {
				t.Fatal(fixtureErr)
			}
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				var request fixtures.ChatCompletionsRequest
				if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
					t.Error(err)
				}
				if request.Model != surface.model || len(request.Messages) != 1 || request.Messages[0].Content != "Original biology question" || request.Messages[0].Role != "user" {
					t.Errorf("changed original classifier request: %#v", request)
				}
				if r.Header.Get("x-vsr-debug") != "true" {
					t.Error("Domain signal observation requires debug headers")
				}
				w.Header().Set("x-vsr-selected-recipe", surface.recipe)
				w.Header().Set("x-vsr-response-path", surface.path)
				w.Header().Set("x-vsr-selected-category", "biology")
				fmt.Fprint(w, `{}`)
			}))
			defer server.Close()
			_, port, err := net.SplitHostPort(strings.TrimPrefix(server.URL, "http://"))
			if err != nil {
				t.Fatal(err)
			}
			result := testSingleClassification(context.Background(), "Original biology question", "biology", port, fixture, false)
			if result.Correct != surface.valid {
				t.Fatalf("correct=%t, want %t: %s", result.Correct, surface.valid, result.Error)
			}
		})
	}
}

func TestDomainClassificationRejectsUndeclaredProfilesBeforeConnecting(t *testing.T) {
	for _, profile := range []string{"", "unregistered-profile"} {
		if err := testDomainClassify(context.Background(), nil, pkgtestcases.TestCaseOptions{Profile: profile}); err == nil ||
			!strings.Contains(err.Error(), "no fixture contract") {
			t.Fatalf("profile %q must fail before service setup: %v", profile, err)
		}
	}
}
