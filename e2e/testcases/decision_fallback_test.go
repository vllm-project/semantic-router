package testcases

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"testing"
)

func TestFallbackCorpusUsesExplicitMatchAndNoMatch(t *testing.T) {
	var data DecisionFallbackTestData
	if err := json.Unmarshal(decisionFallbackCasesJSON, &data); err != nil {
		t.Fatal(err)
	}
	var matched, unmatched int
	for _, tc := range data.TestCases {
		if tc.Query == "" || strings.Contains(tc.Query, fallbackMatchMarker) {
			t.Fatal("original query must be nonempty and independent of the fixture marker")
		}
		request := buildFallbackRequest(tc)
		if request.Model != "e2e-fallback" || len(request.Messages) != 1 || request.Messages[0].Role != "user" {
			t.Fatalf("invalid fallback request: %#v", request)
		}
		want := tc.Query
		if tc.ShouldFallback {
			unmatched++
			if tc.ExpectedDecision != "" {
				t.Fatal("no-match oracle must not name a domain decision")
			}
		} else {
			matched++
			want = "__E2E_FALLBACK_MATCH__\n" + tc.Query
			if tc.ExpectedDecision != "fallback_contract_match" {
				t.Fatal("matched oracle must name the single fixture decision")
			}
		}
		if request.Messages[0].Content != want {
			t.Fatal("fixture trigger changed the original question text")
		}
	}
	if len(data.TestCases) != 51 || matched != 43 || unmatched != 8 {
		t.Fatalf("fallback inventory=%d matched=%d unmatched=%d, want 51/43/8", len(data.TestCases), matched, unmatched)
	}
}

func TestFallbackResponseRequiresTrueNoMatchAndProviderDispatch(t *testing.T) {
	for _, fallback := range []bool{false, true} {
		t.Run(fmt.Sprintf("fallback=%t", fallback), func(t *testing.T) {
			tc := DecisionFallbackCase{Query: "Original question", ShouldFallback: fallback, ExpectedDecision: "fallback_contract_match"}
			model := "base-model"
			if fallback {
				tc.ExpectedDecision = ""
				model = "general-expert"
			}
			header := http.Header{}
			for key, value := range map[string]string{"x-vsr-schema-version": "2", "x-vsr-response-path": "upstream", "x-vsr-selected-recipe": "e2e-fallback", "x-vsr-selected-decision": tc.ExpectedDecision, "x-vsr-selected-model": model} {
				header.Set(key, value)
			}
			body := []byte(fmt.Sprintf(`{"object":"chat.completion","model":%q,"choices":[{"message":{"role":"assistant","content":"Provider response"}}]}`, model))
			if err := validateFallbackResponse(tc, header, body); err != nil {
				t.Fatal(err)
			}
			for key, value := range map[string]string{"x-vsr-schema-version": "", "x-vsr-response-path": "cache", "x-vsr-selected-recipe": "mom", "x-vsr-selected-decision": "other_decision", "x-vsr-selected-model": "wrong-model"} {
				changed := header.Clone()
				changed.Set(key, value)
				if err := validateFallbackResponse(tc, changed, body); err == nil {
					t.Errorf("accepted %s=%q", key, value)
				}
			}
			for _, invalid := range []string{`{`, `{}`, `{"object":"chat.completion","model":"wrong-model","choices":[]}`} {
				if err := validateFallbackResponse(tc, header, []byte(invalid)); err == nil {
					t.Errorf("accepted invalid provider response %s", invalid)
				}
			}
			request := buildFallbackRequest(tc)
			request.Model = model
			observe := func() []byte {
				observed, err := json.Marshal(map[string]any{"body": request})
				if err != nil {
					t.Fatal(err)
				}
				return observed
			}
			if err := validateFallbackProviderRequest(tc, observe()); err != nil {
				t.Fatal(err)
			}
			request.Model = "e2e-fallback"
			if err := validateFallbackProviderRequest(tc, observe()); err == nil {
				t.Fatal("accepted dispatch without resolving the entrypoint to a provider")
			}
			request.Model = model
			request.Messages[0].Content = "Different request"
			if err := validateFallbackProviderRequest(tc, observe()); err == nil {
				t.Fatal("accepted an unrelated provider request")
			}
		})
	}
}
