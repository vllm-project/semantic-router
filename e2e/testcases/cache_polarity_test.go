package testcases

import (
	"context"
	"encoding/json"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
)

func cacheRequestTestServer(t *testing.T, handler http.HandlerFunc) string {
	t.Helper()
	server := httptest.NewServer(handler)
	t.Cleanup(server.Close)
	_, port, err := net.SplitHostPort(strings.TrimPrefix(server.URL, "http://"))
	if err != nil {
		t.Fatal(err)
	}
	return port
}

func TestCacheRequestPreservesDefaultAndExplicitModels(t *testing.T) {
	models := make(chan string, 2)
	port := cacheRequestTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		var request struct {
			Model    string `json:"model"`
			Messages []struct {
				Content string `json:"content"`
			} `json:"messages"`
		}
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Error(err)
		}
		models <- request.Model
		if len(request.Messages) != 1 || request.Messages[0].Content != "unchanged query" {
			t.Errorf("request text changed: %+v", request)
		}
		if r.Header.Get("x-vsr-debug") != "true" {
			t.Error("cache score observation must request the debug surface")
		}
		w.WriteHeader(http.StatusOK)
	})
	for _, model := range []string{"MoM", "e2e-cache"} {
		var response *http.Response
		var err error
		if model == "MoM" {
			response, err = sendChatRequest(context.Background(), "unchanged query", port, false)
		} else {
			response, err = sendChatRequestForModel(context.Background(), "unchanged query", port, model, false)
		}
		if err != nil {
			t.Fatal(err)
		}
		response.Body.Close()
		if actual := <-models; actual != model {
			t.Fatalf("sent model=%q, require %q", actual, model)
		}
	}
}

func TestCachePolarityRejectsWrongPolicyBeforePriming(t *testing.T) {
	var calls atomic.Int32
	port := cacheRequestTestServer(t, func(w http.ResponseWriter, _ *http.Request) {
		calls.Add(1)
		w.Header().Set("x-vsr-selected-recipe", "default")
		w.Header().Set("x-vsr-selected-decision", "computer_science_decision")
	})
	result := runCachePolarityCase(context.Background(), cachePolarityCase{OriginalQuestion: "How to turn on dark mode?"}, port, false)
	if len(result.failures) != 1 || !strings.Contains(result.failures[0], "cache policy was not selected") || !strings.Contains(result.failures[0], "computer_science_decision") {
		t.Fatalf("wrong policy must fail explicitly: %+v", result)
	}
	if calls.Load() != 1 || result.rejected || result.served {
		t.Fatalf("wrong policy must stop before interpreting cache scores: calls=%d result=%+v", calls.Load(), result)
	}
}

func TestCachePolarityRequiresObservedPriming(t *testing.T) {
	var calls atomic.Int32
	port := cacheRequestTestServer(t, func(w http.ResponseWriter, _ *http.Request) {
		calls.Add(1)
		w.Header().Set("x-vsr-selected-recipe", "e2e-cache")
		w.Header().Set("x-vsr-selected-decision", "e2e_cache_decision")
	})
	result := runCachePolarityCase(context.Background(), cachePolarityCase{OriginalQuestion: "How to turn on dark mode?"}, port, false)
	if len(result.failures) != 1 || !strings.Contains(result.failures[0], "never served the original question") || calls.Load() != 4 {
		t.Fatalf("a configured policy without observed cache priming cannot qualify polarity: calls=%d result=%+v", calls.Load(), result)
	}
}

func TestCachePolarityRetainsContradictionAndParaphraseAssertions(t *testing.T) {
	cases, err := loadCachePolarityCases("testdata/cache_polarity_cases.json")
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range cases {
		t.Run(tc.OriginalQuestion, func(t *testing.T) {
			var calls atomic.Int32
			port := cacheRequestTestServer(t, func(w http.ResponseWriter, r *http.Request) {
				var request struct {
					Model    string `json:"model"`
					Messages []struct {
						Content string `json:"content"`
					} `json:"messages"`
				}
				if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
					t.Error(err)
					return
				}
				if request.Model != "e2e-cache" || len(request.Messages) != 1 {
					t.Errorf("unexpected polarity request: %+v", request)
					return
				}
				index := calls.Add(1)
				w.Header().Set("x-vsr-selected-recipe", "e2e-cache")
				w.Header().Set("x-vsr-selected-decision", "e2e_cache_decision")
				question := tc.OriginalQuestion
				switch index {
				case 1:
					// The first request populates an empty cache.
				case 2, 3, 4:
					w.Header().Set("x-vsr-cache-hit", "true")
					w.Header().Set("x-vsr-cache-similarity", "1.0")
				case 5:
					question = tc.Contradiction
					w.Header().Set("x-vsr-cache-hit", "false")
					w.Header().Set("x-vsr-cache-similarity", "0.95")
				case 6:
					question = tc.Paraphrase
					w.Header().Set("x-vsr-cache-hit", "true")
					w.Header().Set("x-vsr-cache-similarity", "0.94")
				default:
					t.Errorf("unexpected extra request %d", index)
				}
				if request.Messages[0].Content != question {
					t.Errorf("authored question changed: got %q require %q", request.Messages[0].Content, question)
				}
			})
			result := runCachePolarityCase(context.Background(), tc, port, false)
			if len(result.failures) != 0 || !result.rejected || !result.served || calls.Load() != 6 {
				t.Fatalf("expected both original assertions: calls=%d result=%+v", calls.Load(), result)
			}
		})
	}
}

func TestCachePolarityRejectsMissingScoresAndIncorrectOutcomes(t *testing.T) {
	tc := cachePolarityCase{Contradiction: "opposite", Paraphrase: "same meaning"}
	for _, result := range []CacheResult{
		{},
		{SimilarityReported: true, Similarity: 0.79},
		{SimilarityReported: true, Similarity: 0.95, CacheHit: true},
	} {
		if message := assertPolarityContradictionRejected(tc, result); message == "" {
			t.Errorf("contradiction incorrectly accepted: %+v", result)
		}
	}
	for _, result := range []CacheResult{{}, {SimilarityReported: true, Similarity: 0.95}} {
		if message := assertPolarityParaphraseServed(tc, result); message == "" {
			t.Errorf("paraphrase miss incorrectly accepted: %+v", result)
		}
	}
	for _, message := range []string{
		assertPolarityContradictionRejected(tc, CacheResult{}),
		assertPolarityParaphraseServed(tc, CacheResult{}),
	} {
		if !strings.Contains(message, "no cache candidate score") {
			t.Errorf("absent evidence must not be reported as a model score: %s", message)
		}
	}
}
