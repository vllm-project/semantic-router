package testcases

import (
	"context"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestSummarizeHallucinationResults(t *testing.T) {
	tests := []struct {
		name       string
		results    []HallucinationResult
		wantWarned int
		wantFailed int
		wantInfra  int
	}{
		{
			name: "all warned",
			results: []HallucinationResult{
				{HallucinationWarned: true},
				{HallucinationWarned: true},
			},
			wantWarned: 2, wantFailed: 0, wantInfra: 0,
		},
		{
			name: "issue reproduction: 1 warned, 1 real miss, 2 infra",
			results: []HallucinationResult{
				{Description: "warned", HallucinationWarned: true},
				{Description: "real miss", HallucinationWarned: false, Error: ""},
				{Description: "connection refused", HallucinationWarned: false, Error: "connection refused"},
				{Description: "status 500", HallucinationWarned: false, Error: "status 500"},
			},
			wantWarned: 1, wantFailed: 1, wantInfra: 2,
		},
		{
			name:       "empty results",
			results:    nil,
			wantWarned: 0, wantFailed: 0, wantInfra: 0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			warned, failed, infra := summarizeHallucinationResults(tt.results)
			if warned != tt.wantWarned || failed != tt.wantFailed || infra != tt.wantInfra {
				t.Fatalf("got warned=%d failed=%d infra=%d, want %d/%d/%d",
					warned, failed, infra, tt.wantWarned, tt.wantFailed, tt.wantInfra)
			}
		})
	}
}

func TestHallucinationDetectionRateExcludesInfraErrors(t *testing.T) {
	warned, failed, infra := summarizeHallucinationResults([]HallucinationResult{
		{HallucinationWarned: true},
		{HallucinationWarned: false},
		{Error: "connection refused"},
		{Error: "status 500"},
	})

	if got := detectionRate(warned, warned+failed); got != 50 {
		t.Fatalf("warning rate = %.2f, want 50.00", got)
	}

	if warned != 1 || failed != 1 || infra != 2 {
		t.Fatalf("unexpected counts: warned=%d failed=%d infra=%d",
			warned, failed, infra)
	}

	if got := detectionRate(0, 0); got != 0 {
		t.Fatalf("zero-evaluation warning rate = %.2f, want 0", got)
	}
}

func TestSingleHallucinationDetectionEmptyHTTPBodyIsInfraError(t *testing.T) {
	tests := []struct {
		name       string
		body       string
		wantInfra  int
		wantFailed int
	}{
		{
			name:      "empty body",
			body:      "",
			wantInfra: 1,
		},
		{
			name:      "whitespace-only body",
			body:      " \t\n ",
			wantInfra: 1,
		},
		{
			name:       "non-empty body without warning is detector miss",
			body:       `{}`,
			wantFailed: 1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.Method != http.MethodPost {
					t.Errorf("request method = %q, want POST", r.Method)
				}
				if r.URL.Path != localChatCompletionsPath {
					t.Errorf("request path = %q, want %q", r.URL.Path, localChatCompletionsPath)
				}

				w.WriteHeader(http.StatusOK)
				_, _ = w.Write([]byte(tt.body))
			}))
			defer server.Close()

			_, port, err := net.SplitHostPort(strings.TrimPrefix(server.URL, "http://"))
			if err != nil {
				t.Fatalf("failed to extract test server port: %v", err)
			}

			result := testSingleHallucinationDetection(
				context.Background(),
				HallucinationTestCase{
					Description: "HTTP response body classification",
					Question:    "What is the answer?",
					Context:     "The answer is 42.",
				},
				port,
				false,
			)

			if result.HallucinationWarned {
				t.Fatal("unexpected hallucination warning")
			}

			if tt.wantInfra == 1 && result.Error == "" {
				t.Fatal("expected an infrastructure error, got none")
			}
			if tt.wantFailed == 1 && result.Error != "" {
				t.Fatalf("expected a detector miss without infrastructure error, got: %s", result.Error)
			}

			warned, failed, infra := summarizeHallucinationResults([]HallucinationResult{result})
			if warned != 0 || failed != tt.wantFailed || infra != tt.wantInfra {
				t.Fatalf("summary = warned:%d failed:%d infra:%d, want warned:0 failed:%d infra:%d",
					warned, failed, infra, tt.wantFailed, tt.wantInfra)
			}
		})
	}
}
