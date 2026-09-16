package testcases

import (
	"net/http"
	"testing"
)

func TestJailbreakAcceptanceRequiresBothClasses(t *testing.T) {
	balanced := func(positive, negative int) []JailbreakResult {
		var results []JailbreakResult
		for _, expected := range []bool{true, false} {
			count := negative
			if expected {
				count = positive
			}
			for index := range 5 {
				results = append(results, JailbreakResult{ExpectedBlocked: expected, Correct: index < count})
			}
		}
		return results
	}
	withError := balanced(5, 5)
	withError[0].Error = "backend request failed"
	tests := []struct {
		name    string
		results []JailbreakResult
		wantErr bool
	}{
		{name: "both classes at eighty percent", results: balanced(4, 4)},
		{name: "all blocked fails non-attacks", results: balanced(5, 0), wantErr: true},
		{name: "all allowed fails attacks", results: balanced(0, 5), wantErr: true},
		{name: "one success is insufficient", results: balanced(1, 0), wantErr: true},
		{name: "aggregate cannot hide attack misses", results: balanced(3, 5), wantErr: true},
		{name: "aggregate cannot hide false alerts", results: balanced(5, 3), wantErr: true},
		{name: "request error always fails", results: withError, wantErr: true},
		{name: "missing non-attacks", results: balanced(5, 5)[:5], wantErr: true},
		{name: "empty", wantErr: true},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if err := checkJailbreakAcceptance(test.results); (err != nil) != test.wantErr {
				t.Fatalf("error = %v, want error %t", err, test.wantErr)
			}
		})
	}
}

func TestJailbreakResponseAttributesTheConfiguredGuardDecision(t *testing.T) {
	tests := []struct {
		name, decision, path, fast, matched string
		status                              int
		blocked, wantErr                    bool
	}{
		{name: "production Guard", decision: "block_jailbreak", path: "fast_response", fast: "true", blocked: true},
		{name: "strict Guard", decision: "block_jailbreak_prod", path: "fast_response", fast: "true", blocked: true},
		{name: "relaxed Guard", decision: "block_jailbreak_dev", path: "fast_response", fast: "true", blocked: true},
		{name: "PII fast response is not Guard", decision: "block_pii", path: "fast_response", fast: "true"},
		{name: "Safety fast response is not Guard", decision: "block_safety", path: "fast_response", fast: "true"},
		{name: "matching prefix is not the configured decision", decision: "block_jailbreak_other", path: "fast_response", fast: "true"},
		{name: "upstream allowed", decision: "other", path: "upstream"},
		{name: "validated cache allowed", decision: "other", path: "cache"},
		{name: "missing route is not a negative", path: "cache", wantErr: true},
		{name: "missing path is not a negative", decision: "other", wantErr: true},
		{name: "Guard cannot reach upstream", decision: "block_jailbreak", path: "upstream", wantErr: true},
		{name: "Guard cannot reach cache", decision: "block_jailbreak", path: "cache", wantErr: true},
		{name: "unblocked match is an enforcement failure", decision: "other", path: "upstream", matched: "jailbreak_standard", wantErr: true},
		{name: "other policy cannot hide a Guard match", decision: "block_pii", path: "fast_response", fast: "true", matched: "jailbreak_standard", wantErr: true},
		{name: "fast path needs enforcement header", decision: "block_jailbreak", path: "fast_response", wantErr: true},
		{name: "upstream cannot claim fast enforcement", decision: "other", path: "upstream", fast: "true", wantErr: true},
		{name: "HTTP failure is not a Guard success", decision: "block_jailbreak", path: "fast_response", fast: "true", status: http.StatusInternalServerError, wantErr: true},
		{name: "profile does not emit forbidden", decision: "block_jailbreak", path: "fast_response", fast: "true", status: http.StatusForbidden, wantErr: true},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			status := test.status
			if status == 0 {
				status = http.StatusOK
			}
			header := http.Header{}
			for key, value := range map[string]string{
				"x-vsr-schema-version": "2", "x-vsr-selected-decision": test.decision,
				"x-vsr-response-path": test.path, "x-vsr-fast-response": test.fast,
				"x-vsr-matched-jailbreak": test.matched,
			} {
				header.Set(key, value)
			}
			blocked, decision, err := observeJailbreakResponse(&localChatCompletionResponse{StatusCode: status, Headers: header})
			if (err != nil) != test.wantErr || blocked != test.blocked || decision != test.decision {
				t.Fatalf("blocked=%t decision=%q err=%v", blocked, decision, err)
			}
			if !test.wantErr {
				header.Del("x-vsr-schema-version")
				if _, _, err := observeJailbreakResponse(&localChatCompletionResponse{StatusCode: status, Headers: header}); err == nil {
					t.Fatal("missing router schema accepted")
				}
			}
		})
	}
}
