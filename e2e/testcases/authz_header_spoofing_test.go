package testcases

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"testing"
)

func TestAuthzSpoofingRequiresObservedDefaultModel(t *testing.T) {
	const model = "Qwen/Qwen2.5-7B-Instruct"
	for _, test := range []struct {
		name, body, selected string
		status               int
		wantError            bool
	}{
		{"top-level model", `{"model":"` + model + `"}`, "", 200, false},
		{"selected header", `{}`, model, 200, false},
		{"matching evidence", `{"model":"` + model + `"}`, model, 200, false},
		{"missing model", `{}`, "", 200, true},
		{"nested model is not protocol", `{"choices":[{"message":{"model":"` + model + `"}}]}`, "", 200, true},
		{"unauthorized model", `{"model":"Qwen/Qwen2.5-14B-Instruct"}`, "", 200, true},
		{"conflicting evidence", `{"model":"Qwen/Qwen2.5-14B-Instruct"}`, model, 200, true},
		{"invalid body", `not json`, model, 200, true},
		{"HTTP failure", `{"model":"` + model + `"}`, model, 503, true},
	} {
		t.Run(test.name, func(t *testing.T) {
			response := &http.Response{StatusCode: test.status, Header: make(http.Header), Body: io.NopCloser(strings.NewReader(test.body))}
			response.Header.Set("x-vsr-selected-model", test.selected)
			err := validateAuthzResponse(response, &AuthzSpoofingResult{}, AuthzSpoofingTestCase{ExpectedModel: model})
			if (err != nil) != test.wantError {
				t.Fatalf("error = %v, wantError = %v", err, test.wantError)
			}
		})
	}
}

func TestAuthzSpoofingRequiresEveryCase(t *testing.T) {
	passed := AuthzSpoofingResult{Correct: true}
	for _, test := range []struct {
		name      string
		expected  int
		results   []AuthzSpoofingResult
		wantError bool
	}{
		{"all passed", 2, []AuthzSpoofingResult{passed, passed}, false},
		{"one failed", 2, []AuthzSpoofingResult{passed, {}}, true},
		{"empty", 0, nil, true},
		{"missing case", 2, []AuthzSpoofingResult{passed}, true},
		{"request error", 1, []AuthzSpoofingResult{{Correct: true, Error: "read failure"}}, true},
	} {
		t.Run(test.name, func(t *testing.T) {
			if err := validateAuthzSpoofingResults(test.expected, test.results); (err != nil) != test.wantError {
				t.Fatalf("error = %v, wantError = %v", err, test.wantError)
			}
		})
	}
}

func TestAuthzSpoofingPromptDistinguishesPremiumRoute(t *testing.T) {
	req, err := createAuthzSpoofingRequest(context.Background(), AuthzSpoofingTestCase{UserGroups: "premium-tier"}, "8080")
	if err != nil {
		t.Fatal(err)
	}
	var body struct {
		Messages []struct{ Content string } `json:"messages"`
	}
	if err := json.NewDecoder(req.Body).Decode(&body); err != nil {
		t.Fatal(err)
	}
	if len(body.Messages) != 1 || !strings.Contains(strings.ToLower(body.Messages[0].Content), "write code") {
		t.Fatal("the probe must select premium_code when spoofed identity is not stripped")
	}
	if req.Header.Get("x-authz-user-groups") != "premium-tier" {
		t.Fatal("spoofed identity is missing from the probe")
	}
}
