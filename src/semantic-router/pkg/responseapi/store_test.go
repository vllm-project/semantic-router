package responseapi

import (
	"encoding/json"
	"testing"
)

func TestStoredResponseIncompleteDetailsJSONRoundTrip(t *testing.T) {
	for _, tc := range []struct {
		name, body, reason string
	}{
		{"output limit", `{"id":"resp_1","status":"incomplete","incomplete_details":{"reason":"max_output_tokens"}}`, "max_output_tokens"},
		{"content filter", `{"id":"resp_1","status":"incomplete","incomplete_details":{"reason":"content_filter"}}`, "content_filter"},
		{"completed", `{"id":"resp_1","status":"completed","incomplete_details":null}`, ""},
		{"legacy incomplete", `{"id":"resp_1","status":"incomplete"}`, ""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var stored StoredResponse
			if err := json.Unmarshal([]byte(tc.body), &stored); err != nil {
				t.Fatal(err)
			}
			encoded, err := json.Marshal(stored)
			if err != nil {
				t.Fatal(err)
			}
			var response ResponseAPIResponse
			if err = json.Unmarshal(encoded, &response); err != nil {
				t.Fatal(err)
			}
			if tc.reason == "" {
				if response.IncompleteDetails != nil {
					t.Fatalf("invented incomplete details: %s", encoded)
				}
			} else if response.IncompleteDetails == nil || response.IncompleteDetails.Reason != tc.reason {
				t.Fatalf("stored JSON lost reason %q: %s", tc.reason, encoded)
			}
		})
	}
}
