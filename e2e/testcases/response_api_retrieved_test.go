package testcases

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
)

func TestValidateRetrievedResponse(t *testing.T) {
	const body = `{"id":"resp_test","object":"response","status":"completed","previous_response_id":"resp_parent","output":[{"type":"message","content":[{"type":"output_text","text":"","annotations":[]}]}]}`
	for _, test := range []struct {
		name  string
		body  string
		valid bool
	}{
		{"complete", body, true},
		{"missing_text", strings.Replace(body, `"text":"",`, "", 1), false},
		{"different_id", strings.Replace(body, "resp_test", "resp_other", 1), false},
		{"different_status", strings.Replace(body, "completed", "in_progress", 1), false},
		{"different_lineage", strings.Replace(body, "resp_parent", "resp_other", 1), false},
	} {
		t.Run(test.name, func(t *testing.T) {
			var created, retrieved fixtures.ResponseAPIResponse
			if err := json.Unmarshal([]byte(body), &created); err != nil {
				t.Fatal(err)
			}
			if err := json.Unmarshal([]byte(test.body), &retrieved); err != nil {
				t.Fatal(err)
			}
			err := validateRetrievedResponse(&created, &retrieved)
			if (err == nil) != test.valid {
				t.Fatalf("validation error=%v, want valid=%t", err, test.valid)
			}
		})
	}
}
