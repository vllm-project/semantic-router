package testcases

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestContextCompressionProviderContract(t *testing.T) {
	original := contextCompressionToolOutput()
	compressed := strings.Join([]string{
		"diagnostic source header",
		contextCompressionOmission,
		contextCompressionRelevant,
		contextCompressionOmission,
		"diagnostic source footer",
	}, "\n")
	for _, testCase := range []struct {
		name             string
		toolOutput       string
		expectCompressed bool
		wantError        bool
	}{
		{name: "compressed", toolOutput: compressed, expectCompressed: true},
		{name: "unchanged compression", toolOutput: original, expectCompressed: true, wantError: true},
		{name: "bypass", toolOutput: original},
		{name: "changed bypass", toolOutput: compressed, wantError: true},
		{name: "lost relevant evidence", toolOutput: contextCompressionOmission, expectCompressed: true, wantError: true},
	} {
		t.Run(testCase.name, func(t *testing.T) {
			observed := contextCompressionObservedRequest(t, testCase.toolOutput)
			err := validateContextCompressionProviderRequest(observed, original, testCase.expectCompressed)
			if (err != nil) != testCase.wantError {
				t.Fatalf("validation error = %v, wantError = %t", err, testCase.wantError)
			}
		})
	}
}

func contextCompressionObservedRequest(t *testing.T, toolOutput string) []byte {
	t.Helper()
	request := contextCompressionChatRequest(toolOutput)
	body, err := json.Marshal(map[string]any{"body": request})
	if err != nil {
		t.Fatal(err)
	}
	return body
}
