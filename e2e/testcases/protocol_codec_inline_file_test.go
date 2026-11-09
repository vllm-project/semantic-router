package testcases

import (
	"fmt"
	"testing"
)

func TestResponsesInlineFileAssertion(t *testing.T) {
	for _, tc := range []struct {
		name, model, data, filename string
		valid                       bool
	}{
		{"valid", inlineFileProviderModel, inlineFileDataURL, "probe.pdf", true},
		{"bare base64", inlineFileProviderModel, "JVBERi0xLjQK", "probe.pdf", false},
		{"wrong MIME", inlineFileProviderModel, "data:text/plain;base64,JVBERi0xLjQK", "probe.pdf", false},
		{"changed bytes", inlineFileProviderModel, "data:application/pdf;base64,YQ==", "probe.pdf", false},
		{"missing filename", inlineFileProviderModel, inlineFileDataURL, "", false},
		{"alias not resolved", inlineFileAlias, inlineFileDataURL, "probe.pdf", false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			raw := fmt.Sprintf(`{"body":{"model":%q,"input":[{"content":[{"type":"input_file","file_data":%q,"filename":%q}]}]}}`, tc.model, tc.data, tc.filename)
			if err := assertResponsesInlineFile([]byte(raw)); (err == nil) != tc.valid {
				t.Fatalf("valid=%t error=%v", tc.valid, err)
			}
		})
	}
	for _, raw := range []string{`{`, fmt.Sprintf(`{"body":{"model":%q,"input":[]}}`, inlineFileProviderModel)} {
		if err := assertResponsesInlineFile([]byte(raw)); err == nil {
			t.Fatalf("accepted invalid capture: %s", raw)
		}
	}
}
