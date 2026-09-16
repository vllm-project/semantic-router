package testcases

import (
	"bytes"
	"strings"
	"testing"
)

const shadowPrimaryTestBody = `{
	"object":"chat.completion","model":"openai/gpt-oss-20b","created":1700000000,
	"choices":[{"index":0,"message":{"role":"assistant","content":"Hello from openai/gpt-oss-20b."},"finish_reason":"stop"}],
	"usage":{"prompt_tokens":4,"completion_tokens":7,"total_tokens":11}
}`

func TestNormalizeShadowPrimaryRejectsChangedContract(t *testing.T) {
	for _, tc := range []struct {
		name, old, replacement string
	}{
		{"wrong model", `"model":"openai/gpt-oss-20b"`, `"model":"openai/shadow-candidate"`},
		{"wrong content", "Hello from openai/gpt-oss-20b.", "Hello from the shadow."},
		{"empty content", "Hello from openai/gpt-oss-20b.", ""},
		{"wrong role", `"role":"assistant"`, `"role":"user"`},
		{"wrong envelope", "chat.completion", "response"},
		{"incomplete response", `"finish_reason":"stop"`, `"finish_reason":"length"`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			body := strings.Replace(shadowPrimaryTestBody, tc.old, tc.replacement, 1)
			if _, err := normalizeShadowPrimary([]byte(body)); err == nil {
				t.Fatalf("accepted a changed primary contract: %s", body)
			}
		})
	}
}

func TestNormalizeShadowPrimaryOnlyIgnoresCreationTime(t *testing.T) {
	baseline, err := normalizeShadowPrimary([]byte(shadowPrimaryTestBody))
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name, old, replacement string
		wantEqual              bool
	}{
		{"created", "1700000000", "1700000010", true},
		{"usage", `"prompt_tokens":4`, `"prompt_tokens":5`, false},
		{"additional field", `"usage":`, `"unexpected":true,"usage":`, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			normalized, normalizeErr := normalizeShadowPrimary([]byte(strings.Replace(shadowPrimaryTestBody, tc.old, tc.replacement, 1)))
			if normalizeErr != nil {
				t.Fatal(normalizeErr)
			}
			if bytes.Equal(normalized, baseline) != tc.wantEqual {
				t.Fatalf("baseline comparison equal=%t, want %t: %s", bytes.Equal(normalized, baseline), tc.wantEqual, normalized)
			}
		})
	}
}
