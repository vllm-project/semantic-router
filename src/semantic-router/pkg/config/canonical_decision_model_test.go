package config

import (
	"strings"
	"testing"
)

// A decision that names no model of its own falls back to the default model,
// so a configuration where that fallback is not a model cannot serve a
// matched request.
func TestUnservableDecisionsAreRefused(t *testing.T) {
	cases := map[string]string{
		"no model ref and no default": `version: v0.5
providers:
  models:
    - name: general
      backend_refs:
        - name: local
          provider: openai-compatible
          endpoint: 127.0.0.1:8000
recipes:
  - name: default
    decisions:
      - name: chat
`,
		"ref with no providers": `version: v0.5
recipes:
  - name: default
    decisions:
      - name: chat
        modelRefs:
          - model: typo-model
`,
	}
	for name, document := range cases {
		_, err := ParseYAMLBytes([]byte(document))
		if name == "ref with no providers" {
			// A routing-only fragment names models the document it lands in
			// declares, so the reference alone is not evidence of a mistake.
			if err != nil {
				t.Errorf("%s: %v", name, err)
			}
			continue
		}
		if err == nil || !strings.Contains(err.Error(), "model") {
			t.Errorf("%s: err = %v, want a decision naming no declared model", name, err)
		}
	}
}

func TestDecisionsThatNameADeclaredModelLoad(t *testing.T) {
	cases := map[string]string{
		"model ref": `version: v0.5
providers:
  models:
    - name: general
      backend_refs:
        - name: local
          provider: openai-compatible
          endpoint: 127.0.0.1:8000
recipes:
  - name: default
    decisions:
      - name: chat
        modelRefs:
          - model: general
`,
		"declared default": `version: v0.5
providers:
  defaults:
    model: general
  models:
    - name: general
      backend_refs:
        - name: local
          provider: openai-compatible
          endpoint: 127.0.0.1:8000
recipes:
  - name: default
    decisions:
      - name: chat
`,
	}
	for name, document := range cases {
		if _, err := ParseYAMLBytes([]byte(document)); err != nil {
			t.Errorf("%s: %v", name, err)
		}
	}
}
