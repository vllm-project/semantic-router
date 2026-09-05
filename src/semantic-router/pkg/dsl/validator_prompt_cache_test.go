package dsl

import "testing"

func TestPromptCacheTemplateNameUsesTemplateType(t *testing.T) {
	source := `
PLUGIN prompt_cache memory {
  retrieval_limit: 5
}

ROUTE memory_route {
  PRIORITY 1
  MODEL "model"
  PLUGIN prompt_cache {
    retrieval_limit: 10
  }
}
`
	diagnostics, parseErrs := Validate(source)
	if len(parseErrs) != 0 {
		t.Fatalf("parse errors = %v", parseErrs)
	}
	if len(diagnostics) != 0 {
		t.Fatalf("validation diagnostics = %v", diagnostics)
	}
}
