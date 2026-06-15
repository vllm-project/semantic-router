package dsl

import (
	"strings"
	"testing"
)

func TestParseUnquotedSignalRef(t *testing.T) {
	input := `SIGNAL domain math { description: "math" }

ROUTE test {
  PRIORITY 1
  WHEN domain(math)
  MODEL "test:1b"
}`
	prog, errs := Parse(input)
	if len(errs) > 0 {
		t.Fatalf("expected no errors for unquoted signal ref, got: %v", errs)
	}
	if len(prog.Routes) != 1 {
		t.Fatalf("expected 1 route, got %d", len(prog.Routes))
	}
	route := prog.Routes[0]
	ref, ok := route.When.(*SignalRefExpr)
	if !ok {
		t.Fatalf("expected SignalRefExpr, got %T", route.When)
	}
	if ref.SignalName != "math" {
		t.Errorf("expected signal name 'math', got %q", ref.SignalName)
	}
}

func TestParseUnquotedModelName(t *testing.T) {
	input := `ROUTE test {
  PRIORITY 1
  MODEL simple_model
}`
	prog, errs := Parse(input)
	if len(errs) > 0 {
		t.Fatalf("expected no errors for unquoted model name, got: %v", errs)
	}
	if len(prog.Routes) != 1 {
		t.Fatalf("expected 1 route, got %d", len(prog.Routes))
	}
	if len(prog.Routes[0].Models) != 1 {
		t.Fatalf("expected 1 model, got %d", len(prog.Routes[0].Models))
	}
	if prog.Routes[0].Models[0].Model != "simple_model" {
		t.Errorf("expected model 'simple_model', got %q", prog.Routes[0].Models[0].Model)
	}
}

func TestParseDescriptionInRouteBody(t *testing.T) {
	input := `ROUTE test {
  PRIORITY 1
  DESCRIPTION "A test route for math queries"
  MODEL "qwen:3b"
}`
	prog, errs := Parse(input)
	if len(errs) > 0 {
		t.Fatalf("expected no errors for DESCRIPTION in route body, got: %v", errs)
	}
	if len(prog.Routes) != 1 {
		t.Fatalf("expected 1 route, got %d", len(prog.Routes))
	}
	if prog.Routes[0].Description != "A test route for math queries" {
		t.Errorf("expected description 'A test route for math queries', got %q", prog.Routes[0].Description)
	}
}

func TestParseTrailingCommaInArray(t *testing.T) {
	input := `SIGNAL keyword test {
  keywords: ["foo", "bar", "baz",]
}`
	_, errs := Parse(input)
	if len(errs) > 0 {
		t.Fatalf("expected no errors for trailing comma in array, got: %v", errs)
	}
}

func TestParseTrailingCommaInModelList(t *testing.T) {
	input := `ROUTE test {
  PRIORITY 1
  MODEL "model_a", "model_b",
}`
	_, errs := Parse(input)
	if len(errs) > 0 {
		t.Fatalf("expected no errors for trailing comma in model list, got: %v", errs)
	}
}

func TestParseTrailingCommaInModelOptions(t *testing.T) {
	input := `ROUTE test {
  PRIORITY 1
  MODEL "qwen" (reasoning = true, effort = "high",)
}`
	_, errs := Parse(input)
	if len(errs) > 0 {
		t.Fatalf("expected no errors for trailing comma in model options, got: %v", errs)
	}
}

func TestParseMixedQuotingSignalRefs(t *testing.T) {
	input := `SIGNAL domain math { description: "math" }
SIGNAL domain coding { description: "coding" }

ROUTE test {
  PRIORITY 1
  WHEN domain("math") OR domain(coding)
  MODEL "qwen:7b"
}`
	prog, errs := Parse(input)
	if len(errs) > 0 {
		t.Fatalf("expected no errors for mixed quoting, got: %v", errs)
	}
	if len(prog.Routes) != 1 {
		t.Fatalf("expected 1 route, got %d", len(prog.Routes))
	}
}

func TestPluginNameNormalization(t *testing.T) {
	input := `ROUTE test {
  PRIORITY 1
  MODEL "qwen:7b"
  PLUGIN semantic-cache {
    enabled: true
  }
}`
	prog, errs := Parse(input)
	if len(errs) > 0 {
		t.Fatalf("expected no errors, got: %v", errs)
	}
	if len(prog.Routes[0].Plugins) != 1 {
		t.Fatalf("expected 1 plugin, got %d", len(prog.Routes[0].Plugins))
	}
	if prog.Routes[0].Plugins[0].Name != "semantic_cache" {
		t.Errorf("expected normalized plugin name 'semantic_cache', got %q", prog.Routes[0].Plugins[0].Name)
	}
}

func TestLLMFriendlyErrorMessages(t *testing.T) {
	tests := []struct {
		name    string
		input   string
		wantMsg string
	}{
		{
			name: "unknown_signal_type",
			input: `SIGNAL foobar test { }
ROUTE r { PRIORITY 1 MODEL "q" }`,
			wantMsg: "Supported signal types:",
		},
		{
			name: "undefined_signal_ref",
			input: `ROUTE r {
  PRIORITY 1
  WHEN domain("math")
  MODEL "q"
}`,
			wantMsg: "Add SIGNAL",
		},
		{
			name: "no_model",
			input: `SIGNAL domain math { description: "m" }
ROUTE r {
  PRIORITY 1
  WHEN domain("math")
}`,
			wantMsg: "Add MODEL",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			diags, _ := Validate(tt.input)
			found := false
			for _, d := range diags {
				if strings.Contains(d.Message, tt.wantMsg) {
					found = true
					break
				}
			}
			if !found {
				msgs := make([]string, len(diags))
				for i, d := range diags {
					msgs[i] = d.Message
				}
				t.Errorf("expected diagnostic containing %q, got: %v", tt.wantMsg, msgs)
			}
		})
	}
}
