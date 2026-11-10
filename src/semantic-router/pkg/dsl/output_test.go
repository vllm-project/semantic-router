package dsl

import (
	"slices"
	"strings"
	"testing"
)

const outputRoundtripDSL = `MODEL "painter" {
  capabilities: ["image_generation"]
  modalities: { input: ["text"], output: ["image"] }
}

MODEL "omni" {
  capabilities: ["chat", "image_generation"]
  modalities: { input: ["text", "image"], output: ["text", "image"] }
}

SIGNAL modality wants_image { labels: ["image"] }

SIGNAL modality wants_text_image { labels: ["text_image"] }

ROUTE image {
  PRIORITY 200
  WHEN modality("wants_image")
  MODEL "painter"
  OUTPUT {
    kind: "image"
    action: "generate"
  }
}

ROUTE text_and_image {
  PRIORITY 190
  WHEN modality("wants_text_image")
  MODEL "omni"
  OUTPUT {
    kind: ["text", "image"]
  }
}
`

// An OUTPUT block compiles to the decision's output contract and decompiles
// back to the same block, and the model facts it relies on survive.
func TestCompileOutputRoundTrip(t *testing.T) {
	cfg, errs := Compile(outputRoundtripDSL)
	if len(errs) > 0 {
		t.Fatalf("compile errors: %v", errs)
	}
	image, both := cfg.Decisions[0].Output, cfg.Decisions[1].Output
	if image == nil || !slices.Equal(image.Kinds(), []string{"image"}) || image.Action != "generate" {
		t.Fatalf("image output %+v", image)
	}
	if both == nil || !slices.Equal(both.Kinds(), []string{"text", "image"}) || both.Action != "" {
		t.Fatalf("text and image output %+v", both)
	}
	if !slices.Equal(cfg.ModelConfig["painter"].Modalities.Output, []string{"image"}) {
		t.Fatalf("painter modalities %+v", cfg.ModelConfig["painter"].Modalities)
	}
	out, err := Decompile(cfg)
	if err != nil {
		t.Fatalf("decompile: %v", err)
	}
	if !strings.Contains(out, "OUTPUT {\n    kind: \"image\"\n    action: \"generate\"\n  }") ||
		!strings.Contains(out, `kind: ["text", "image"]`) {
		t.Fatalf("decompiled DSL:\n%s", out)
	}
	again, errs := Compile(out)
	if len(errs) > 0 || !slices.Equal(again.Decisions[1].Output.Kinds(), []string{"text", "image"}) {
		t.Fatalf("round trip: %v %+v", errs, again.Decisions)
	}
	yamlBytes, errs := EmitRoutingYAML(outputRoundtripDSL)
	if len(errs) > 0 || !strings.Contains(string(yamlBytes), "output:") || !strings.Contains(string(yamlBytes), "modalities:") {
		t.Fatalf("routing YAML %v:\n%s", errs, yamlBytes)
	}
}

func TestOutputBlockErrors(t *testing.T) {
	for _, tc := range []struct{ block, want string }{
		{`OUTPUT { kind: "picture" }`, `ROUTE r OUTPUT: kind "picture" is not one of text, image, audio, video`},
		{`OUTPUT { kinds: "image" }`, `ROUTE r OUTPUT: unknown field "kinds". Supported fields: kind, action`},
		{`OUTPUT { kind: 3 }`, `ROUTE r OUTPUT: field "kind" must be one of text, image, audio, video, or a list of them, got int`},
		{`OUTPUT { kind: "text" action: "edit" }`, "action edit edits an image or a video the request provides, and kind text has neither"},
	} {
		input := "ROUTE r {\n  PRIORITY 1\n  MODEL \"m\"\n  " + tc.block + "\n}\n"
		if _, errs := Compile(input); !containsError(errs, tc.want) {
			t.Errorf("%s: errors %v, want %s", tc.block, errs, tc.want)
		}
		if diagnostics, _ := Validate(input); !containsDiagnostic(diagnostics, tc.want) {
			t.Errorf("%s: diagnostics %v, want %s", tc.block, diagnostics, tc.want)
		}
	}
}

func TestModelModalityKeywordIsRefused(t *testing.T) {
	_, errs := Compile("MODEL \"m\" { modality: \"omni\" }\n\nROUTE r {\n  PRIORITY 1\n  MODEL \"m\"\n}\n")
	if !containsError(errs, "MODEL m: modality was removed; declare what the model reads and writes in capabilities and modalities { input, output }") {
		t.Fatalf("errors %v", errs)
	}
}

func containsError(errs []error, want string) bool {
	return slices.ContainsFunc(errs, func(err error) bool { return strings.Contains(err.Error(), want) })
}

func containsDiagnostic(diagnostics []Diagnostic, want string) bool {
	return slices.ContainsFunc(diagnostics, func(diagnostic Diagnostic) bool { return strings.Contains(diagnostic.Message, want) })
}
