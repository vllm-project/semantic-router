package dsl

import (
	"errors"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestDecompileYAMLRecipeBundleIsProviderNeutralAndExact(t *testing.T) {
	source := []byte(`
recipes:
  - name: first
    description: First reusable policy.
    routing:
      strategy: confidence
      decisions:
        - name: direct
          rules: {}
`)
	decompiled, err := DecompileYAML(source, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(decompiled, "# RECIPE first") ||
		!strings.Contains(decompiled, "strategy: confidence") ||
		strings.HasPrefix(decompiled, "# =============================================================================\n# SIGNALS") {
		t.Fatalf("Recipe bundle decompile contains the wrong scopes:\n%s", decompiled)
	}
	compiled, compileErrs := Compile(decompiled)
	if len(compileErrs) != 0 {
		t.Fatalf("compile decompiled Recipe bundle: %v", compileErrs)
	}
	roundTrip, err := Decompile(compiled)
	if err != nil {
		t.Fatal(err)
	}
	if roundTrip != decompiled {
		t.Fatalf("Recipe bundle round-trip is not byte exact")
	}
}

func TestDecompileYAMLRoutingFragmentIsProviderNeutral(t *testing.T) {
	source := []byte(`
routing:
  strategy: confidence
  decisions:
    - name: direct
      rules: {}
`)
	decompiled, err := DecompileYAML(source, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(decompiled, "ROUTE direct") || strings.Contains(decompiled, "# RECIPE") {
		t.Fatalf("routing fragment decompiled with the wrong scope:\n%s", decompiled)
	}
}

func TestDecompileYAMLAcceptsVersionedV03RecipeBundle(t *testing.T) {
	source := []byte(`
version: v0.3
recipes:
  - name: first
    routing:
      decisions:
        - name: direct
          rules: {}
`)
	decompiled, err := DecompileYAML(source, nil)
	if err != nil {
		t.Fatalf("DecompileYAML rejected a versioned v0.3 Recipe bundle: %v", err)
	}
	if !strings.Contains(decompiled, "# RECIPE first") {
		t.Fatalf("versioned Recipe bundle decompiled with the wrong scope:\n%s", decompiled)
	}
}

func TestDecompileYAMLRejectsUnknownRecipeBundleVersion(t *testing.T) {
	source := []byte("version: v0.4\nrecipes:\n  - name: first\n    routing:\n      decisions:\n        - name: direct\n          rules: {}\n")
	_, err := DecompileYAML(source, nil)
	if err == nil || !strings.Contains(err.Error(), "version must be v0.3") {
		t.Fatalf("DecompileYAML accepted an unknown Recipe bundle version: %v", err)
	}
}

func TestRecipeBundleRejectsPublicationAndLegacyEnvelopeFields(t *testing.T) {
	for _, field := range []string{"id", "revision", "document"} {
		t.Run(field, func(t *testing.T) {
			source := []byte("recipes:\n  - name: first\n    " + field + ": stale\n    routing:\n      decisions:\n        - name: direct\n          rules: {}\n")
			_, err := DecompileYAML(source, nil)
			if err == nil || !strings.Contains(err.Error(), field) {
				t.Fatalf("DecompileYAML accepted %s in a Recipe bundle: %v", field, err)
			}
		})
	}
}

func TestDecompileYAMLPreservesManifestAndRecipeErrors(t *testing.T) {
	manifestErr := errors.New("manifest sentinel")
	parser := func([]byte) (*config.RouterConfig, error) { return nil, manifestErr }
	_, err := DecompileYAML([]byte("unknown: true\n"), parser)
	if err == nil || !errors.Is(err, manifestErr) ||
		!strings.Contains(err.Error(), "recipe authoring YAML") {
		t.Fatalf("DecompileYAML error = %v", err)
	}
}
