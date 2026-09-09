package v1alpha1

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// The CEL rules exist so an invalid complexity rule is a refused write rather
// than a Router crashloop. They live in kubebuilder markers, which are easy to
// drop or reword by accident and produce no build error when they disappear -
// the CRD simply stops validating. This asserts the generated artifacts still
// carry each expression.
//
// It checks presence, not CEL semantics: evaluating the expressions needs an
// API server, so a rule that is present but wrong would pass here. What it
// does catch is the failure that actually happened once on this branch - a
// generated CRD copy left behind by an API change.
func TestGeneratedCRDsCarryComplexityValidationRules(t *testing.T) {
	expressions := []string{
		// threshold and an explicit pair are mutually exclusive
		"!(has(self.threshold) && (has(self.hard_above) || has(self.easy_below) || has(self.hard_below) || has(self.easy_above)))",
		// one direction per rule
		"!((has(self.hard_above) || has(self.easy_below)) && (has(self.hard_below) || has(self.easy_above)))",
		// both halves of a pair, together
		"has(self.hard_above) == has(self.easy_below)",
		"has(self.hard_below) == has(self.easy_above)",
		// and ordered, so a reversed pair is refused at admission
		"double(self.easy_below) < double(self.hard_above)",
		"double(self.hard_below) < double(self.easy_above)",
		// complexity reads two response shapes, so it must name one
		"!has(self.backend) || has(self.backend.contract)",
	}

	// Both copies matter: config/crd/bases is what `make install` applies and
	// bundle/manifests is what an OLM install uses. A change landing in only
	// one leaves the other install path unvalidated.
	for _, relative := range []string{
		filepath.Join("..", "..", "config", "crd", "bases", "vllm.ai_semanticrouters.yaml"),
		filepath.Join("..", "..", "bundle", "manifests", "vllm.ai_semanticrouters.yaml"),
	} {
		data, err := os.ReadFile(relative)
		if err != nil {
			t.Fatalf("read %s: %v", relative, err)
		}
		// The generator wraps long lines, so compare with whitespace collapsed
		// rather than requiring the expression to sit on one line.
		flattened := strings.Join(strings.Fields(string(data)), " ")
		for _, expression := range expressions {
			if !strings.Contains(flattened, strings.Join(strings.Fields(expression), " ")) {
				t.Errorf("%s is missing the CEL rule %q; run 'make manifests' and 'make bundle'",
					filepath.Base(filepath.Dir(relative))+"/"+filepath.Base(relative), expression)
			}
		}
	}
}

// The PII module's remote backend and failure policy must be in both generated
// CRD copies, otherwise a Kubernetes user cannot select the token_spans.v1
// backend or its on_error policy at all (review on #3498).
func TestGeneratedCRDsCarryPIIBackendAndOnError(t *testing.T) {
	for _, relative := range []string{
		filepath.Join("..", "..", "config", "crd", "bases", "vllm.ai_semanticrouters.yaml"),
		filepath.Join("..", "..", "bundle", "manifests", "vllm.ai_semanticrouters.yaml"),
	} {
		data, err := os.ReadFile(relative)
		if err != nil {
			t.Fatalf("read %s: %v", relative, err)
		}
		flattened := strings.Join(strings.Fields(string(data)), " ")
		for _, want := range []string{
			"- token_spans.v1",
			"OnError selects what a PII backend failure",
			"Backend names a remote token classifier speaking token_spans.v1",
		} {
			if !strings.Contains(flattened, want) {
				t.Errorf("%s is missing %q; run 'make manifests' and refresh bundle/manifests",
					filepath.Base(filepath.Dir(relative))+"/"+filepath.Base(relative), want)
			}
		}
	}
}
