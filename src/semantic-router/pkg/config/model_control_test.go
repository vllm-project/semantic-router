package config

import (
	"strings"
	"testing"
)

func TestModelControlRoundTripsCanonicalConfig(t *testing.T) {
	document := strings.Replace(strictV03AuthoringYAML, "    - name: model-a\n", `    - name: model-a
      control:
        retry: {count: 2, on: [unavailable]}
        timeout: {request: 45s, stream: 3m}
`, 1)
	parsed, err := testAuthoringParser(t).ParseYAMLBytes([]byte(document))
	if err != nil {
		t.Fatalf("ParseYAMLBytes: %v", err)
	}
	execution := parsed.ModelConfig["model-a"].Execution
	if execution.MaxRetries != 2 || execution.RequestTimeout != "45s" || execution.StreamTimeout != "3m" {
		t.Fatalf("control did not normalize: %#v", execution)
	}
	exported := CanonicalConfigFromRouterConfig(parsed)
	var exportedControl ModelControl
	for _, model := range exported.Providers.Models {
		if model.Name == "model-a" {
			exportedControl = model.Control
		}
	}
	if len(exported.Providers.Models) != 3 || exportedControl.Retry == nil ||
		exportedControl.Timeout == nil || exportedControl.Retry.Count != execution.MaxRetries ||
		exportedControl.Timeout.Request != execution.RequestTimeout ||
		exportedControl.Timeout.Stream != execution.StreamTimeout {
		t.Fatalf("control did not round trip: %+v", exported.Providers.Models)
	}
}
