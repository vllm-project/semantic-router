package nlgen

import (
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestSchemaReferenceTeachesOnlyTypesTheRouterAccepts(t *testing.T) {
	algorithms := strings.FieldsFunc(schemaSection(t, "### Algorithm Types\n", "\n\n"), func(r rune) bool {
		return r == ',' || r == ' ' || r == '\n'
	})
	if len(algorithms) == 0 {
		t.Fatal("SchemaReference lists no algorithm types")
	}
	for _, algorithm := range algorithms {
		if !config.IsSupportedDecisionAlgorithmType(algorithm) {
			t.Errorf("SchemaReference teaches algorithm type %q, which the Router does not accept", algorithm)
		}
	}

	signals := 0
	for _, line := range strings.Split(schemaSection(t, "## Signal Types\n", "\n## "), "\n") {
		signal, ok := strings.CutPrefix(line, "### ")
		if !ok {
			continue
		}
		signals++
		if !config.IsSupportedSignalType(signal) {
			t.Errorf("SchemaReference teaches signal type %q, which the Router does not accept", signal)
		}
	}
	if signals == 0 {
		t.Fatal("SchemaReference lists no signal types")
	}
}

func schemaSection(t *testing.T, start, end string) string {
	t.Helper()
	_, section, ok := strings.Cut(SchemaReference, start)
	if !ok {
		t.Fatalf("SchemaReference has no %q section", strings.TrimSpace(start))
	}
	section, _, _ = strings.Cut(section, end)
	return section
}
