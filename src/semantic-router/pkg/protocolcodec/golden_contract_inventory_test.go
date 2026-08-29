package protocolcodec

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"testing"
)

type goldenMatrixInventory struct {
	Protocols []string                        `json:"protocols"`
	Scenarios []goldenMatrixInventoryScenario `json:"scenarios"`
}

type goldenMatrixInventoryScenario struct {
	Name   string            `json:"name"`
	Kind   string            `json:"kind"`
	Inputs map[string]string `json:"inputs"`
}

func TestGoldenOutputInventoryIsExact(t *testing.T) {
	if os.Getenv(updateProtocolGoldensEnv) == "1" {
		t.Skip("golden outputs are being regenerated")
	}
	for _, kind := range []string{"request", "response", "error", "stream", "capability", "rejection"} {
		t.Run(kind, func(t *testing.T) {
			assertGoldenOutputInventory(t, kind)
		})
	}
}

func assertGoldenOutputInventory(t *testing.T, kind string) {
	t.Helper()
	directory := filepath.Join("testdata", "golden", kind)
	inputs, err := filepath.Glob(filepath.Join(directory, "*-in.json"))
	if err != nil {
		t.Fatal(err)
	}
	if len(inputs) == 0 {
		t.Fatalf("no %s golden inputs found", kind)
	}
	expected := make([]string, 0, len(inputs)*len(goldenFormats))
	for _, input := range inputs {
		prefix := strings.TrimSuffix(filepath.Base(input), "-in.json")
		for _, target := range goldenFormats {
			expected = append(expected, prefix+"-"+target.name+"-out.json")
		}
	}
	outputs, err := filepath.Glob(filepath.Join(directory, "*-out.json"))
	if err != nil {
		t.Fatal(err)
	}
	actual := make([]string, 0, len(outputs))
	for _, output := range outputs {
		actual = append(actual, filepath.Base(output))
	}
	sort.Strings(expected)
	sort.Strings(actual)
	if !reflect.DeepEqual(actual, expected) {
		t.Fatalf("%s golden output inventory is not exact\n got: %v\nwant: %v", kind, actual, expected)
	}
}

func TestGoldenCoreMatrixInventoryIsClosed(t *testing.T) {
	body, err := os.ReadFile(filepath.Join("testdata", "contracts", "matrix-cases.json"))
	if err != nil {
		t.Fatal(err)
	}
	var inventory goldenMatrixInventory
	if err := json.Unmarshal(body, &inventory); err != nil {
		t.Fatal(err)
	}
	wantProtocols := []string{"anthropic", "chat", "responses"}
	gotProtocols := append([]string(nil), inventory.Protocols...)
	sort.Strings(gotProtocols)
	if !reflect.DeepEqual(gotProtocols, wantProtocols) {
		t.Fatalf("matrix protocols = %v, want %v", gotProtocols, wantProtocols)
	}
	if len(inventory.Scenarios) == 0 {
		t.Fatal("matrix inventory has no scenarios")
	}
	seen := make(map[string]struct{}, len(inventory.Scenarios))
	for _, scenario := range inventory.Scenarios {
		assertGoldenMatrixScenario(t, scenario, wantProtocols, seen)
	}
}

func assertGoldenMatrixScenario(
	t *testing.T,
	scenario goldenMatrixInventoryScenario,
	protocols []string,
	seen map[string]struct{},
) {
	t.Helper()
	if strings.TrimSpace(scenario.Name) == "" {
		t.Fatal("matrix scenario name is required")
	}
	if _, duplicate := seen[scenario.Name]; duplicate {
		t.Fatalf("duplicate matrix scenario %q", scenario.Name)
	}
	seen[scenario.Name] = struct{}{}
	if scenario.Kind != "request" && scenario.Kind != "response" && scenario.Kind != "error" && scenario.Kind != "stream" {
		t.Fatalf("matrix scenario %q has invalid kind %q", scenario.Name, scenario.Kind)
	}
	if len(scenario.Inputs) != len(protocols) {
		t.Fatalf("matrix scenario %q has %d sources, want %d", scenario.Name, len(scenario.Inputs), len(protocols))
	}
	for _, protocol := range protocols {
		assertGoldenScenarioInput(t, scenario, protocol)
	}
}

func assertGoldenScenarioInput(t *testing.T, scenario goldenMatrixInventoryScenario, protocol string) {
	t.Helper()
	name, ok := scenario.Inputs[protocol]
	if !ok {
		t.Fatalf("matrix scenario %q is missing %s input", scenario.Name, protocol)
	}
	if filepath.Base(name) != name || !strings.HasSuffix(name, "-in.json") || !strings.Contains(name, "-"+protocol+"-") {
		t.Fatalf("matrix scenario %q has invalid %s input %q", scenario.Name, protocol, name)
	}
	path := filepath.Join("testdata", "golden", scenario.Kind, name)
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("matrix scenario %q input %q: %v", scenario.Name, path, err)
	}
}
