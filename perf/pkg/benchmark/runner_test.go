package benchmark

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

type fakeCommandExecutor struct {
	commands []Command
	output   []byte
	err      error
}

func (f *fakeCommandExecutor) Run(_ context.Context, command Command) ([]byte, error) {
	f.commands = append(f.commands, command)
	return f.output, f.err
}

func TestRunnerExecutesAndAggregatesManifestSuite(t *testing.T) {
	repoRoot := t.TempDir()
	if err := os.Mkdir(filepath.Join(repoRoot, "module"), 0o755); err != nil {
		t.Fatal(err)
	}
	manifest := &Manifest{
		SchemaVersion: ManifestSchemaVersion,
		Environments: map[string]EnvironmentConfig{
			"cpu": {Kind: "cpu", Capabilities: []string{"host"}},
		},
		Profiles: map[string]ProfileConfig{
			"ci": {Suites: []string{"core"}, Count: 2, BenchTime: "100ms", Timeout: "1m"},
		},
		Suites: map[string]SuiteConfig{
			"core": {
				Runner: "go_benchmark", Module: "module", Packages: []string{"./..."},
				Benchmark: "^BenchmarkCore$", Environments: []string{"cpu"}, Requires: []string{"host"},
			},
		},
	}
	fake := &fakeCommandExecutor{output: []byte(`BenchmarkCore-8  100  120 ns/op  64 B/op  2 allocs/op
BenchmarkCore-8  100  100 ns/op  64 B/op  2 allocs/op
`)}
	runner, err := newRunnerWithExecutor(manifest, repoRoot, fake)
	if err != nil {
		t.Fatal(err)
	}
	outputDir := filepath.Join(t.TempDir(), "report")
	result, err := runner.Run(context.Background(), RunOptions{Environment: "cpu", Profile: "ci", OutputDir: outputDir})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	metric := result.Benchmarks["BenchmarkCore"]
	if metric.Suite != "core" || metric.Samples != 2 || metric.NsPerOp != 110 {
		t.Fatalf("aggregated metric = %+v", metric)
	}
	if len(fake.commands) != 1 || fake.commands[0].Name != "go" {
		t.Fatalf("commands = %+v", fake.commands)
	}
	if joined := strings.Join(fake.commands[0].Args, " "); !strings.Contains(joined, "-count 2") {
		t.Fatalf("go benchmark args do not include profile count: %s", joined)
	}
	if _, err := os.Stat(filepath.Join(outputDir, "current.json")); err != nil {
		t.Fatalf("current.json not written: %v", err)
	}
}
