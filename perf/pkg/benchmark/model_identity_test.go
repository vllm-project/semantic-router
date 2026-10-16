package benchmark

import (
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

func modelFixture(revision string) *ModelIdentity {
	return &ModelIdentity{Provider: "model_runtime", Device: "cpu", Precision: "float32", Protocol: "model-runtime-v1", Artifacts: map[string]ModelArtifact{"embedding": {RepoID: "test/model", Revision: revision, ContentsSHA256: "contents"}}}
}

func TestParseModelIdentityAfterBenchmarkResults(t *testing.T) {
	identity := modelFixture("commit-a")
	data, _ := json.Marshal(identity)
	output := "BenchmarkCacheSearch-8 10 25 ns/op 8 B/op 1 allocs/op\n" + ModelIdentityPrefix + "BenchmarkCacheSearch " + string(data) + "\n"
	parsed, err := ParseBenchOutput(strings.NewReader(output))
	if err != nil {
		t.Fatal(err)
	}
	if got := parsed.Benchmarks["BenchmarkCacheSearch"].ModelIdentity; got == nil || got.Artifacts["embedding"].Revision != "commit-a" {
		t.Fatalf("lost actual model identity: %+v", got)
	}
}

func TestModelComparisonRejectsChangedCheckpointOrContents(t *testing.T) {
	for _, change := range []string{"revision", "contents", "device", "missing"} {
		t.Run(change, func(t *testing.T) {
			current := &Baseline{Benchmarks: map[string]BenchmarkMetric{"BenchmarkCacheSearch": {ModelIdentity: modelFixture("commit-a")}}}
			prior := &Baseline{Benchmarks: map[string]BenchmarkMetric{"BenchmarkCacheSearch": {ModelIdentity: modelFixture("commit-a")}}}
			identity := prior.Benchmarks["BenchmarkCacheSearch"].ModelIdentity
			switch change {
			case "revision":
				identity.Artifacts["embedding"] = ModelArtifact{RepoID: "test/model", Revision: "commit-b", ContentsSHA256: "contents"}
			case "contents":
				identity.Artifacts["embedding"] = ModelArtifact{RepoID: "test/model", Revision: "commit-a", ContentsSHA256: "other"}
			case "device":
				identity.Device = "cuda:0"
			case "missing":
				delete(prior.Benchmarks, "BenchmarkCacheSearch")
			}
			if _, err := CompareWithBaseline(current, prior, nil); err == nil {
				t.Fatal("different or absent checkpoint baseline passed")
			}
		})
	}
}

func TestMeasuredModelBaselineOverlaysOnlyModels(t *testing.T) {
	current := &Baseline{Benchmarks: map[string]BenchmarkMetric{"BenchmarkCacheSearch": {ModelIdentity: modelFixture("commit-a"), AllocsPerOp: 20}}}
	committed := &Baseline{Benchmarks: map[string]BenchmarkMetric{"BenchmarkEvaluate": {AllocsPerOp: 10}}}
	measured := &Baseline{GitCommit: "base-source", Benchmarks: map[string]BenchmarkMetric{"BenchmarkCacheSearch": {ModelIdentity: modelFixture("commit-a"), AllocsPerOp: 10}, "BenchmarkEvaluate": {AllocsPerOp: 99}}}
	if err := OverlayModelBaseline(committed, current, measured); err != nil {
		t.Fatal(err)
	}
	if committed.Benchmarks["BenchmarkEvaluate"].AllocsPerOp != 10 {
		t.Fatal("model overlay replaced non-model baseline")
	}
	results, err := CompareWithBaseline(current, committed, nil)
	if err != nil || !HasRegressions(results) {
		t.Fatalf("same-checkpoint allocation regression escaped existing thresholds: %v %+v", err, results)
	}
	delete(measured.Benchmarks, "BenchmarkCacheSearch")
	if err := OverlayModelBaseline(committed, current, measured); err == nil {
		t.Fatal("unmeasured model silently passed")
	}
}

func TestResetModelBaselineLeavesModelsMeasuredButUngated(t *testing.T) {
	current := &Baseline{Benchmarks: map[string]BenchmarkMetric{
		"BenchmarkCacheSearch": {ModelIdentity: modelFixture("commit-a"), AllocsPerOp: 20},
		"BenchmarkEvaluate":    {AllocsPerOp: 10},
	}}
	committed := &Baseline{Benchmarks: map[string]BenchmarkMetric{"BenchmarkEvaluate": {AllocsPerOp: 10}}}
	reset := &Baseline{GitCommit: "base-source", ModelBaselineReset: "base predates the model runtime", Benchmarks: map[string]BenchmarkMetric{}}
	if err := OverlayModelBaseline(&Baseline{Benchmarks: map[string]BenchmarkMetric{}}, current, reset); err == nil {
		t.Fatal("a reset was accepted without the legacy-versus-runtime records")
	}
	reset.LegacyComparisonRecords = []string{"src/model-runtime/docs/records/decision1-parity.md"}
	if err := OverlayModelBaseline(committed, current, reset); err != nil {
		t.Fatal(err)
	}
	if committed.ModelBaselineReset == "" || !slices.Equal(committed.LegacyComparisonRecords, reset.LegacyComparisonRecords) {
		t.Fatal("the reset and its records were not recorded on the baseline")
	}
	if ungated := UngatedBenchmarks(current, committed); !slices.Equal(ungated, []string{"BenchmarkCacheSearch"}) {
		t.Fatalf("only model benchmarks may go ungated: %v", ungated)
	}
	results, err := CompareWithBaseline(current, committed, nil)
	if err != nil || len(results) != 1 || results[0].BenchmarkName != "BenchmarkEvaluate" {
		t.Fatalf("a reset must compare non-model benchmarks only: %v %+v", err, results)
	}
	if _, err := CompareWithBaseline(current, &Baseline{Benchmarks: committed.Benchmarks}, nil); err == nil {
		t.Fatal("a model benchmark without a reset or a measured baseline was compared")
	}
	path := filepath.Join(t.TempDir(), "inventory.json")
	if err := os.WriteFile(path, []byte(`{"version":1,"benchmarks":["BenchmarkCacheSearch","BenchmarkEvaluate"]}`), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := ValidateInventory(path, current, committed); err != nil {
		t.Fatal(err)
	}
	delete(committed.Benchmarks, "BenchmarkEvaluate")
	if err := ValidateInventory(path, current, committed); err == nil {
		t.Fatal("a reset excused a non-model baseline")
	}
	delete(current.Benchmarks, "BenchmarkCacheSearch")
	committed.Benchmarks["BenchmarkEvaluate"] = BenchmarkMetric{AllocsPerOp: 10}
	if err := ValidateInventory(path, current, committed); err == nil {
		t.Fatal("a reset excused an unmeasured current model benchmark")
	}
	reset.Benchmarks["BenchmarkCacheSearch"] = BenchmarkMetric{AllocsPerOp: 1}
	if err := OverlayModelBaseline(&Baseline{Benchmarks: map[string]BenchmarkMetric{}}, current, reset); err == nil {
		t.Fatal("a reset model baseline carried measurements")
	}
}
