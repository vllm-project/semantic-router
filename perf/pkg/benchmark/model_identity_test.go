package benchmark

import (
	"encoding/json"
	"strings"
	"testing"
)

func modelFixture(revision string) *ModelIdentity {
	return &ModelIdentity{Provider: "candle", Device: "cpu", Precision: "float32", Protocol: "owned-v1", Artifacts: map[string]ModelArtifact{"embedding": {RepoID: "test/model", Revision: revision, ContentsSHA256: "contents"}}}
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
