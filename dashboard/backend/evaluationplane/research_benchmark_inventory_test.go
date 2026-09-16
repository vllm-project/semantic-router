package evaluationplane

import (
	"bytes"
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

func TestResearchBenchmarkInventoryMirrorMatchesCanonicalPythonPackageData(t *testing.T) {
	_, currentFile, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("resolve research benchmark inventory test location")
	}
	canonicalPath := filepath.Join(
		filepath.Dir(currentFile),
		"../../../src/vllm-sr/cli/evaluation/golden/research_benchmark_inventory.v1.json",
	)
	canonical, err := os.ReadFile(canonicalPath)
	if err != nil {
		t.Fatalf("read canonical Python package inventory: %v", err)
	}
	if !bytes.Equal(researchBenchmarkInventoryJSON, canonical) {
		t.Fatalf(
			"Go embedded inventory differs from canonical Python package data %q; run python3 tools/ci/sync_evaluation_catalogs.py",
			canonicalPath,
		)
	}
}

func TestResearchBenchmarkInventoryRejectsUnknownFields(t *testing.T) {
	mutated := bytes.Replace(
		researchBenchmarkInventoryJSON,
		[]byte(`"benchmarks":`),
		[]byte(`"baggage":true,"benchmarks":`),
		1,
	)
	if _, err := decodeResearchBenchmarkInventoryJSON(mutated); err == nil {
		t.Fatal("research benchmark inventory accepted unknown root baggage")
	}
}

func TestResearchBenchmarkInventoryIsExactlyTheAuditedThirteen(t *testing.T) {
	if err := ValidateResearchBenchmarkInventory(); err != nil {
		t.Fatalf("embedded research benchmark inventory is invalid: %v", err)
	}
	benchmarks := ResearchBenchmarkInventory()
	want := map[string]struct{}{
		"routerarena": {}, "routejudge-orbit": {}, "coderouterbench": {}, "llmrouterbench": {},
		"routereval": {}, "routerbench": {}, "xroutebench": {}, "twinrouterbench": {},
		"mmr-bench": {}, "acebench": {}, "continuity-bench": {}, "fusionfactory": {}, "r2-router": {},
	}
	if len(benchmarks) != len(want) {
		t.Fatalf("research benchmark inventory count=%d, want %d", len(benchmarks), len(want))
	}
	for _, benchmark := range benchmarks {
		if _, found := want[benchmark.AdapterID]; !found {
			t.Fatalf("unexpected research benchmark %q", benchmark.AdapterID)
		}
		delete(want, benchmark.AdapterID)
		method := researchBenchmarkMethod(benchmark)
		if method.Status != benchmark.Status || method.NativeParity != benchmark.NativeParity ||
			method.EvidenceCeiling != benchmark.EvidenceCeiling || method.Status == "native-qualified" ||
			method.NativeParity == "native" || method.EvidenceCeiling != "E0" {
			t.Fatalf("benchmark %q readiness drifted: benchmark=%+v method=%+v", benchmark.AdapterID, benchmark, method)
		}
	}
	if len(want) != 0 {
		t.Fatalf("research benchmark inventory is missing %v", want)
	}
}

func TestBlockedResearchBenchmarksCannotBecomeNormalizedImports(t *testing.T) {
	for _, adapterID := range []string{"routejudge-orbit", "routereval"} {
		benchmark, found := researchBenchmarkByAdapter(adapterID)
		if !found || benchmark.Status != "blocked" || len(benchmark.ImportTracks) != 0 {
			t.Fatalf("blocked benchmark %q inventory=%+v", adapterID, benchmark)
		}
		contract := normalizedAdapterContracts[adapterID]
		if normalizedAdapterTracksMatch(contract, []TrackID{"model_pool"}) {
			t.Fatalf("blocked benchmark %q accepted a normalized import track", adapterID)
		}
	}
}

func TestResearchBenchmarkAnalysisPlansAreDescriptorDriven(t *testing.T) {
	r2, found := researchBenchmarkByAdapter("r2-router")
	if !found {
		t.Fatal("r2-router benchmark is unavailable")
	}
	method := researchBenchmarkMethod(r2)
	if method.AnalysisPlan.ID != r2.AnalysisPlanID ||
		method.AnalysisPlan.AnalysisUnit != r2.AnalysisUnit ||
		method.AnalysisPlan.CurveDomain != r2.CurveDomain ||
		method.AnalysisPlan.CurveDomain != "shared_budget" {
		t.Fatalf("r2-router analysis plan drifted from its descriptor: benchmark=%+v method=%+v", r2, method)
	}
}
