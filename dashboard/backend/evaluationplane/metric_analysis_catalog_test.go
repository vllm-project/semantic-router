package evaluationplane

import (
	"bytes"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"sort"
	"strings"
	"testing"
)

func TestMetricAnalysisCatalogPackagedMirrorsAreByteIdentical(t *testing.T) {
	_, currentFile, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("resolve metric analysis catalog test location")
	}
	directory := filepath.Dir(currentFile)
	paths := []string{
		filepath.Join(directory, "../../../src/vllm-sr/cli/evaluation/golden/metric_analysis_catalog.v1.json"),
		filepath.Join(directory, "../../frontend/src/contracts/metric_analysis_catalog.v1.json"),
	}
	for _, path := range paths {
		mirrored, err := os.ReadFile(path)
		if err != nil {
			t.Fatalf("read metric analysis catalog mirror %q: %v", path, err)
		}
		if !bytes.Equal(metricAnalysisCatalogJSON, mirrored) {
			t.Fatalf(
				"metric analysis catalog mirror %q is stale; run python3 tools/ci/sync_evaluation_catalogs.py",
				path,
			)
		}
	}
}

func TestMetricAnalysisCatalogResolvesExactAndTypedDynamicMetrics(t *testing.T) {
	if err := ValidateMetricAnalysisCatalog(); err != nil {
		t.Fatalf("metric analysis catalog is invalid: %v", err)
	}
	if len(metricAnalysisCatalogData.document.StaticMetrics) != 136 || len(metricAnalysisCatalogData.document.DynamicFamilies) != 6 {
		t.Fatalf("metric analysis universe has unexpected cardinality")
	}
	staticIDs := make([]string, 0, len(metricAnalysisCatalogData.document.StaticMetrics))
	for _, metric := range metricAnalysisCatalogData.document.StaticMetrics {
		staticIDs = append(staticIDs, metric.ID)
	}
	if !sort.StringsAreSorted(staticIDs) {
		t.Fatal("metric analysis static ids are not sorted")
	}
	for _, metricID := range staticIDs {
		match, err := ResolveMetricAnalysisCatalog(metricID)
		if err != nil || match.MetricID != metricID || match.FamilyID != "" {
			t.Fatalf("resolve exact metric %q match=%+v err=%v", metricID, match, err)
		}
	}

	tests := []struct {
		id, family, analysisRef, analysisUnit, weighting string
	}{
		{"routing.accuracy", "", "routing.case.ratio", "route_case", "uniform_case"},
		{"agentic.recovery_cluster_pass_rate", "", "agentic.recovery.cluster-ratio", "recovery_cluster", "uniform_cluster"},
		{"agentic.recovery_cluster_pass_rate_lower_95", "", "agentic.recovery.cluster-wilson-bounds", "recovery_cluster", "uniform_cluster"},
		{"model_pool.arm.fast.quality", "model-pool-arm", "model-pool.arm.case-mean", "pool_case", "uniform_case"},
		{"model_pool.arm.u-bW9kZWwudjE.quality", "model-pool-arm", "model-pool.arm.case-mean", "pool_case", "uniform_case"},
		{"capacity.level.16.latency_p95_ms", "capacity-level", "capacity.level.request-quantile", "measurement_request", "uniform_request"},
		{"capacity.level.16.error_rate", "capacity-level", "capacity.level.request-repetition-rate-mean", "measurement_cluster", "uniform_cluster"},
		{"capacity.level.16.success_rate", "capacity-level", "capacity.level.request-repetition-rate-mean", "measurement_cluster", "uniform_cluster"},
		{"capacity.level.16.error_rate_upper_bound", "capacity-level", "capacity.level.request-worst-cluster-wilson-upper", "measurement_cluster", "worst_cluster"},
		{"capacity.level.16.error_rate_cluster_range", "capacity-level", "capacity.level.request-repetition-rate-range", "measurement_cluster", "uniform_cluster"},
		{"routing_recipe.e1.signal.u-ZG9tYWluOnJlYXNvbmluZw.present_rate", "routing-recipe-e1-input", "routing-recipe.e1.input-rate", "decision_input", "uniform_case"},
		{"routing_recipe.e2.feasible_oracle_recall_at_64", "routing-recipe-e2-recall", "routing-recipe.e2.feasible-recall", "decision_case", "uniform_case"},
	}
	for _, test := range tests {
		match, err := ResolveMetricAnalysisCatalog(test.id)
		if err != nil || match.FamilyID != test.family || match.Specification.AnalysisRef != test.analysisRef ||
			match.Specification.AnalysisUnit != test.analysisUnit || match.Specification.Weighting != test.weighting {
			t.Fatalf("resolve %q match=%+v err=%v", test.id, match, err)
		}
	}

	for _, unknown := range []string{
		"routing.injected",
		"model_pool.arm.model.v1.quality",
		"model_pool.arm.u-abc.quality",
		"capacity.level.0.success_rate",
		"capacity.level.16.injected",
		"routing_recipe.e2.feasible_oracle_recall_at_65",
	} {
		if _, err := ResolveMetricAnalysisCatalog(unknown); !errors.Is(err, ErrUnknownMetricAnalysisID) {
			t.Fatalf("unknown metric %q error=%v", unknown, err)
		}
	}
}

func TestMetricAnalysisSubjectCodecGoldenVectors(t *testing.T) {
	if err := ValidateMetricAnalysisCatalog(); err != nil {
		t.Fatal(err)
	}
	for _, vector := range metricAnalysisCatalogData.document.IdentifierEncoding.Vectors {
		encoded, err := EncodeMetricAnalysisSubjectID(vector.Raw)
		if err != nil || encoded != vector.Encoded {
			t.Fatalf("encode %q=%q err=%v, want %q", vector.Raw, encoded, err, vector.Encoded)
		}
		decoded, err := DecodeMetricAnalysisSubjectID(vector.Encoded)
		if err != nil || decoded != vector.Raw {
			t.Fatalf("decode %q=%q err=%v, want %q", vector.Encoded, decoded, err, vector.Raw)
		}
	}
	for raw, want := range map[string]string{
		"domain:reasoning":      "u-ZG9tYWluOnJlYXNvbmluZw",
		"classifier:risk:RISKY": "u-Y2xhc3NpZmllcjpyaXNrOlJJU0tZ",
	} {
		got, err := EncodeMetricAnalysisSubjectID(raw)
		if err != nil || got != want {
			t.Fatalf("router key codec %q=%q err=%v, want %q", raw, got, err, want)
		}
	}
}

func TestMetricAnalysisResolverRejectsAmbiguity(t *testing.T) {
	if err := ValidateMetricAnalysisCatalog(); err != nil {
		t.Fatal(err)
	}
	copyIndex := metricAnalysisCatalogData
	copyIndex.document.DynamicFamilies = append(
		append([]metricAnalysisDynamicFamily(nil), metricAnalysisCatalogData.document.DynamicFamilies...),
		metricAnalysisCatalogData.document.DynamicFamilies[0],
	)
	if _, err := resolveMetricAnalysisCatalog(&copyIndex, "capacity.level.16.success_rate"); !errors.Is(err, ErrAmbiguousMetricAnalysisID) {
		t.Fatalf("ambiguous resolver error=%v", err)
	}
}

func TestMetricAnalysisCatalogValidatorAcceptsReferencedExtensions(t *testing.T) {
	var document metricAnalysisCatalogDocument
	if err := json.Unmarshal(metricAnalysisCatalogJSON, &document); err != nil {
		t.Fatal(err)
	}
	var extension MetricAnalysisCatalogSpecification
	for _, template := range document.AnalysisTemplates {
		if template.AnalysisRef == "routing.case.ratio" {
			extension = template
			break
		}
	}
	if extension.AnalysisRef == "" {
		t.Fatal("routing case template is unavailable")
	}
	extension.AnalysisRef = "routing.catalog-extensibility-probe"
	document.AnalysisTemplates = append(document.AnalysisTemplates, extension)
	sort.Slice(document.AnalysisTemplates, func(left, right int) bool {
		return document.AnalysisTemplates[left].AnalysisRef < document.AnalysisTemplates[right].AnalysisRef
	})
	document.StaticMetrics = append(document.StaticMetrics, metricAnalysisStaticEntry{
		ID: "routing.catalog_extensibility_probe", AnalysisRef: extension.AnalysisRef,
	})
	sort.Slice(document.StaticMetrics, func(left, right int) bool {
		return document.StaticMetrics[left].ID < document.StaticMetrics[right].ID
	})
	encoded, marshalErr := json.Marshal(document)
	if marshalErr != nil {
		t.Fatal(marshalErr)
	}
	if _, err := decodeMetricAnalysisCatalog(encoded); err != nil {
		t.Fatalf("referenced catalog extension rejected: %v", err)
	}

	orphan := extension
	orphan.AnalysisRef = "routing.unreferenced-probe"
	document.AnalysisTemplates = append(document.AnalysisTemplates, orphan)
	sort.Slice(document.AnalysisTemplates, func(left, right int) bool {
		return document.AnalysisTemplates[left].AnalysisRef < document.AnalysisTemplates[right].AnalysisRef
	})
	encoded, marshalErr = json.Marshal(document)
	if marshalErr != nil {
		t.Fatal(marshalErr)
	}
	if _, err := decodeMetricAnalysisCatalog(encoded); err == nil || !strings.Contains(err.Error(), "referenced exhaustively") {
		t.Fatalf("unreferenced template error=%v", err)
	}
}

func TestMetricAnalysisCatalogRootRejectsRemovedBaggage(t *testing.T) {
	withBaggage := bytes.Replace(
		metricAnalysisCatalogJSON,
		[]byte("{"),
		[]byte(`{"legacy_metric_inventory":[],`),
		1,
	)
	if _, err := decodeMetricAnalysisCatalog(withBaggage); err == nil {
		t.Fatal("unknown root baggage was accepted")
	}
}

func TestBuiltinTrackCapabilitiesComeOnlyFromCanonicalMetricCatalog(t *testing.T) {
	if err := ValidateMetricAnalysisCatalog(); err != nil {
		t.Fatal(err)
	}
	seen := make(map[string]struct{}, len(metricAnalysisCatalogData.document.StaticMetrics))
	for _, track := range builtinTracks() {
		want := StaticMetricAnalysisIDsForTrack(track.ID)
		if !slices.Equal(track.Metrics, want) {
			t.Fatalf("track %q metrics=%v, want canonical %v", track.ID, track.Metrics, want)
		}
		for _, metricID := range track.Metrics {
			if _, duplicate := seen[metricID]; duplicate {
				t.Fatalf("canonical metric %q is advertised by multiple tracks", metricID)
			}
			seen[metricID] = struct{}{}
		}
	}
	if len(seen) != len(metricAnalysisCatalogData.document.StaticMetrics) {
		t.Fatalf("advertised %d canonical metrics, want %d", len(seen), len(metricAnalysisCatalogData.document.StaticMetrics))
	}
}
