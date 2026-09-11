package evaluationplane

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"
)

type capacityClusterParityFixture struct {
	SchemaVersion                      string  `json:"schema_version"`
	ConfidenceLevel                    float64 `json:"confidence_level"`
	MinimumMeasurementClustersPerLevel int64   `json:"minimum_measurement_clusters_per_level"`
	MaxErrorRateClusterRange           float64 `json:"max_error_rate_cluster_range"`
	Levels                             []struct {
		Concurrency int64 `json:"concurrency"`
		Clusters    []struct {
			LoadPhase           string  `json:"load_phase"`
			LoadRepetition      int64   `json:"load_repetition"`
			Requests            int64   `json:"requests"`
			Errors              int64   `json:"errors"`
			ErrorRate           float64 `json:"error_rate"`
			ErrorRateUpperBound float64 `json:"error_rate_upper_bound"`
		} `json:"clusters"`
		Expected struct {
			MeasurementClusterCount int64   `json:"measurement_cluster_count"`
			ErrorRate               float64 `json:"error_rate"`
			ErrorRateUpperBound     float64 `json:"error_rate_upper_bound"`
			ErrorRateClusterRange   float64 `json:"error_rate_cluster_range"`
		} `json:"expected"`
	} `json:"levels"`
	ExpectedSummary struct {
		MeasurementClusterCount    int     `json:"measurement_cluster_count"`
		MeasurementClusterCountMin int64   `json:"measurement_cluster_count_min"`
		ErrorRate                  float64 `json:"error_rate"`
		SuccessRate                float64 `json:"success_rate"`
		ErrorRateUpperBound        float64 `json:"error_rate_upper_bound"`
		ErrorRateClusterRangeMax   float64 `json:"error_rate_cluster_range_max"`
	} `json:"expected_summary"`
}

func TestCapacityClusterReducerMatchesSharedGoPythonFixture(t *testing.T) {
	_, currentFile, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("resolve capacity parity fixture location")
	}
	fixturePath := filepath.Join(filepath.Dir(currentFile), "../../../src/vllm-sr/tests/fixtures/capacity_cluster_metric_parity.v1.json")
	encoded, err := os.ReadFile(fixturePath)
	if err != nil {
		t.Fatal(err)
	}
	var fixture capacityClusterParityFixture
	if err := json.Unmarshal(encoded, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.SchemaVersion != "capacity-cluster-metric-parity.v1" ||
		fixture.ConfidenceLevel != capacityLoadConfidence ||
		fixture.MinimumMeasurementClustersPerLevel != minimumCapacityMeasurementClusters ||
		fixture.MaxErrorRateClusterRange != capacityMaxErrorRateClusterRange {
		t.Fatalf("unexpected capacity cluster parity contract: %#v", fixture)
	}

	allRates := make([]float64, 0, fixture.ExpectedSummary.MeasurementClusterCount)
	worstUpper := 0.0
	worstRange := 0.0
	minimumClusters := int64(0)
	for levelIndex, level := range fixture.Levels {
		rates := make([]float64, 0, len(level.Clusters))
		levelWorstUpper := 0.0
		for clusterIndex, cluster := range level.Clusters {
			if cluster.LoadPhase != "measurement" || cluster.LoadRepetition != int64(clusterIndex+1) {
				t.Fatalf("level %d cluster %d has a non-canonical identity", levelIndex+1, clusterIndex+1)
			}
			rate := float64(cluster.Errors) / float64(cluster.Requests)
			upper := capacityOneSidedWilsonUpper(cluster.Errors, cluster.Requests)
			if !reducedFloatsEqual(rate, cluster.ErrorRate) ||
				!reducedFloatsEqual(upper, cluster.ErrorRateUpperBound) {
				t.Fatalf("level %d cluster %d statistics differ: rate=%g upper=%g", levelIndex+1, clusterIndex+1, rate, upper)
			}
			rates = append(rates, rate)
			allRates = append(allRates, rate)
			levelWorstUpper = math.Max(levelWorstUpper, upper)
		}
		clusterCount := int64(len(rates))
		if clusterCount != level.Expected.MeasurementClusterCount ||
			!reducedFloatsEqual(capacityMean(rates), level.Expected.ErrorRate) ||
			!reducedFloatsEqual(levelWorstUpper, level.Expected.ErrorRateUpperBound) ||
			!reducedFloatsEqual(capacityRange(rates), level.Expected.ErrorRateClusterRange) {
			t.Fatalf("level %d aggregate differs from fixture", levelIndex+1)
		}
		if minimumClusters == 0 || clusterCount < minimumClusters {
			minimumClusters = clusterCount
		}
		worstUpper = math.Max(worstUpper, levelWorstUpper)
		worstRange = math.Max(worstRange, capacityRange(rates))
	}
	if len(allRates) != fixture.ExpectedSummary.MeasurementClusterCount ||
		minimumClusters != fixture.ExpectedSummary.MeasurementClusterCountMin ||
		!reducedFloatsEqual(capacityMean(allRates), fixture.ExpectedSummary.ErrorRate) ||
		!reducedFloatsEqual(1-capacityMean(allRates), fixture.ExpectedSummary.SuccessRate) ||
		!reducedFloatsEqual(worstUpper, fixture.ExpectedSummary.ErrorRateUpperBound) ||
		!reducedFloatsEqual(worstRange, fixture.ExpectedSummary.ErrorRateClusterRangeMax) {
		t.Fatalf("capacity cluster summary differs from fixture")
	}
}

func TestValidateCapacityProfileArtifactAcceptsRepeatedClosedLoopEvidence(t *testing.T) {
	runDir := t.TempDir()
	writeCapacityRecords(t, runDir)
	writeCapacityProfile(t, runDir, capacityTestProfile())
	attestation, err := validateCapacityProfileArtifact(
		runDir,
		capacityManifest(),
		capacityReport(),
		capacityRecordsAttestation(),
	)
	if err != nil {
		t.Fatalf("validateCapacityProfileArtifact: %v", err)
	}
	if attestation == nil || attestation.Headroom != 1 || attestation.LevelCount != 2 ||
		attestation.MeasurementClusterCount != 6 || attestation.MinimumClustersPerLevel != 3 ||
		attestation.RequiredClustersPerLevel != minimumCapacityMeasurementClusters ||
		!reducedFloatsEqual(attestation.WorstErrorRateUpperBound, capacityOneSidedWilsonUpper(0, 100)) ||
		!reducedFloatsEqual(attestation.ReleaseErrorRateUpperBound, capacityOneSidedWilsonUpper(0, 100)) ||
		attestation.WorstErrorRateClusterRange != 0 ||
		attestation.ReleaseErrorRateClusterRange != 0 ||
		attestation.MaxErrorRateClusterRange != capacityMaxErrorRateClusterRange {
		t.Fatalf("capacity SLO attestation = %#v", attestation)
	}
}

func TestCapacityReportMetricsAreAttestedFromIndependentClusters(t *testing.T) {
	runDir := t.TempDir()
	writeCapacityRecords(t, runDir)
	writeCapacityProfile(t, runDir, capacityTestProfile())
	manifest := capacityManifest()
	attestation, err := validateCapacityProfileArtifact(
		runDir, manifest, capacityReport(), capacityRecordsAttestation(),
	)
	if err != nil {
		t.Fatal(err)
	}
	report := Report{
		Run: Run{
			TrackIDs: []TrackID{"capacity"}, CapacitySLO: manifest.CapacitySLO,
			CapacityLoadProtocol: manifest.CapacityLoadProtocol,
		},
		Metrics: capacityClusterAttestedMetrics(attestation),
	}
	if err := validateCapacitySLOMetric(report, attestation); err != nil {
		t.Fatalf("cluster-attested capacity metrics rejected: %v", err)
	}
	for _, metric := range report.Metrics {
		if metric.ID == "capacity.error_rate_cluster_range_max" && metric.SampleCount != attestation.LevelCount {
			t.Fatalf("cluster-range analysis sample count=%d, want %d load levels", metric.SampleCount, attestation.LevelCount)
		}
		if metric.ID == "capacity.error_rate_upper_bound" && metric.SampleCount != attestation.MeasurementClusterCount {
			t.Fatalf("error-bound analysis sample count=%d, want %d clusters", metric.SampleCount, attestation.MeasurementClusterCount)
		}
	}
	forged := report
	forged.Metrics = append([]Metric(nil), report.Metrics...)
	for index := range forged.Metrics {
		if forged.Metrics[index].ID == "capacity.error_rate_upper_bound" {
			forged.Metrics[index].Value = capacityFloatPointer(capacityOneSidedWilsonUpper(0, 300))
		}
	}
	if err := validateCapacitySLOMetric(forged, attestation); !errors.Is(err, ErrInvalid) {
		t.Fatalf("pooled report error bound accepted: %v", err)
	}
}

func TestRecordedCapacityCannotPublishPooledRequestStatistics(t *testing.T) {
	report := Report{
		Run:     Run{TrackIDs: []TrackID{"capacity"}},
		Metrics: capacityClusterAttestedMetrics(nil),
	}
	if err := validateCapacitySLOMetric(report, nil); err != nil {
		t.Fatalf("unavailable recorded-capacity cluster metrics rejected: %v", err)
	}
	for _, metric := range report.Metrics {
		if metric.Value != nil || metric.SampleCount != 0 || metric.ConfidenceInterval != nil {
			t.Fatalf("recorded capacity metric %s published unsupported statistics", metric.ID)
		}
	}
	forged := report
	forged.Metrics = append([]Metric(nil), report.Metrics...)
	forged.Metrics[0].Value = capacityFloatPointer(0)
	forged.Metrics[0].SampleCount = 300
	if err := validateCapacitySLOMetric(forged, nil); !errors.Is(err, ErrInvalid) {
		t.Fatalf("pooled recorded-capacity statistic accepted: %v", err)
	}
}

func TestValidateCapacityProfileArtifactRejectsMalformedOrWeakEvidence(t *testing.T) {
	valid, err := json.Marshal(capacityTestProfile())
	if err != nil {
		t.Fatal(err)
	}
	tests := []struct {
		name string
		raw  []byte
	}{
		{name: "malformed JSON", raw: []byte("{not-json\n")},
		{name: "unknown field", raw: bytes.Replace(valid, []byte(`"kind":"repeated-closed-loop-capacity"`), []byte(`"kind":"repeated-closed-loop-capacity","forged":true`), 1)},
		{name: "missing protocol", raw: bytes.Replace(valid, []byte(`"protocol":{`), []byte(`"protocol":null,"discarded":{`), 1)},
		{name: "tiny measurement window", raw: bytes.Replace(valid, []byte(`"measurement_requests_per_repetition":100`), []byte(`"measurement_requests_per_repetition":2`), 1)},
		{name: "forged derived cluster range", raw: bytes.Replace(valid, []byte(`"error_rate_cluster_range":0`), []byte(`"error_rate_cluster_range":0.1`), 1)},
		{name: "missing repetition", raw: bytes.Replace(valid, []byte(`"repetition":2`), []byte(`"repetition":9`), 1)},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			runDir := t.TempDir()
			writeCapacityRecords(t, runDir)
			if err := os.WriteFile(filepath.Join(runDir, capacityProfileArtifactName), test.raw, 0o600); err != nil {
				t.Fatal(err)
			}
			_, validationErr := validateCapacityProfileArtifact(
				runDir,
				capacityManifest(),
				capacityReport(),
				capacityRecordsAttestation(),
			)
			if !errors.Is(validationErr, ErrInvalid) {
				t.Fatalf("validation error=%v, want ErrInvalid", validationErr)
			}
		})
	}
}

func TestValidateCapacityProfileArtifactRejectsPooledRequestErrorBound(t *testing.T) {
	runDir := t.TempDir()
	writeCapacityRecords(t, runDir)
	profile := capacityTestProfile()
	pooledRequestUpper := capacityOneSidedWilsonUpper(0, 300)
	for index := range profile.Levels {
		profile.Levels[index].ErrorRateUpperBound = capacityFloatPointer(pooledRequestUpper)
	}
	writeCapacityProfile(t, runDir, profile)
	_, err := validateCapacityProfileArtifact(
		runDir,
		capacityManifest(),
		capacityReport(),
		capacityRecordsAttestation(),
	)
	if !errors.Is(err, ErrInvalid) || !strings.Contains(err.Error(), "does not match records") {
		t.Fatalf("pooled request bound error=%v, want independent-cluster mismatch ErrInvalid", err)
	}
}

func TestValidateCapacityProfileArtifactRejectsSelfConsistentProfileForgery(t *testing.T) {
	runDir := t.TempDir()
	writeCapacityRecords(t, runDir)
	profile := capacityTestProfile()
	*profile.Levels[0].Throughput += 1
	writeCapacityProfile(t, runDir, profile)
	_, err := validateCapacityProfileArtifact(
		runDir,
		capacityManifest(),
		capacityReport(),
		capacityRecordsAttestation(),
	)
	if !errors.Is(err, ErrInvalid) || !strings.Contains(err.Error(), "does not match records") {
		t.Fatalf("forged profile error=%v, want record mismatch ErrInvalid", err)
	}
}

func TestRealWorkerSealRejectsForgedCapacityProfileWithRecomputedReceipts(t *testing.T) {
	python := realEvaluationWorkerPython(t)
	service, root, run := newRealCapacitySealFixture(t, python)
	publishRealCapacityEvidence(t, service, root, run, python)

	pristine, pristineErr := service.prepareReportSeal(run.ID)
	if pristineErr != nil {
		t.Fatalf("prepare pristine real capacity seal: %v", pristineErr)
	}
	if pristine.sealedLevels.Run != "E5" || pristine.sealedLevels.ByTrack["capacity"] != "E5" {
		t.Fatalf("pristine capacity evidence levels = %#v, want E5", pristine.sealedLevels)
	}
	if err := forgeCapacityProfileWithRecomputedReceipts(filepath.Join(root, "runs", run.ID)); err != nil {
		t.Fatalf("forge self-consistent capacity bundle: %v", err)
	}
	if _, err := service.validatePrivateReceipt(run.ID); err != nil {
		t.Fatalf("recomputed private receipt is not self-consistent: %v", err)
	}
	sealErr := service.store.withEvidencePublication(func() error {
		return service.validateAndAnchorReportDuringPublication(run.ID)
	})
	if !errors.Is(sealErr, ErrInvalid) || !strings.Contains(sealErr.Error(), "capacity profile does not match records") {
		t.Fatalf("forged capacity seal error=%v, want records mismatch ErrInvalid", sealErr)
	}
}

func realEvaluationWorkerPython(t *testing.T) string {
	t.Helper()
	python := os.Getenv("VLLM_SR_EVALUATION_TEST_PYTHON")
	if python == "" {
		t.Skip("set VLLM_SR_EVALUATION_TEST_PYTHON to run the real Python worker")
	}
	pythonRoot, err := filepath.Abs("../../../src/vllm-sr")
	if err != nil {
		t.Fatal(err)
	}
	t.Setenv("PYTHONPATH", pythonRoot)
	t.Setenv("TMPDIR", "/tmp")
	return python
}

func newRealCapacitySealFixture(t *testing.T, python string) (*Service, string, Run) {
	t.Helper()
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		writer.Header().Set("Content-Type", "application/json")
		switch request.URL.Path {
		case "/v1/models":
			_, _ = writer.Write([]byte(`{"data":[{"id":"entrypoint-a","routing":{"resolution":"virtual","selectable":true,"default_route":true,"recipe":"default"}}]}`))
		case "/v1/chat/completions":
			writer.Header().Set("x-vsr-selected-model", "Org/Fast Model")
			writer.Header().Set("x-vsr-selected-algorithm", "static")
			writer.Header().Set("x-vsr-selected-recipe", "default")
			writer.Header().Set("x-vsr-selected-decision", "route")
			_, _ = writer.Write([]byte(`{"choices":[{"message":{"content":"ok"}}],"usage":{"prompt_tokens":10,"completion_tokens":2}}`))
		default:
			http.NotFound(writer, request)
		}
	}))
	t.Cleanup(server.Close)

	root := filepath.Join(t.TempDir(), "evaluation")
	if mkdirErr := os.Mkdir(root, 0o700); mkdirErr != nil {
		t.Fatal(mkdirErr)
	}
	configPath := filepath.Join(root, "config.yaml")
	if writeErr := os.WriteFile(configPath, []byte(modelArmTestYAML), 0o600); writeErr != nil {
		t.Fatal(writeErr)
	}
	service, serviceErr := NewService(Options{
		DataDir: root, PythonPath: python, ConfigPath: configPath,
		RouterAPIURL: server.URL, EnvoyURL: server.URL,
		CodeRevision: testSourceRevision, MaxConcurrent: 1, Process: &controlledProcess{},
	})
	if serviceErr != nil {
		t.Fatalf("NewService: %v", serviceErr)
	}
	t.Cleanup(func() { _ = service.Close() })
	run, createErr := service.CreateRunAs(context.Background(), SystemActor(), CreateRunRequest{
		ClientRequestID: newTestClientRequestID(),
		Name:            "capacity receipt forgery", SuiteIDs: []string{"live-capacity"},
		TrackIDs: []TrackID{"capacity"}, Mode: ModeLive, TargetID: mixtureTargetID("default"),
		ChangeProfile: "runtime_capacity", SampleLimit: 1, Concurrency: 2, Seed: 17,
		CapacitySLO:          testCapacitySLO(2),
		CapacityLoadProtocol: defaultCapacityLoadProtocol(2),
	})
	if createErr != nil {
		t.Fatalf("CreateRun: %v", createErr)
	}
	startedAt := time.Now().UTC()
	run.Status = StatusRunning
	run.StartedAt = &startedAt
	if updateErr := service.store.updateRunFixture(run); updateErr != nil {
		t.Fatalf("stage running run: %v", updateErr)
	}
	return service, root, run
}

func publishRealCapacityEvidence(t *testing.T, service *Service, root string, run Run, python string) {
	t.Helper()
	spec := ProcessSpec{
		ManifestPath:        filepath.Join(root, "runs", run.ID, manifestFileName),
		StorePath:           root,
		SuiteStorePath:      service.store.SuiteRoot(),
		executionContracts:  serviceExecutionContractsForTest(t, service),
		evidencePublication: service.store.withEvidenceSerialization,
	}
	result, processErr := NewCommandProcess(python).Run(context.Background(), spec, func(WorkerEvent) error { return nil })
	if processErr != nil {
		t.Fatalf("real capacity worker: %v", processErr)
	}
	t.Cleanup(result.discardStagedEvidence)
	if err := service.beginSealing(run.ID); err != nil {
		t.Fatalf("begin capacity evidence sealing: %v", err)
	}
	if err := result.publishStagedEvidence(); err != nil {
		t.Fatalf("publish capacity worker evidence: %v", err)
	}
	if _, err := service.persistExecutionAttestation(run.ID, result.ExecutionTranscript); err != nil {
		t.Fatalf("attest real capacity worker: %v", err)
	}
}

func forgeCapacityProfileWithRecomputedReceipts(runDir string) error {
	var profile map[string]any
	if err := readJSON(filepath.Join(runDir, capacityProfileArtifactName), &profile); err != nil {
		return err
	}
	levels, ok := profile["levels"].([]any)
	if !ok || len(levels) == 0 {
		return fmt.Errorf("capacity profile has no forgeable level")
	}
	level, ok := levels[0].(map[string]any)
	if !ok {
		return fmt.Errorf("capacity profile level is not an object")
	}
	throughput, ok := level["throughput_rps"].(float64)
	if !ok {
		return fmt.Errorf("capacity throughput is not numeric")
	}
	level["throughput_rps"] = throughput + 1
	if err := writeJSONAtomic(filepath.Join(runDir, capacityProfileArtifactName), profile); err != nil {
		return err
	}

	var report Report
	if err := readJSON(filepath.Join(runDir, reportFileName), &report); err != nil {
		return err
	}
	var receipt strings.Builder
	update := func(artifact *Artifact, includeReceipt bool) error {
		if artifact.Name == publicChecksumArtifactName {
			return nil
		}
		data, err := os.ReadFile(filepath.Join(runDir, artifact.Name))
		if err != nil {
			return err
		}
		artifact.Digest = digestBytes(data)
		artifact.SizeBytes = int64(len(data))
		if includeReceipt {
			receipt.WriteString(strings.TrimPrefix(artifact.Digest, "sha256:"))
			receipt.WriteString("  ")
			receipt.WriteString(artifact.Name)
			receipt.WriteByte('\n')
		}
		return nil
	}
	for index := range report.Artifacts {
		if err := update(&report.Artifacts[index], true); err != nil {
			return err
		}
	}
	for trackIndex := range report.Tracks {
		for artifactIndex := range report.Tracks[trackIndex].Artifacts {
			if err := update(&report.Tracks[trackIndex].Artifacts[artifactIndex], true); err != nil {
				return err
			}
		}
	}
	receiptBytes := []byte(receipt.String())
	if err := os.WriteFile(filepath.Join(runDir, publicChecksumArtifactName), receiptBytes, 0o600); err != nil {
		return err
	}
	updateReceipt := func(artifact *Artifact) {
		if artifact.Name == publicChecksumArtifactName {
			artifact.Digest = digestBytes(receiptBytes)
			artifact.SizeBytes = int64(len(receiptBytes))
		}
	}
	for index := range report.Artifacts {
		updateReceipt(&report.Artifacts[index])
	}
	for trackIndex := range report.Tracks {
		for artifactIndex := range report.Tracks[trackIndex].Artifacts {
			updateReceipt(&report.Tracks[trackIndex].Artifacts[artifactIndex])
		}
	}
	if err := writeJSONAtomic(filepath.Join(runDir, reportFileName), workerReportFromReport(report)); err != nil {
		return err
	}
	return writeTestPrivateReceiptWithoutTesting(runDir)
}

func capacityManifest() RunManifest {
	return RunManifest{
		Mode: ModeLive, TrackIDs: []TrackID{"capacity"}, Concurrency: 2,
		CapacitySLO:          testCapacitySLO(1),
		CapacityLoadProtocol: defaultCapacityLoadProtocol(2),
	}
}

func testCapacitySLO(required int64) *CapacitySLO {
	return &CapacitySLO{
		SchemaVersion: SchemaVersion, RequiredConcurrency: required,
		MaxLatencyP95MS: 30, MaxErrorRate: 0.05, MinThroughputRPS: 1,
		MinThroughputScalingEfficiency: 0.5,
	}
}

func capacityReport() Report {
	return Report{Artifacts: []Artifact{{
		Name: capacityProfileArtifactName, URI: capacityProfileArtifactName, MediaType: "application/json",
	}}}
}

func capacityTestProfile() capacityProfileEvidence {
	protocol := defaultCapacityLoadProtocol(2)
	slo := testCapacitySLO(1)
	levels := make([]capacityProfileLevel, 0, len(protocol.ConcurrencyLevels))
	for levelIndex, concurrency := range protocol.ConcurrencyLevels {
		throughput := float64(concurrency * 10)
		repetitions := make([]capacityProfileRepetition, 0, protocol.RepetitionsPerLevel)
		for repetition := int64(1); repetition <= protocol.RepetitionsPerLevel; repetition++ {
			errorUpper := capacityOneSidedWilsonUpper(0, 100)
			repetitions = append(repetitions, capacityProfileRepetition{
				Concurrency: capacityInt64Pointer(concurrency), Repetition: capacityInt64Pointer(repetition),
				Requests: capacityInt64Pointer(100), Successes: capacityInt64Pointer(100), Errors: capacityInt64Pointer(0),
				Elapsed: capacityFloatPointer(100 / throughput), Throughput: capacityFloatPointer(throughput),
				LatencyP95MS: capacityFloatPointer(20),
				ErrorRate:    capacityFloatPointer(0), ErrorUpper: capacityFloatPointer(errorUpper),
			})
		}
		runtimeCost := 0.0
		for range 300 {
			runtimeCost += 0.00001488
		}
		scaling := json.RawMessage("null")
		if levelIndex > 0 {
			scaling = json.RawMessage("1")
		}
		levels = append(levels, capacityProfileLevel{
			Concurrency:    capacityInt64Pointer(concurrency),
			WarmupRequests: capacityInt64Pointer(concurrency * 2), WarmupErrors: capacityInt64Pointer(0),
			WarmupElapsed: capacityFloatPointer(0.2), MeasurementRequests: capacityInt64Pointer(300),
			Successes: capacityInt64Pointer(300), Errors: capacityInt64Pointer(0),
			Elapsed: capacityFloatPointer(300 / throughput), Throughput: capacityFloatPointer(throughput),
			ThroughputCV: capacityFloatPointer(0), LatencyP50MS: capacityFloatPointer(20),
			LatencyP95MS: capacityFloatPointer(20), LatencyP99MS: capacityFloatPointer(20),
			LatencyP95CV: capacityFloatPointer(0), ErrorRate: capacityFloatPointer(0),
			ErrorRateUpperBound:     capacityFloatPointer(capacityOneSidedWilsonUpper(0, 100)),
			MeasurementClusterCount: capacityInt64Pointer(3), ErrorRateClusterRange: capacityFloatPointer(0),
			InputTokens: capacityInt64Pointer(300), OutputTokens: capacityInt64Pointer(300),
			RuntimeCost: capacityFloatPointer(runtimeCost), Repetitions: repetitions,
			ScalingEfficiency: scaling, WarmupPassed: capacityBoolPointer(true),
			LatencySLOPassed: capacityBoolPointer(true), ClusterCoveragePassed: capacityBoolPointer(true),
			ErrorRateStabilityPassed: capacityBoolPointer(true), ErrorSLOPassed: capacityBoolPointer(true),
			ThroughputSLOPassed: capacityBoolPointer(true), ScalingSLOPassed: capacityBoolPointer(true),
			ThroughputStabilityPassed: capacityBoolPointer(true), LatencyStabilityPassed: capacityBoolPointer(true),
			Qualified: capacityBoolPointer(true),
		})
	}
	return capacityProfileEvidence{
		SchemaVersion: SchemaVersion, Kind: "repeated-closed-loop-capacity",
		Protocol: protocol, Levels: levels, SLO: slo,
		Assessment: capacityProfileAssessment{
			QualifiedConcurrency: json.RawMessage("2"), SaturationConcurrency: json.RawMessage("null"),
			SLOHeadroom: capacityInt64Pointer(1), Verdict: "pass", FailureReasons: []string{},
		},
	}
}

func writeCapacityProfile(t *testing.T, runDir string, profile capacityProfileEvidence) {
	t.Helper()
	if err := writeJSONAtomic(filepath.Join(runDir, capacityProfileArtifactName), profile); err != nil {
		t.Fatal(err)
	}
}

func writeCapacityRecords(t *testing.T, runDir string) {
	t.Helper()
	protocol := defaultCapacityLoadProtocol(2)
	var output bytes.Buffer
	for _, concurrency := range protocol.ConcurrencyLevels {
		writeCapacityBatchRows(t, &output, concurrency, "warmup", 0, concurrency*2, float64(concurrency*10))
		for repetition := int64(1); repetition <= protocol.RepetitionsPerLevel; repetition++ {
			writeCapacityBatchRows(t, &output, concurrency, "measurement", repetition, 100, float64(concurrency*10))
		}
	}
	if err := os.WriteFile(filepath.Join(runDir, "records.jsonl"), output.Bytes(), 0o600); err != nil {
		t.Fatal(err)
	}
}

func writeCapacityBatchRows(
	t *testing.T,
	output *bytes.Buffer,
	concurrency int64,
	phase string,
	repetition int64,
	requests int64,
	throughput float64,
) {
	t.Helper()
	for index := int64(0); index < requests; index++ {
		attempt := fmt.Sprintf("capacity-c%d-%s%d-q%d", concurrency, phase[:1], repetition, index)
		receipt := digestBytes([]byte(attempt))
		record := executionRecordEvidence{
			SchemaVersion: SchemaVersion, ID: attempt, TrackID: "capacity", CaseID: "case-1", AttemptID: attempt,
			Status: "succeeded", Success: capacityBoolPointer(true), LatencyMS: capacityFloatPointer(20),
			InputTokens: capacityInt64Pointer(1), OutputTokens: capacityInt64Pointer(1), RuntimeCost: capacityFloatPointer(0.00001488),
			Concurrency: capacityInt64Pointer(concurrency), ThroughputRPS: capacityFloatPointer(throughput),
			LoadElapsedSeconds: capacityFloatPointer(float64(requests) / throughput),
			LoadPhase:          &phase, LoadRepetition: capacityInt64Pointer(repetition), LoadRequestIndex: capacityInt64Pointer(index),
			EvidenceKind: capacityStringPointer("capacity.closed-loop.v1"), BrokerReceipt: &receipt,
		}
		encoded, err := json.Marshal(record)
		if err != nil {
			t.Fatal(err)
		}
		output.Write(encoded)
		output.WriteByte('\n')
	}
}

func capacityRecordsAttestation() recordAttestation {
	const total = 606
	return recordAttestation{
		validated: true, Total: total, Succeeded: total,
		ByTrack: map[TrackID]recordStatusCounts{"capacity": {Succeeded: total}},
	}
}

func capacityBoolPointer(value bool) *bool { return &value }

func capacityStringPointer(value string) *string { return &value }
