package evaluationplane

import (
	"bytes"
	"encoding/json"
	"fmt"
	"path/filepath"
	"reflect"
	"strings"
	"time"
)

var requiredGateIDs = []string{"G0", "G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G9"}

var gateNames = []string{
	"Reproducibility", "Static correctness", "Hard policy", "Offline value", "Robustness / OOD",
	"Live fidelity", "Reliability / trajectory", "Cost / latency / capacity", "Shadow / canary", "Online preference",
}

var gateTracks = []TrackID{"", "", "safety", "joint", "routing", "joint", "agentic", "capacity", "", "preference"}

var gateEvidenceRefs = [][]string{
	{manifestFileName, "lineage.json", "provenance.json", publicChecksumArtifactName},
	{manifestFileName, "records.jsonl"},
	{"records.jsonl", "metric:safety.violation_rate"},
	{"metrics.json", "metric:joint.normalized_regret"},
	{"records.jsonl", "metric:routing.accuracy"},
	{"records.jsonl", "provenance.json"},
	{"records.jsonl", "metric:agentic.success_rate"},
	{"records.jsonl", "metrics.json"},
	{manifestFileName, "records.jsonl"},
	{"records.jsonl", "metric:preference.propensity_coverage"},
}

var gateDispositionMatrix = map[ChangeProfile][]string{
	"schema_adapter":    {"required", "required", "advisory", "advisory", "required", "advisory", "not_applicable", "advisory", "not_applicable", "not_applicable"},
	"recipe":            {"required", "required", "required", "required", "required", "required", "not_applicable", "required", "advisory", "not_applicable"},
	"selector":          {"required", "required", "required", "required", "required", "required", "advisory", "required", "required", "not_applicable"},
	"model_pool":        {"required", "required", "required", "required", "required", "required", "advisory", "required", "required", "not_applicable"},
	"runtime_capacity":  {"required", "required", "required", "advisory", "advisory", "required", "advisory", "required", "required", "not_applicable"},
	"agent_multimodal":  {"required", "required", "required", "required", "required", "required", "required", "required", "required", "advisory"},
	"online_adaptation": {"required", "required", "required", "required", "required", "required", "required", "required", "required", "required"},
}

func (s *Service) validateAndAnchorReport(runID string) error {
	run, err := s.store.GetRun(runID)
	if err != nil {
		return err
	}
	if run.Status != StatusRunning {
		return fmt.Errorf("%w: only a running evaluation can seal a report", ErrConflict)
	}
	data, err := s.store.ReadReport(runID)
	if err != nil {
		return err
	}
	report, err := decodeReportStrict(runID, data)
	if err != nil {
		return err
	}
	manifest, manifestBytes, err := s.readDurableManifest(runID)
	if err != nil {
		return err
	}
	if frozenFieldsErr := validateReportFrozenFields(run, manifest, report); frozenFieldsErr != nil {
		return frozenFieldsErr
	}
	checksums, err := s.validatePrivateReceipt(runID)
	if err != nil {
		return err
	}
	if bundleErr := s.validateReportBundle(runID, manifest, report, checksums); bundleErr != nil {
		return bundleErr
	}
	evidenceFiles, err := s.buildSealedEvidenceSnapshot(runID, checksums)
	if err != nil {
		return err
	}
	reportDigest, reportSize := digestAndSize(data)
	manifestDigest, _ := digestAndSize(manifestBytes)
	privateReceipt, err := readEvidenceBytes(filepath.Join(s.store.runsRoot, runID, privateChecksumArtifactName), maxStructuredArtifactBytes)
	if err != nil {
		return err
	}
	privateReceiptDigest, _ := digestAndSize(privateReceipt)
	sealedAt := time.Now().UTC()
	if err := validateReportExecutionTimestamp(run, manifest, report.Provenance.GeneratedAt, sealedAt); err != nil {
		return err
	}
	return s.store.writeReportAnchor(runID, reportAnchor{
		SchemaVersion: SchemaVersion, RunID: runID, ReportDigest: reportDigest,
		ReportSize: reportSize, ManifestDigest: manifestDigest, PrivateReceiptDigest: privateReceiptDigest,
		EvidenceFiles: evidenceFiles, CreatedAt: sealedAt,
	})
}

func validateReportExecutionTimestamp(run Run, manifest RunManifest, generatedAt, sealedAt time.Time) error {
	if run.StartedAt == nil || generatedAt.IsZero() || generatedAt.After(sealedAt) {
		return fmt.Errorf("%w: report provenance timestamp is outside the server-owned execution window", ErrInvalid)
	}
	// Replay evidence may be deterministically timestamped at manifest creation;
	// it makes no claim about a live observation window. Live evidence must be
	// generated after the server transitions the run to running.
	if manifest.Mode == ModeReplay {
		if generatedAt.Before(manifest.CreatedAt) {
			return fmt.Errorf("%w: replay report provenance predates the immutable manifest", ErrInvalid)
		}
		return nil
	}
	if manifest.Mode != ModeLive || generatedAt.Before(run.StartedAt.UTC()) {
		return fmt.Errorf("%w: report provenance timestamp is outside the server-owned execution window", ErrInvalid)
	}
	return nil
}

func (s *Service) readDurableManifest(runID string) (RunManifest, []byte, error) {
	path, err := s.store.ManifestPath(runID)
	if err != nil {
		return RunManifest{}, nil, err
	}
	manifest, raw, err := readRunManifestStrict(path)
	if err != nil {
		return RunManifest{}, nil, fmt.Errorf("%w: durable run manifest is invalid: %w", ErrInvalid, err)
	}
	if manifest.RunID != runID {
		return RunManifest{}, nil, fmt.Errorf("%w: run manifest identity mismatch", ErrInvalid)
	}
	return manifest, raw, nil
}

func validateReportFrozenFields(run Run, manifest RunManifest, report Report) error {
	if report.Run.Status != StatusCompleted || report.Run.Error != "" {
		return fmt.Errorf("%w: worker report must describe a successful completed run", ErrInvalid)
	}
	checks := []struct {
		name string
		ok   bool
	}{
		{"identity", run.ID == manifest.RunID && report.Run.ID == run.ID},
		{"mode", run.Mode == manifest.Mode && report.Run.Mode == run.Mode},
		{"target", run.TargetID == manifest.Target.ID && report.Run.TargetID == run.TargetID},
		{"change_profile", run.ChangeProfile == manifest.ChangeProfile && report.Run.ChangeProfile == run.ChangeProfile},
		{"sample_limit", run.SampleLimit == manifest.SampleLimit && report.Run.SampleLimit == run.SampleLimit},
		{"concurrency", run.Concurrency == manifest.Concurrency && report.Run.Concurrency == run.Concurrency},
		{"seed", run.Seed == manifest.Seed && report.Run.Seed == run.Seed},
		{"baseline", run.BaselineRunID == manifest.BaselineRunID && report.Run.BaselineRunID == run.BaselineRunID},
		{"evidence_level", run.EvidenceLevel == report.Run.EvidenceLevel},
		{"suites", reflect.DeepEqual(run.SuiteIDs, manifest.SuiteIDs) && reflect.DeepEqual(report.Run.SuiteIDs, run.SuiteIDs)},
		{"tracks", reflect.DeepEqual(run.TrackIDs, manifest.TrackIDs) && reflect.DeepEqual(report.Run.TrackIDs, run.TrackIDs)},
		{"created_at", run.CreatedAt.Equal(manifest.CreatedAt) && report.Run.CreatedAt.Equal(run.CreatedAt.Truncate(time.Microsecond))},
	}
	for _, check := range checks {
		if !check.ok {
			if check.name == "created_at" {
				return fmt.Errorf("%w: report created_at does not match the durable run manifest (run=%s manifest=%s report=%s)",
					ErrInvalid, run.CreatedAt.Format(time.RFC3339Nano), manifest.CreatedAt.Format(time.RFC3339Nano), report.Run.CreatedAt.Format(time.RFC3339Nano))
			}
			return fmt.Errorf("%w: report %s does not match the durable run manifest", ErrInvalid, check.name)
		}
	}
	if report.Provenance.CodeRevision != manifest.CodeRevision ||
		report.Provenance.TargetID != manifest.Target.ID || report.Provenance.Seed != manifest.Seed ||
		report.Provenance.RedactionPolicy != manifest.RedactionPolicy {
		return fmt.Errorf("%w: report provenance does not match the durable run manifest", ErrInvalid)
	}
	if manifest.GateContractVersion != GateContractVersion ||
		!validSuiteRevisionSnapshot(manifest.SuiteIDs, manifest.SuiteRevisions) ||
		!reflect.DeepEqual(report.Provenance.BenchmarkRevisions, manifest.SuiteRevisions) {
		return fmt.Errorf("%w: report benchmark or gate contract revisions do not match the durable manifest", ErrInvalid)
	}
	return nil
}

func (s *Service) validateReportBundle(runID string, manifest RunManifest, report Report, checksums map[string]string) error {
	runDir, err := s.store.checkedRunDir(runID)
	if err != nil {
		return err
	}
	records, err := validateRecordsAndFailureSummary(runDir, manifest)
	if err != nil {
		return err
	}
	if err := validateReportMetricsAndGates(runDir, report, records); err != nil {
		return err
	}
	if err := validateReportGateEvidence(report, checksums); err != nil {
		return err
	}
	if err := s.validatePublicArtifacts(runID, report, checksums); err != nil {
		return err
	}
	return validateReportProvenance(runDir, manifest, report, checksums)
}

func validateReportMetricsAndGates(runDir string, report Report, records recordAttestation) error {
	var metricFile struct {
		SchemaVersion string   `json:"schema_version"`
		Metrics       []Metric `json:"metrics"`
	}
	if err := decodeStrictEvidence(filepath.Join(runDir, "metrics.json"), &metricFile); err != nil {
		return err
	}
	var gateFile struct {
		SchemaVersion string `json:"schema_version"`
		Gates         []Gate `json:"gates"`
	}
	if err := decodeStrictEvidence(filepath.Join(runDir, "gates.json"), &gateFile); err != nil {
		return err
	}
	if metricFile.SchemaVersion != SchemaVersion || gateFile.SchemaVersion != SchemaVersion ||
		!reflect.DeepEqual(metricFile.Metrics, report.Metrics) || !reflect.DeepEqual(gateFile.Gates, report.Gates) {
		return fmt.Errorf("%w: report metrics or gates do not match their verified evidence files", ErrInvalid)
	}
	if len(report.Gates) != len(requiredGateIDs) {
		return fmt.Errorf("%w: report must contain the complete G0-G9 gate set", ErrInvalid)
	}
	metricIDs := make(map[string]bool, len(report.Metrics))
	for _, metric := range report.Metrics {
		if strings.TrimSpace(metric.ID) == "" || metricIDs[metric.ID] {
			return fmt.Errorf("%w: report contains a duplicate or blank metric id", ErrInvalid)
		}
		metricIDs[metric.ID] = true
	}
	passed, failed, unavailable := 0, 0, 0
	dispositions := gateDispositionMatrix[report.Run.ChangeProfile]
	requiredVerdict := GateVerdict("pass")
	for index, gate := range report.Gates {
		if gate.ID != requiredGateIDs[index] || gate.Name != gateNames[index] || gate.TrackID != gateTracks[index] ||
			gate.Disposition != dispositions[index] || !reflect.DeepEqual(gate.EvidenceRefs, gateEvidenceRefs[index]) ||
			gate.Owner == "" || gate.EvaluatedAt == nil || gate.SampleCount == nil || gate.Coverage == nil {
			return fmt.Errorf("%w: report gate order must be canonical G0-G9", ErrInvalid)
		}
		if gate.Disposition == "not_applicable" {
			if gate.Verdict != "not_applicable" {
				return fmt.Errorf("%w: not-applicable gate %s has an invalid verdict", ErrInvalid, gate.ID)
			}
		} else if gate.Verdict != "pass" && gate.Verdict != "fail" && gate.Verdict != "unavailable" {
			return fmt.Errorf("%w: applicable gate %s has an invalid verdict", ErrInvalid, gate.ID)
		}
		if gate.Disposition == "required" {
			if gate.Verdict == "fail" {
				requiredVerdict = "fail"
			} else if gate.Verdict == "unavailable" && requiredVerdict != "fail" {
				requiredVerdict = "unavailable"
			}
		}
		switch gate.Verdict {
		case "pass":
			passed++
		case "fail":
			failed++
		case "unavailable":
			unavailable++
		}
	}
	if report.Summary.PassedGates != passed || report.Summary.FailedGates != failed || report.Summary.UnavailableGates != unavailable {
		return fmt.Errorf("%w: report summary gate counts are inconsistent", ErrInvalid)
	}
	if report.Summary.Verdict != requiredVerdict {
		return fmt.Errorf("%w: report summary verdict does not match required gates", ErrInvalid)
	}
	if err := validateServerOwnedGateSemantics(report, records); err != nil {
		return err
	}
	if err := validatePromotionSummary(report); err != nil {
		return err
	}
	if len(report.Tracks) != len(report.Run.TrackIDs) {
		return fmt.Errorf("%w: report track coverage does not match the run", ErrInvalid)
	}
	for index, track := range report.Tracks {
		if track.TrackID != report.Run.TrackIDs[index] {
			return fmt.Errorf("%w: report track order does not match the run", ErrInvalid)
		}
		expectedGates := make([]Gate, 0)
		expectedMetrics := make([]Metric, 0)
		for _, gate := range report.Gates {
			if gate.TrackID == track.TrackID {
				expectedGates = append(expectedGates, gate)
			}
		}
		for _, metric := range report.Metrics {
			if metric.TrackID == track.TrackID {
				expectedMetrics = append(expectedMetrics, metric)
			}
		}
		if !reflect.DeepEqual(track.Gates, expectedGates) || !reflect.DeepEqual(track.Metrics, expectedMetrics) {
			return fmt.Errorf("%w: track report does not match top-level metrics and gates", ErrInvalid)
		}
	}
	return nil
}

func validatePromotionSummary(report Report) error {
	if report.Run.EvidenceLevel == "E0" {
		if report.Summary.QualityScore != nil || report.Summary.LatencyP95MS != nil ||
			report.Summary.RuntimeCost != nil || report.Summary.CapacityTCO != nil {
			return fmt.Errorf("%w: E0 reports cannot publish promotion headline metrics", ErrInvalid)
		}
		return nil
	}
	metricValue := func(ids ...string) *float64 {
		for _, id := range ids {
			for _, metric := range report.Metrics {
				if metric.ID == id && metric.Value != nil {
					value := *metric.Value
					return &value
				}
			}
		}
		return nil
	}
	quality := metricValue("joint.realized_quality", "routing.accuracy", "model_pool.oracle_quality")
	latency := metricValue("joint.latency_p95_ms", "capacity.latency_p95_ms", "routing.latency_p95_ms")
	if !reflect.DeepEqual(report.Summary.QualityScore, quality) || !reflect.DeepEqual(report.Summary.LatencyP95MS, latency) ||
		!reflect.DeepEqual(report.Summary.RuntimeCost, report.Costs.Runtime.Amount) ||
		!reflect.DeepEqual(report.Summary.CapacityTCO, report.Costs.CapacityTCO.Amount) {
		return fmt.Errorf("%w: report promotion summary does not match typed evidence", ErrInvalid)
	}
	return nil
}

func validateReportGateEvidence(report Report, checksums map[string]string) error {
	metrics := make(map[string]Metric, len(report.Metrics))
	for _, metric := range report.Metrics {
		metrics[metric.ID] = metric
	}
	for _, gate := range report.Gates {
		seen := make(map[string]bool, len(gate.EvidenceRefs))
		for _, ref := range gate.EvidenceRefs {
			if seen[ref] {
				return fmt.Errorf("%w: gate %s contains duplicate evidence references", ErrInvalid, gate.ID)
			}
			seen[ref] = true
			if metricID, ok := strings.CutPrefix(ref, "metric:"); ok {
				metric, exists := metrics[metricID]
				if (gate.Verdict == "pass" || gate.Verdict == "fail") && (!exists || metric.Value == nil) {
					return fmt.Errorf("%w: gate %s references unavailable metric evidence", ErrInvalid, gate.ID)
				}
				continue
			}
			if checksums[ref] == "" {
				return fmt.Errorf("%w: gate %s references unverified artifact evidence", ErrInvalid, gate.ID)
			}
		}
	}
	return nil
}

func decodeStrictEvidence(path string, destination any) error {
	data, err := readEvidenceBytes(path, maxStructuredArtifactBytes)
	if err != nil {
		return fmt.Errorf("read evidence file %s: %w", filepath.Base(path), err)
	}
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(destination); err != nil {
		return fmt.Errorf("decode evidence file %s: %w", filepath.Base(path), err)
	}
	return ensureJSONEOF(decoder)
}
