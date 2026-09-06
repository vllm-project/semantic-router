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

var requiredGateIDs = canonicalReleaseGateIDs()

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
	run, err := s.store.GetRun(runID)
	if err != nil {
		return RunManifest{}, nil, err
	}
	if err := validateRunManifestFrozenFields(run, manifest); err != nil {
		return RunManifest{}, nil, err
	}
	return manifest, raw, nil
}

func validateRunManifestFrozenFields(run Run, manifest RunManifest) error {
	checks := []struct {
		name string
		ok   bool
	}{
		{"identity", manifest.RunID == run.ID},
		{"name", manifest.Name == run.Name},
		{"description", manifest.Description == run.Description},
		{"mode", manifest.Mode == run.Mode},
		{"target", manifest.Target.ID == run.TargetID},
		{"mixture", reflect.DeepEqual(run.Mixture, catalogMixtureFromManifest(manifest.Target.Mixture))},
		{"change_profile", manifest.ChangeProfile == run.ChangeProfile},
		{"suites", reflect.DeepEqual(manifest.SuiteIDs, run.SuiteIDs)},
		{"tracks", reflect.DeepEqual(manifest.TrackIDs, run.TrackIDs)},
		{"sample_limit", manifest.SampleLimit == run.SampleLimit},
		{"concurrency", manifest.Concurrency == run.Concurrency},
		{"capacity_slo", reflect.DeepEqual(manifest.CapacitySLO, run.CapacitySLO)},
		{"capacity_load_protocol", reflect.DeepEqual(manifest.CapacityLoadProtocol, run.CapacityLoadProtocol)},
		{"seed", manifest.Seed == run.Seed},
		{"baseline", manifest.BaselineRunID == run.BaselineRunID},
		{"created_at", manifest.CreatedAt.Equal(run.CreatedAt)},
	}
	for _, check := range checks {
		if !check.ok {
			return fmt.Errorf("%w: run manifest %s does not match durable status", ErrInvalid, check.name)
		}
	}
	return nil
}

func validateReportFrozenFields(run Run, manifest RunManifest, report Report) error {
	if report.Run.Status != StatusCompleted || report.Run.Error != "" {
		return fmt.Errorf("%w: worker report must describe a successful completed run", ErrInvalid)
	}
	if run.Mode == ModeReplay && (run.Mixture != nil) != manifestUsesMoMCohortReplay(manifest) {
		return fmt.Errorf("%w: replay mixture is not authorized by the manifest executor contract", ErrInvalid)
	}
	checks := []struct {
		name string
		ok   bool
	}{
		{"identity", run.ID == manifest.RunID && report.Run.ID == run.ID},
		{"name", reportRunNameMatches(run, report.Run)},
		{"description", reportRunDescriptionMatches(run, report.Run)},
		{"client_request_id", reportRunClientRequestIDMatches(run, report.Run)},
		{"mode", run.Mode == manifest.Mode && report.Run.Mode == run.Mode},
		{"target", run.TargetID == manifest.Target.ID && report.Run.TargetID == run.TargetID},
		{"mixture", reflect.DeepEqual(run.Mixture, catalogMixtureFromManifest(manifest.Target.Mixture)) && reflect.DeepEqual(report.Run.Mixture, run.Mixture)},
		{"change_profile", run.ChangeProfile == manifest.ChangeProfile && report.Run.ChangeProfile == run.ChangeProfile},
		{"sample_limit", run.SampleLimit == manifest.SampleLimit && report.Run.SampleLimit == run.SampleLimit},
		{"concurrency", run.Concurrency == manifest.Concurrency && report.Run.Concurrency == run.Concurrency},
		{"capacity_slo", reflect.DeepEqual(run.CapacitySLO, manifest.CapacitySLO) && reflect.DeepEqual(report.Run.CapacitySLO, run.CapacitySLO)},
		{"capacity_load_protocol", reflect.DeepEqual(run.CapacityLoadProtocol, manifest.CapacityLoadProtocol) && reflect.DeepEqual(report.Run.CapacityLoadProtocol, run.CapacityLoadProtocol)},
		{"seed", run.Seed == manifest.Seed && report.Run.Seed == run.Seed},
		{"baseline", run.BaselineRunID == manifest.BaselineRunID && report.Run.BaselineRunID == run.BaselineRunID},
		{"controlled_pair", reflect.DeepEqual(report.Run.ControlledPair, run.ControlledPair)},
		{"evidence_level", run.EvidenceLevel == report.Run.EvidenceLevel},
		{"track_evidence_levels", reflect.DeepEqual(run.TrackEvidenceLevels, report.Run.TrackEvidenceLevels)},
		{"suites", reflect.DeepEqual(run.SuiteIDs, manifest.SuiteIDs) && reflect.DeepEqual(report.Run.SuiteIDs, run.SuiteIDs)},
		{"tracks", reflect.DeepEqual(run.TrackIDs, manifest.TrackIDs) && reflect.DeepEqual(report.Run.TrackIDs, run.TrackIDs)},
		{"created_at", run.CreatedAt.Equal(manifest.CreatedAt) && report.Run.CreatedAt.Equal(run.CreatedAt.Truncate(time.Microsecond))},
		{"execution_times", reportRunTimesMatch(run, report.Run)},
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
	if err := validateRoutingRecipeReportFrozenFields(run, manifest, report); err != nil {
		return err
	}
	return nil
}

func (s *Service) validateReportBundle(
	runID string,
	manifest RunManifest,
	report Report,
	checksums map[string]string,
	executionContract resolvedExecutionContract,
	executionAttestation *executionAttestation,
	records recordAttestation,
) (sealedEvidenceLevels, error) {
	runDir, err := s.store.checkedRunDir(runID)
	if err != nil {
		return sealedEvidenceLevels{}, err
	}
	qualification, err := resolveSuiteGateQualification(s.registrySource.suiteStorePath, manifest, executionContract.Executor)
	if err != nil {
		return sealedEvidenceLevels{}, err
	}
	if evidenceErr := validateReportGateEvidence(report, checksums); evidenceErr != nil {
		return sealedEvidenceLevels{}, evidenceErr
	}
	if artifactErr := s.validatePublicArtifacts(runID, manifest, report, checksums, records); artifactErr != nil {
		return sealedEvidenceLevels{}, artifactErr
	}
	capacitySLO, err := validateCapacityProfileArtifact(runDir, manifest, report, records)
	if err != nil {
		return sealedEvidenceLevels{}, err
	}
	sealedLevels, err := deriveSealedEvidenceLevels(
		runDir,
		manifest,
		records,
		qualification,
		executionContract.Executor,
		capacitySLO,
		executionAttestation,
	)
	if err != nil {
		return sealedEvidenceLevels{}, err
	}
	if err := validateReportMetricsAndGates(runDir, report, records, qualification, sealedLevels, capacitySLO); err != nil {
		return sealedEvidenceLevels{}, err
	}
	if err := validateSealedRoutingRecipeReport(report.RoutingRecipeReport, manifest, records, executionAttestation); err != nil {
		return sealedEvidenceLevels{}, err
	}
	if err := validateReportProvenance(runDir, manifest, report, checksums, executionContract); err != nil {
		return sealedEvidenceLevels{}, err
	}
	return sealedLevels, nil
}

func validateReportMetricsAndGates(
	runDir string,
	report Report,
	records recordAttestation,
	qualification suiteGateQualification,
	sealedLevels sealedEvidenceLevels,
	capacitySLO *capacitySLOAttestation,
) error {
	qualification = qualification.withSealedEvidenceLevels(sealedLevels)
	if err := validateSealedMethodReports(report.MethodReports, records.Methods); err != nil {
		return fmt.Errorf("%w: %w", ErrInvalid, err)
	}
	if err := validateReportMetricAndGateFiles(runDir, report); err != nil {
		return err
	}
	if len(report.Gates) != len(requiredGateIDs) {
		return fmt.Errorf("%w: report must contain the complete G0-G9 gate set", ErrInvalid)
	}
	if err := validateReportMetrics(report.Metrics, report.Run.TrackIDs); err != nil {
		return fmt.Errorf("%w: %w", ErrInvalid, err)
	}
	if err := validateWorkerSingleRunMetricOwnership(report.Metrics); err != nil {
		return err
	}
	if err := validateServerReducedMetrics(report, records.Metrics); err != nil {
		return err
	}
	if err := validateServerReducedMethodMetrics(report, records.Methods); err != nil {
		return err
	}
	if err := validateCapacitySLOMetric(report, capacitySLO); err != nil {
		return err
	}
	if err := validateServerReducedCosts(report.Costs, records.Costs); err != nil {
		return err
	}
	if err := validateReportGateInventory(report); err != nil {
		return err
	}
	if err := validateServerOwnedGateSemantics(report, records, qualification, capacitySLO); err != nil {
		return err
	}
	if err := validatePromotionSummary(report); err != nil {
		return err
	}
	if len(report.Tracks) != len(report.Run.TrackIDs) {
		return fmt.Errorf("%w: report track coverage does not match the run", ErrInvalid)
	}
	if err := validateTrackReportMirrors(report); err != nil {
		return err
	}
	return validateServerOwnedReportPresentation(report, records, sealedLevels)
}

type reportMetricEvidenceFile struct {
	SchemaVersion string   `json:"schema_version"`
	Metrics       []Metric `json:"metrics"`
}

type reportGateEvidenceFile struct {
	SchemaVersion string `json:"schema_version"`
	Gates         []Gate `json:"gates"`
}

func validateReportMetricAndGateFiles(runDir string, report Report) error {
	var metricFile reportMetricEvidenceFile
	if err := decodeStrictEvidence(filepath.Join(runDir, "metrics.json"), &metricFile); err != nil {
		return err
	}
	var gateFile reportGateEvidenceFile
	if err := decodeStrictEvidence(filepath.Join(runDir, "gates.json"), &gateFile); err != nil {
		return err
	}
	if metricFile.SchemaVersion != SchemaVersion || gateFile.SchemaVersion != SchemaVersion ||
		!reflect.DeepEqual(metricFile.Metrics, report.Metrics) || !reflect.DeepEqual(gateFile.Gates, report.Gates) {
		return fmt.Errorf("%w: report metrics or gates do not match their verified evidence files", ErrInvalid)
	}
	return nil
}

func validateReportGateInventory(report Report) error {
	passed, failed, unavailable := 0, 0, 0
	requiredVerdict := DecisionVerdictPass
	for index, gate := range report.Gates {
		definition, defined := releaseGateDefinitionByID(gate.ID)
		disposition, profiled := releaseProfileDisposition(report.Run.ChangeProfile, gate.ID)
		if !defined || !profiled || gate.ID != requiredGateIDs[index] || gate.Name != definition.Name ||
			gate.TrackID != definition.TrackID || gate.EvidenceLevel != definition.EvidenceLevel ||
			gate.Owner != definition.Owner || gate.Disposition != disposition ||
			!reflect.DeepEqual(gate.EvidenceRefs, definition.EvidenceRefs) ||
			gate.EvaluatedAt == nil || gate.SampleCount == nil || gate.Coverage == nil {
			return fmt.Errorf("%w: report gate order must be canonical G0-G9", ErrInvalid)
		}
		if gate.Disposition == GateDispositionNotApplicable {
			if gate.Verdict != GateVerdictNotApplicable {
				return fmt.Errorf("%w: not-applicable gate %s has an invalid verdict", ErrInvalid, gate.ID)
			}
		} else if !validGateVerdict(gate.Verdict) || gate.Verdict == GateVerdictNotApplicable {
			return fmt.Errorf("%w: applicable gate %s has an invalid verdict", ErrInvalid, gate.ID)
		}
		if gate.Disposition == GateDispositionRequired {
			if gate.Verdict == "fail" {
				requiredVerdict = DecisionVerdictFail
			} else if gate.Verdict == "unavailable" && requiredVerdict != "fail" {
				requiredVerdict = DecisionVerdictUnavailable
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
	return nil
}

func validateTrackReportMirrors(report Report) error {
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

func validateWorkerSingleRunMetricOwnership(metrics []Metric) error {
	for _, metric := range metrics {
		if isRoutingRecipeMetricID(metric.ID) {
			return fmt.Errorf(
				"%w: routing recipe metric %s is server-owned and must be read from routing_recipe_report",
				ErrInvalid,
				metric.ID,
			)
		}
		if metric.BaselineValue != nil || metric.Delta != nil {
			return fmt.Errorf("%w: single-run metric %s cannot publish worker-owned baseline_value or delta", ErrInvalid, metric.ID)
		}
	}
	return nil
}

func validateServerOwnedReportPresentation(report Report, records recordAttestation, sealedLevels sealedEvidenceLevels) error {
	if err := validateServerCoverage("report summary", report.Summary.Coverage, records.expectedSummaryCoverage()); err != nil {
		return err
	}
	if report.Run.EvidenceLevel != sealedLevels.Run {
		return fmt.Errorf("%w: run evidence level does not match server-sealed case-track evidence", ErrInvalid)
	}
	for _, track := range report.Tracks {
		if err := validateServerCoverage("track "+string(track.TrackID), track.Coverage, records.expectedTrackCoverage(track.TrackID)); err != nil {
			return err
		}
		counts := records.ByTrack[track.TrackID]
		available := counts.Succeeded + counts.Failed
		expectedStatus := "unavailable"
		expectedSummary := "No qualified evidence was produced."
		if available > 0 {
			expectedStatus = "completed"
			expectedSummary = fmt.Sprintf("Collected %d evidence records.", available)
			if counts.Failed > 0 {
				expectedSummary = fmt.Sprintf("Collected %d evidence records; %d executions failed and remain in the denominator.", available, counts.Failed)
			}
		}
		if track.Status != expectedStatus || track.Summary != expectedSummary || track.Error != "" {
			return fmt.Errorf("%w: track %s presentation does not match records", ErrInvalid, track.TrackID)
		}
		if track.EvidenceLevel != sealedLevels.ByTrack[track.TrackID] {
			return fmt.Errorf("%w: track %s evidence level does not match server-sealed case evidence", ErrInvalid, track.TrackID)
		}
	}
	return nil
}

func validateServerCoverage(label string, actual, expected Coverage) error {
	if actual.Evaluated != expected.Evaluated || actual.Total != expected.Total || actual.Unavailable != expected.Unavailable ||
		!reducedFloatsEqual(actual.Fraction, expected.Fraction) ||
		!reducedFloatsEqual(actual.ConfidenceLevel, expected.ConfidenceLevel) ||
		!reducedIntervalsEqual(actual.ConfidenceInterval, expected.ConfidenceInterval) {
		return fmt.Errorf("%w: %s coverage does not match records", ErrInvalid, label)
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
				if gate.Verdict == "unavailable" || gate.Verdict == "not_applicable" {
					continue
				}
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
	if err := rejectDuplicateJSONKeys(data); err != nil {
		return fmt.Errorf("decode evidence file %s: %w", filepath.Base(path), err)
	}
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(destination); err != nil {
		return fmt.Errorf("decode evidence file %s: %w", filepath.Base(path), err)
	}
	return ensureJSONEOF(decoder)
}
