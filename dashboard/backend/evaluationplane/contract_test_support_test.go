package evaluationplane

func float64Pointer(value float64) *float64 { return &value }

func validRoutingReportMetric() Metric {
	value := 0.8
	return Metric{
		ID: "routing.accuracy", Name: "Routing accuracy", TrackID: "routing",
		Value: &value, Unit: "fraction", Direction: "higher_is_better",
		ConfidenceInterval: []float64{0.7, 0.9}, SampleCount: 10,
		AnalysisProvenance: validMetricAnalysisProvenance(0),
	}
}

func testResolvedLineageSnapshot(manifestDigest string) map[string]any {
	return map[string]any{
		"schema_version": SchemaVersion, "manifest_digest": manifestDigest,
		"workload": map[string]any{}, "policy": map[string]any{}, "binding": map[string]any{},
		"pool": map[string]any{}, "arms": []any{}, "environment": map[string]any{},
		"fixture_ref": nil, "discovered_entrypoints": []any{}, "executors": []any{},
	}
}

func defaultCapacityLoadProtocol(concurrency int) *CapacityLoadProtocol {
	levels, err := capacityConcurrencyLevels(concurrency)
	if err != nil {
		return nil
	}
	return &CapacityLoadProtocol{
		SchemaVersion:                      SchemaVersion,
		Kind:                               capacityLoadKind,
		ConcurrencyLevels:                  levels,
		WarmupRequestMultiplier:            minimumCapacityWarmupMultiplier,
		MeasurementRequestsPerRepetition:   minimumCapacityMeasurementRequests,
		RepetitionsPerLevel:                minimumCapacityRepetitions,
		MinimumMeasurementClustersPerLevel: minimumCapacityMeasurementClusters,
		ConfidenceLevel:                    capacityLoadConfidence,
		MaxErrorRateClusterRange:           capacityMaxErrorRateClusterRange,
		MaxThroughputCV:                    maximumCapacityStabilityCV,
		MaxLatencyP95CV:                    maximumCapacityStabilityCV,
	}
}

func (s *Service) persistExecutionAttestation(
	runID string,
	transcript *brokerExecutionTranscript,
) (string, error) {
	var digest string
	err := s.store.withEvidencePublication(func() error {
		var persistErr error
		digest, persistErr = s.persistExecutionAttestationDuringPublication(runID, transcript)
		return persistErr
	})
	return digest, err
}

func workerReportFromReport(report Report) workerReport {
	return workerReport{
		SchemaVersion: report.SchemaVersion,
		Run:           workerRunStateFromRun(report.Run), Summary: report.Summary, Tracks: report.Tracks,
		Metrics: report.Metrics, Gates: report.Gates, Costs: report.Costs,
		Provenance: report.Provenance,
		Artifacts:  report.Artifacts,
	}
}

func workerRunStateFromRun(run Run) workerRunState {
	return workerRunState{
		SchemaVersion: run.SchemaVersion, ID: run.ID, ClientRequestID: run.ClientRequestID,
		Name: run.Name, Description: run.Description, Status: run.Status, Mode: run.Mode,
		EvidenceLevel: run.EvidenceLevel, TargetID: run.TargetID, Mixture: run.Mixture,
		ChangeProfile: run.ChangeProfile, SuiteIDs: run.SuiteIDs, TrackIDs: run.TrackIDs,
		SampleLimit: run.SampleLimit, Concurrency: run.Concurrency, CapacitySLO: run.CapacitySLO,
		CapacityLoadProtocol: run.CapacityLoadProtocol, Seed: run.Seed,
		BaselineRunID: run.BaselineRunID, Progress: run.Progress, CreatedAt: run.CreatedAt,
		StartedAt: run.StartedAt, CompletedAt: run.CompletedAt, Error: run.Error,
	}
}

func (s *Store) activeRunListWarnings() []runListWarning {
	_, _, warnings, _ := s.runIndex.page(nil, 0)
	return warnings
}
