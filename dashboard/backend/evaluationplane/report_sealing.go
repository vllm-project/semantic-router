package evaluationplane

import (
	"encoding/json"
	"fmt"
	"time"
)

type reportSealPreparation struct {
	run                        Run
	report                     Report
	manifest                   RunManifest
	manifestBytes              []byte
	executionContract          resolvedExecutionContract
	executionAttestationDigest string
	checksums                  map[string]string
	sealedLevels               sealedEvidenceLevels
}

func (s *Service) validateAndAnchorReportDuringPublication(runID string) error {
	preparation, err := s.prepareReportSeal(runID)
	if err != nil {
		return err
	}
	return s.publishReportSeal(runID, preparation)
}

func (s *Service) prepareReportSeal(runID string) (reportSealPreparation, error) {
	run, runErr := s.store.GetRun(runID)
	if runErr != nil {
		return reportSealPreparation{}, runErr
	}
	if run.Status != StatusSealing {
		return reportSealPreparation{}, fmt.Errorf("%w: only an evaluation in the sealing phase can seal a report", ErrConflict)
	}
	data, reportReadErr := s.store.ReadReport(runID)
	if reportReadErr != nil {
		return reportSealPreparation{}, reportReadErr
	}
	if err := s.rejectConfiguredSecretBytes(data); err != nil {
		return reportSealPreparation{}, err
	}
	report, decodeErr := decodeWorkerReportStrict(runID, data)
	if decodeErr != nil {
		return reportSealPreparation{}, decodeErr
	}
	if report.Run.Status != StatusCompleted || report.Run.Error != "" {
		return reportSealPreparation{}, fmt.Errorf("%w: worker report must describe a successful completed run", ErrInvalid)
	}
	manifest, manifestBytes, err := s.readDurableManifest(runID)
	if err != nil {
		return reportSealPreparation{}, err
	}
	registry, err := s.registrySnapshot()
	if err != nil {
		return reportSealPreparation{}, err
	}
	executionContract, err := registry.executionContracts().resolve(manifest)
	if err != nil {
		return reportSealPreparation{}, err
	}
	if report.Run.Name != manifest.Name || report.Run.Description != manifest.Description {
		return reportSealPreparation{}, fmt.Errorf("%w: worker report metadata does not match the immutable run manifest", ErrInvalid)
	}
	executionAttestation, err := s.validatedExecutionAttestation(runID, manifest)
	if err != nil {
		return reportSealPreparation{}, err
	}
	executionAttestationDigest := ""
	if executionAttestation != nil {
		executionAttestationDigest = executionAttestation.Digest
	}
	checksums, err := s.validatePrivateReceipt(runID)
	if err != nil {
		return reportSealPreparation{}, err
	}
	runDir, err := s.store.checkedRunDir(runID)
	if err != nil {
		return reportSealPreparation{}, err
	}
	records, err := validateRecordsAndFailureSummary(runDir, manifest, executionContract.Executor)
	if err != nil {
		return reportSealPreparation{}, err
	}
	methodReports, err := ReduceSealedMethodReports(records.Methods)
	if err != nil {
		return reportSealPreparation{}, fmt.Errorf("%w: reduce sealed method reports: %w", ErrInvalid, err)
	}
	report.MethodReports = methodReports
	routingRecipeReport, err := reduceSealedRoutingRecipeReport(manifest, records, executionAttestation)
	if err != nil {
		return reportSealPreparation{}, err
	}
	report.RoutingRecipeReport = routingRecipeReport
	sealedLevels, err := s.validateReportBundle(
		runID, manifest, report, checksums, executionContract, executionAttestation, records,
	)
	if err != nil {
		return reportSealPreparation{}, err
	}
	// Recommendations are publication claims, so the server derives them only
	// from the metrics, gates, and evidence levels it has just verified. Worker
	// recommendation text is never copied into the attested report.
	recommendations, err := serverReportRecommendations(report, sealedLevels)
	if err != nil {
		return reportSealPreparation{}, fmt.Errorf(
			"%w: derive server report recommendations: %w",
			ErrInvalid,
			err,
		)
	}
	report.Recommendations = recommendations
	return reportSealPreparation{
		run: run, report: report, manifest: manifest, manifestBytes: manifestBytes,
		executionContract: executionContract, executionAttestationDigest: executionAttestationDigest,
		checksums: checksums, sealedLevels: sealedLevels,
	}, nil
}

func (s *Service) validatedExecutionAttestation(runID string, manifest RunManifest) (*executionAttestation, error) {
	if manifest.Mode != ModeLive {
		return nil, nil
	}
	attestation, err := s.store.readExecutionAttestationForManifest(runID, manifest)
	if err != nil {
		return nil, fmt.Errorf("%w: live report lacks its exact server execution attestation", ErrInvalid)
	}
	return &attestation, nil
}

func (s *Service) publishReportSeal(runID string, preparation reportSealPreparation) error {
	run := preparation.run
	report := preparation.report
	run.EvidenceLevel = preparation.sealedLevels.Run
	run.TrackEvidenceLevels = copyTrackEvidenceLevels(preparation.sealedLevels.ByTrack)
	completedAt := time.Now().UTC()
	canonicalizeReportRun(run, &report, completedAt)
	run.CompletedAt = &completedAt
	if err := validateReportFrozenFields(run, preparation.manifest, report); err != nil {
		return err
	}
	canonicalData, encodeErr := json.Marshal(report)
	if encodeErr != nil {
		return fmt.Errorf("encode canonical evaluation report: %w", encodeErr)
	}
	if err := s.rejectConfiguredSecretBytes(canonicalData); err != nil {
		return err
	}
	evidenceFiles, evidenceErr := s.buildSealedEvidenceSnapshot(runID, preparation.checksums)
	if evidenceErr != nil {
		return evidenceErr
	}
	manifestArtifactDigest, _ := digestAndSize(preparation.manifestBytes)
	privateReceipt, receiptErr := s.store.readPrivateChecksumReceipt(runID)
	if receiptErr != nil {
		return receiptErr
	}
	privateReceiptDigest, _ := digestAndSize(privateReceipt)
	sealedAt := time.Now().UTC()
	if err := validateReportExecutionTimestamp(run, preparation.manifest, report.Provenance.GeneratedAt, sealedAt); err != nil {
		return err
	}
	if _, err := s.store.commitSealedEvidenceLevelsWithinLifecycle(runID, preparation.sealedLevels); err != nil {
		return err
	}
	// The revision is exclusively server-owned and is published only after all
	// worker bundle and canonical report validators have succeeded.
	report.AttestationRevision = ServerAttestationRevision
	if err := s.store.WriteReport(runID, report); err != nil {
		return err
	}
	data, reportReadErr := s.store.ReadReport(runID)
	if reportReadErr != nil {
		return reportReadErr
	}
	reportDigest, reportSize := digestAndSize(data)
	return s.store.writeReportAnchor(runID, reportAnchor{
		SchemaVersion: SchemaVersion, AttestationRevision: ServerAttestationRevision,
		RunID: runID, ReportDigest: reportDigest,
		ReportSize:                 reportSize,
		ManifestSemanticDigest:     preparation.manifest.ManifestDigest,
		ManifestArtifactDigest:     manifestArtifactDigest,
		PrivateReceiptDigest:       privateReceiptDigest,
		ExecutionAttestationDigest: preparation.executionAttestationDigest,
		EvidenceFiles:              evidenceFiles, CreatedAt: sealedAt,
	})
}
