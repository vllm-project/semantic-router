package evaluationplane

import (
	"errors"
	"fmt"
	"path/filepath"
	"reflect"
)

// createControlledPairBundlesAs publishes two fresh pending members behind one
// durable aggregate transaction. The generic CreateBundleAs baseline rule
// remains strict; only this path can publish a pending candidate referencing
// the pending baseline in the same validated pair.
func (s *Store) createControlledPairBundlesAs(
	actor Actor,
	pair controlledPairManifest,
	baselineManifest RunManifest,
	candidateManifest RunManifest,
) (controlledPairManifest, error) {
	if err := validateActor(actor); err != nil {
		return controlledPairManifest{}, err
	}
	if err := validateControlledPairInitialBundles(pair, baselineManifest, candidateManifest); err != nil {
		return controlledPairManifest{}, err
	}

	s.lifecycle.mu.Lock()
	defer s.lifecycle.mu.Unlock()
	s.lifecycle.evidenceMu.Lock()
	defer s.lifecycle.evidenceMu.Unlock()
	s.runIndex.coordinator.Lock()
	defer s.runIndex.coordinator.Unlock()
	s.mu.Lock()
	defer s.mu.Unlock()
	if pair.OwnerPrincipalDigest != actor.principalDigest {
		if err := s.appendLifecycleDenialsUnlocked(
			actor, "create", "invalid_request", pair.OwnerPrincipalDigest,
			pair.BaselineRunID, pair.CandidateRunID,
		); err != nil {
			return controlledPairManifest{}, err
		}
		return controlledPairManifest{}, fmt.Errorf("%w: controlled pair owner must match the creating principal", ErrForbidden)
	}
	if err := requirePrivateDirectory(s.runsRoot); err != nil {
		return controlledPairManifest{}, fmt.Errorf("validate evaluation runs directory: %w", err)
	}
	if existing, err := s.readControlledPair(pair.PairID); err == nil {
		return s.reconcileControlledPairCreateUnlocked(actor, existing, pair)
	} else if !errors.Is(err, ErrNotFound) {
		return controlledPairManifest{}, err
	}
	for _, sourceID := range []string{pair.BaselineSourceRunID, pair.CandidateSourceRunID} {
		sourceRun, sourceErr := s.getRunPhysical(sourceID)
		if sourceErr != nil {
			return controlledPairManifest{}, sourceErr
		}
		sourceLifecycle, sourceErr := s.readRunLifecycle(sourceRun)
		if sourceErr != nil {
			return controlledPairManifest{}, sourceErr
		}
		if sourceLifecycle.OwnerPrincipalDigest != actor.principalDigest && !actor.administrator {
			if err := s.appendLifecycleDenialsUnlocked(
				actor, "create", "not_owner", sourceLifecycle.OwnerPrincipalDigest,
				pair.BaselineRunID, pair.CandidateRunID,
			); err != nil {
				return controlledPairManifest{}, err
			}
			return controlledPairManifest{}, fmt.Errorf("%w: controlled pair source belongs to another principal", ErrForbidden)
		}
	}
	if err := s.validateControlledPairAuthoritativeIdentityUnlocked(pair, baselineManifest, candidateManifest); err != nil {
		return controlledPairManifest{}, err
	}
	aggregateReservationBytes, err := controlledPairIntentReservationBytes(pair)
	if err != nil {
		return controlledPairManifest{}, err
	}
	return s.publishControlledPairInitialBundlesUnlocked(
		actor, pair, baselineManifest, candidateManifest, aggregateReservationBytes,
	)
}

func (s *Store) publishControlledPairInitialBundlesUnlocked(
	actor Actor,
	pair controlledPairManifest,
	baselineManifest RunManifest,
	candidateManifest RunManifest,
	aggregateReservationBytes int64,
) (controlledPairManifest, error) {
	baselineMembership := controlledPairMembership{
		SchemaVersion: SchemaVersion, PairID: pair.PairID,
		RunID: pair.BaselineRunID, Role: controlledPairRoleBaseline,
	}
	candidateMembership := controlledPairMembership{
		SchemaVersion: SchemaVersion, PairID: pair.PairID,
		RunID: pair.CandidateRunID, Role: controlledPairRoleCandidate,
	}
	items, err := s.prepareInitialBundlePublicationUnlocked(actor, []initialBundleSpec{
		{
			run: pair.BaselineRun, manifest: baselineManifest,
			lifecycle: newRunLifecycle(pair.BaselineRun, actor),
			decorate:  func(path string) error { return writeControlledPairMembership(path, baselineMembership) },
		},
		{
			run: pair.CandidateRun, manifest: candidateManifest,
			lifecycle: newRunLifecycle(pair.CandidateRun, actor),
			decorate:  func(path string) error { return writeControlledPairMembership(path, candidateMembership) },
		},
	}, aggregateReservationBytes)
	if err != nil {
		return controlledPairManifest{}, err
	}
	intentDurable := false
	defer func() {
		if !intentDurable {
			cleanupStagedInitialBundles(items)
		}
	}()

	pair.State = controlledPairStatePublishing
	pair.BaselineStageName = filepath.Base(items[0].stagedDir)
	pair.CandidateStageName = filepath.Base(items[1].stagedDir)
	if err := s.writeControlledPairDurably(pair); err != nil {
		// A post-rename directory-sync error is still an error, but the visible
		// intent remains the authoritative recovery identity. Preserve its
		// staged members so a retry or restart can deterministically resume it.
		if visible, readErr := s.readControlledPair(pair.PairID); readErr == nil && reflect.DeepEqual(visible, pair) {
			intentDurable = true
		}
		return controlledPairManifest{}, err
	}
	intentDurable = true
	if err := publishStagedInitialBundle(items[0], s.pairPersistence.Rename); err != nil {
		return controlledPairManifest{}, err
	}
	if err := publishStagedInitialBundle(items[1], s.pairPersistence.Rename); err != nil {
		return controlledPairManifest{}, err
	}
	if err := s.pairPersistence.SyncDirectory(s.runsRoot, "controlled pair members"); err != nil {
		return controlledPairManifest{}, err
	}
	pair.State = controlledPairStatePending
	pair.BaselineStageName, pair.CandidateStageName = "", ""
	if err := s.writeControlledPairDurably(pair); err != nil {
		return controlledPairManifest{}, err
	}
	s.runIndex.upsertOwnedBatch(
		[]Run{pair.BaselineRun, pair.CandidateRun},
		pair.OwnerPrincipalDigest,
		map[string]uint64{pair.BaselineRunID: 1, pair.CandidateRunID: 1},
	)
	return pair, nil
}

func (s *Store) validateControlledPairAuthoritativeIdentityUnlocked(
	pair controlledPairManifest,
	baselineManifest RunManifest,
	candidateManifest RunManifest,
) error {
	if baselineManifest.ManifestDigest != pair.BaselineMemberManifestDigest ||
		candidateManifest.ManifestDigest != pair.CandidateMemberManifestDigest {
		return fmt.Errorf("%w: controlled pair member manifest binding is invalid", ErrInvalid)
	}
	var sources [2]controlledPairSource
	for index, source := range []struct {
		runID, semanticDigest, artifactDigest, anchorDigest, attestationDigest string
	}{
		{
			pair.BaselineSourceRunID, pair.BaselineSourceManifestSemanticDigest,
			pair.BaselineSourceManifestArtifactDigest, pair.BaselineSourceAnchorDigest,
			pair.BaselineSourceAttestationDigest,
		},
		{
			pair.CandidateSourceRunID, pair.CandidateSourceManifestSemanticDigest,
			pair.CandidateSourceManifestArtifactDigest, pair.CandidateSourceAnchorDigest,
			pair.CandidateSourceAttestationDigest,
		},
	} {
		run, runErr := s.getRunPhysical(source.runID)
		if runErr != nil || run.Status != StatusCompleted {
			return fmt.Errorf("%w: controlled pair source is not sealed completed evidence", ErrInvalid)
		}
		manifestPath := filepath.Join(s.runsRoot, source.runID, manifestFileName)
		manifestBytes, manifestBytesErr := readEvidenceBytes(manifestPath, maxStructuredArtifactBytes)
		if manifestBytesErr != nil {
			return manifestBytesErr
		}
		var manifest RunManifest
		if err := readJSON(manifestPath, &manifest); err != nil {
			return err
		}
		if err := validateRunManifestContract(manifest); err != nil {
			return err
		}
		if err := validateRunManifestFrozenFields(run, manifest); err != nil {
			return err
		}
		artifactDigest, _ := digestAndSize(manifestBytes)
		if manifest.ManifestDigest != source.semanticDigest || artifactDigest != source.artifactDigest {
			return fmt.Errorf("%w: controlled pair source manifest binding changed", ErrInvalid)
		}
		anchorBytes, anchorBytesErr := readEvidenceBytes(
			filepath.Join(s.runsRoot, source.runID, reportAnchorFileName), maxReportAnchorBytes,
		)
		if anchorBytesErr != nil {
			return anchorBytesErr
		}
		anchorDigest, _ := digestAndSize(anchorBytes)
		anchor, anchorErr := s.readReportAnchor(source.runID)
		if anchorErr != nil {
			return anchorErr
		}
		attestation, attestationErr := s.readExecutionAttestationForManifest(source.runID, manifest)
		if attestationErr != nil || anchorDigest != source.anchorDigest ||
			anchor.ExecutionAttestationDigest != source.attestationDigest ||
			attestation.Digest != source.attestationDigest {
			return fmt.Errorf("%w: controlled pair source seal binding changed", ErrInvalid)
		}
		reportBytes, reportBytesErr := s.ReadReport(source.runID)
		if reportBytesErr != nil {
			return reportBytesErr
		}
		report, reportErr := decodeReportStrict(source.runID, reportBytes)
		if reportErr != nil {
			return reportErr
		}
		if err := validateReportFrozenFields(run, manifest, report); err != nil {
			return err
		}
		if err := s.verifyReportAnchorBundle(
			source.runID, reportBytes, report.AttestationRevision, manifest, manifestBytes,
		); err != nil {
			return err
		}
		sources[index] = controlledPairSource{
			run: run, manifest: manifest, report: report,
			manifestArtifactDigest: artifactDigest, anchorDigest: anchorDigest,
			attestationDigest: attestation.Digest,
		}
	}
	cohortDigest, treatmentDigest, err := controlledPairCohortTreatmentDigests(sources[0], sources[1])
	if err != nil {
		return err
	}
	if cohortDigest != pair.CohortDigest || treatmentDigest != pair.TreatmentDigest {
		return fmt.Errorf("%w: controlled pair cohort or treatment binding changed", ErrInvalid)
	}
	return nil
}

func validateControlledPairInitialBundles(
	pair controlledPairManifest,
	baselineManifest RunManifest,
	candidateManifest RunManifest,
) error {
	if pair.State != controlledPairStatePending {
		return fmt.Errorf("%w: new controlled pair must begin pending", ErrInvalid)
	}
	if err := validateControlledPairManifest(pair); err != nil {
		return err
	}
	if err := validateInitialRunBundle(pair.BaselineRun, baselineManifest); err != nil {
		return fmt.Errorf("controlled pair baseline: %w", err)
	}
	if err := validateInitialRunBundle(pair.CandidateRun, candidateManifest); err != nil {
		return fmt.Errorf("controlled pair candidate: %w", err)
	}
	if candidateManifest.BaselineRunID != pair.BaselineRunID || baselineManifest.BaselineRunID != "" {
		return fmt.Errorf("%w: controlled pair manifests do not form one causal reference", ErrInvalid)
	}
	return nil
}
