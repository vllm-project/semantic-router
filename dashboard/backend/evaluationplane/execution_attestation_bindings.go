package evaluationplane

import (
	"fmt"
	"time"
)

type brokerAttestationEntryIndex struct {
	byReceipt       map[string]int
	modelDiscovery  int
	controlledCount int
}

func validateBrokerRecordBindings(
	manifest RunManifest,
	entries []executionAttestationEntry,
	records []executionRecordEvidence,
	cases visibleCaseSet,
	grading map[string]gradingCaseEvidence,
	transcriptStartedAt time.Time,
	transcriptCompletedAt time.Time,
) ([]executionAttestationEntry, error) {
	index, err := indexBrokerAttestationEntries(
		manifest, entries, transcriptStartedAt, transcriptCompletedAt,
	)
	if err != nil {
		return nil, err
	}
	recordsByReceipt, err := indexBrokerRecords(entries, records, index.byReceipt)
	if err != nil {
		return nil, err
	}
	if err := validateMixtureRecordDensity(manifest, records, cases); err != nil {
		return nil, err
	}
	poolOracleArmIDs := serverPoolOracleArmIDs(manifest, entries, recordsByReceipt, cases, grading)
	return validateBoundBrokerEntries(manifest, entries, recordsByReceipt, cases, grading, poolOracleArmIDs)
}

func indexBrokerAttestationEntries(
	manifest RunManifest,
	entries []executionAttestationEntry,
	transcriptStartedAt time.Time,
	transcriptCompletedAt time.Time,
) (brokerAttestationEntryIndex, error) {
	result := brokerAttestationEntryIndex{byReceipt: make(map[string]int, len(entries))}
	controlledSession := ""
	controlledRole := ""
	expectedRequestID := uint64(1)
	for index := range entries {
		entry := &entries[index]
		if entry.RequestID != expectedRequestID || entry.Headers == nil ||
			!digestPattern.MatchString(entry.RequestDigest) || !digestPattern.MatchString(entry.ResponseDigest) ||
			!digestPattern.MatchString(entry.BrokerReceipt) || entry.LatencyMicroseconds < 0 {
			return brokerAttestationEntryIndex{}, fmt.Errorf("%w: broker transcript entry %d is invalid", ErrInvalid, index+1)
		}
		if entry.FetchedAt != nil && (entry.FetchedAt.Before(transcriptStartedAt) || entry.FetchedAt.After(transcriptCompletedAt)) {
			return brokerAttestationEntryIndex{}, fmt.Errorf("%w: broker transcript fetched_at lies outside its server window", ErrInvalid)
		}
		expectedRequestID++
		receipt, err := brokerEntryReceipt(*entry)
		if err != nil || receipt != entry.BrokerReceipt || result.byReceipt[receipt] != 0 {
			return brokerAttestationEntryIndex{}, fmt.Errorf("%w: broker transcript receipt is invalid or duplicated", ErrInvalid)
		}
		result.byReceipt[receipt] = index + 1
		if entry.Operation == workerBrokerListModels {
			result.modelDiscovery++
			if entry.TrackID != "" || entry.CaseID != "" || entry.AttemptID != "" ||
				!entry.UpstreamAttempted || !entry.Success {
				return brokerAttestationEntryIndex{}, fmt.Errorf("%w: runtime model discovery was not attested", ErrInvalid)
			}
			continue
		}
		if err := indexControlledPairEntry(
			manifest, *entry, transcriptStartedAt, transcriptCompletedAt,
			&result, &controlledSession, &controlledRole, index,
		); err != nil {
			return brokerAttestationEntryIndex{}, err
		}
		if (!entry.UpstreamAttempted && !unattemptedRoutingDecisionUnavailable(*entry)) ||
			!containsTrack(manifest.TrackIDs, entry.TrackID) ||
			!evidenceIDPattern.MatchString(entry.CaseID) || !evidenceIDPattern.MatchString(entry.AttemptID) {
			return brokerAttestationEntryIndex{}, fmt.Errorf("%w: broker execution entry is not a bounded manifest operation", ErrInvalid)
		}
		if err := validateBrokerRoutingRecipeDecision(manifest.Target.Mixture, *entry); err != nil {
			return brokerAttestationEntryIndex{}, fmt.Errorf("%w: broker execution entry %d: %w", ErrInvalid, index+1, err)
		}
		if err := validateBrokerMixtureBinding(manifest.Target.Mixture, *entry); err != nil {
			return brokerAttestationEntryIndex{}, fmt.Errorf("%w: broker execution entry %d: %w", ErrInvalid, index+1, err)
		}
	}
	if result.modelDiscovery != 1 {
		return brokerAttestationEntryIndex{}, fmt.Errorf("%w: live execution requires exactly one successful model discovery", ErrInvalid)
	}
	if result.controlledCount != 0 && result.controlledCount != len(entries)-result.modelDiscovery {
		return brokerAttestationEntryIndex{}, fmt.Errorf("%w: controlled pair provenance must cover every evidence observation", ErrInvalid)
	}
	return result, nil
}

func indexControlledPairEntry(
	manifest RunManifest,
	entry executionAttestationEntry,
	transcriptStartedAt time.Time,
	transcriptCompletedAt time.Time,
	result *brokerAttestationEntryIndex,
	controlledSession *string,
	controlledRole *string,
	index int,
) error {
	if entry.ControlledPair == nil {
		return nil
	}
	if err := validateControlledPairObservation(
		manifest, entry, transcriptStartedAt, transcriptCompletedAt,
	); err != nil {
		return fmt.Errorf("%w: broker execution entry %d controlled pairing: %w", ErrInvalid, index+1, err)
	}
	result.controlledCount++
	if *controlledSession == "" {
		*controlledSession = entry.ControlledPair.SessionID
		*controlledRole = entry.ControlledPair.Role
		return nil
	}
	if entry.ControlledPair.SessionID != *controlledSession || entry.ControlledPair.Role != *controlledRole {
		return fmt.Errorf("%w: broker transcript mixes controlled pair sessions or roles", ErrInvalid)
	}
	return nil
}

func indexBrokerRecords(
	entries []executionAttestationEntry,
	records []executionRecordEvidence,
	byReceipt map[string]int,
) (map[string][]executionRecordEvidence, error) {
	recordsByReceipt := make(map[string][]executionRecordEvidence, len(entries))
	for _, record := range records {
		if record.BrokerReceipt == nil {
			return nil, fmt.Errorf("%w: live record %q omits its broker receipt", ErrInvalid, record.ID)
		}
		entryIndex := byReceipt[*record.BrokerReceipt]
		if entryIndex == 0 {
			return nil, fmt.Errorf("%w: live record broker receipt is absent", ErrInvalid)
		}
		if entries[entryIndex-1].Operation == workerBrokerListModels {
			return nil, fmt.Errorf("%w: live record cannot bind model discovery", ErrInvalid)
		}
		recordsByReceipt[*record.BrokerReceipt] = append(recordsByReceipt[*record.BrokerReceipt], record)
	}
	return recordsByReceipt, nil
}

func validateBoundBrokerEntries(
	manifest RunManifest,
	entries []executionAttestationEntry,
	recordsByReceipt map[string][]executionRecordEvidence,
	cases visibleCaseSet,
	grading map[string]gradingCaseEvidence,
	poolOracleArmIDs map[string]map[string]struct{},
) ([]executionAttestationEntry, error) {
	var frozenArms []ModelArm
	if manifest.Target.Mixture != nil {
		frozenArms = manifest.Target.Mixture.ModelArms
	}
	for index := range entries {
		entry := &entries[index]
		if entry.Operation == workerBrokerListModels {
			entry.responsePayload = nil
			continue
		}
		if err := validateBoundBrokerEntry(
			manifest, entry, recordsByReceipt[entry.BrokerReceipt], cases, grading, frozenArms, poolOracleArmIDs,
		); err != nil {
			return nil, err
		}
	}
	return entries, nil
}

func validateBoundBrokerEntry(
	manifest RunManifest,
	entry *executionAttestationEntry,
	boundRecords []executionRecordEvidence,
	cases visibleCaseSet,
	grading map[string]gradingCaseEvidence,
	frozenArms []ModelArm,
	poolOracleArmIDs map[string]map[string]struct{},
) error {
	if len(boundRecords) == 0 {
		return fmt.Errorf("%w: broker operation has no exact evidence record", ErrInvalid)
	}
	if isMethodLedgerOperation(entry.Operation) {
		if err := validateMethodLedgerBrokerBinding(*entry, boundRecords, manifest); err != nil {
			return fmt.Errorf("%w: %s: %w", ErrInvalid, entry.Operation, err)
		}
		entry.responsePayload = nil
		return nil
	}
	if len(boundRecords) != 1 {
		return fmt.Errorf("%w: ordinary broker receipt is reused by multiple records", ErrInvalid)
	}
	record := boundRecords[0]
	if err := validateBrokerRecord(
		*entry, record, cases, grading[record.CaseID], frozenArms, poolOracleArmIDs[record.CaseID],
		manifest.Seed,
	); err != nil {
		return fmt.Errorf("%w: live record %q: %w", ErrInvalid, record.ID, err)
	}
	entry.Quality = serverObservedQuality(
		*entry, record.TrackID, grading[record.CaseID], poolOracleArmIDs[record.CaseID],
	)
	entry.responsePayload = nil
	return nil
}
