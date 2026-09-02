package evaluationplane

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strconv"
	"strings"
	"time"
)

const (
	lifecycleAuditCheckpointSchemaVersion = "evaluation-lifecycle-audit-checkpoint.v2"
	lifecycleAuditCheckpointFileName      = "checkpoint.json"
	lifecycleBindingDirectoryName         = "bindings"
	lifecycleBindingFileSuffix            = ".json"
	lifecycleNotFoundDedupeWindow         = time.Minute
	maxLifecycleNotFoundDenialWindows     = 32
	lifecycleAuditAppendReserveBytes      = int64(2 * 1024)
)

type lifecycleAuditDenialWindow struct {
	ActorDigest  string    `json:"actor_digest"`
	ResourceKind string    `json:"resource_kind"`
	Action       string    `json:"action"`
	Timestamp    time.Time `json:"timestamp"`
}

type lifecycleAuditCheckpoint struct {
	SchemaVersion            string                       `json:"schema_version"`
	Sequence                 uint64                       `json:"sequence"`
	SegmentStartSequence     uint64                       `json:"segment_start_sequence"`
	SegmentPreviousDigest    string                       `json:"segment_previous_digest,omitempty"`
	Timestamp                time.Time                    `json:"timestamp"`
	HeadDigest               string                       `json:"head_digest"`
	PreviousCheckpointDigest string                       `json:"previous_checkpoint_digest,omitempty"`
	NotFoundDenials          []lifecycleAuditDenialWindow `json:"not_found_denials,omitempty"`
	Digest                   string                       `json:"digest"`
}

type lifecycleCheckpointCleaner interface {
	Remove(path string) error
	Sync(directory, purpose string) error
}

type atomicLifecycleCheckpointCleaner struct{}

func (atomicLifecycleCheckpointCleaner) Remove(path string) error {
	return os.Remove(path)
}

func (atomicLifecycleCheckpointCleaner) Sync(directory, purpose string) error {
	return syncEvaluationDirectory(directory, purpose)
}

func lifecycleCheckpointDigest(checkpoint lifecycleAuditCheckpoint) string {
	checkpoint.Digest = ""
	encoded, err := json.Marshal(checkpoint)
	if err != nil {
		panic(err)
	}
	return digestBytes(encoded)
}

func validateLifecycleAuditCheckpoint(checkpoint lifecycleAuditCheckpoint) error {
	if checkpoint.SchemaVersion != lifecycleAuditCheckpointSchemaVersion || checkpoint.Sequence == 0 ||
		len(checkpoint.NotFoundDenials) > maxLifecycleNotFoundDenialWindows ||
		checkpoint.SegmentStartSequence == 0 || checkpoint.SegmentStartSequence > checkpoint.Sequence ||
		(checkpoint.SegmentStartSequence == 1) != (checkpoint.SegmentPreviousDigest == "") ||
		(checkpoint.SegmentPreviousDigest != "" && !digestPattern.MatchString(checkpoint.SegmentPreviousDigest)) ||
		checkpoint.Timestamp.IsZero() || !digestPattern.MatchString(checkpoint.HeadDigest) ||
		(checkpoint.PreviousCheckpointDigest != "" && !digestPattern.MatchString(checkpoint.PreviousCheckpointDigest)) ||
		checkpoint.Digest != lifecycleCheckpointDigest(checkpoint) {
		return fmt.Errorf("%w: lifecycle audit checkpoint is invalid", ErrInvalid)
	}
	previousKey := ""
	for _, denial := range checkpoint.NotFoundDenials {
		key := lifecycleNotFoundDenialKey(denial.ActorDigest, denial.ResourceKind, denial.Action)
		if !digestPattern.MatchString(denial.ActorDigest) || !validLifecycleAction(denial.Action) ||
			!validLifecycleResource(
				denial.ResourceKind, "", denial.Action, "denied", "not_found",
			) || denial.Timestamp.IsZero() ||
			denial.Timestamp.After(checkpoint.Timestamp) || key <= previousKey {
			return fmt.Errorf("%w: lifecycle audit checkpoint denial window is invalid", ErrInvalid)
		}
		previousKey = key
	}
	return nil
}

func (s *Store) loadLifecycleAuditCheckpointUnlocked() (lifecycleAuditCheckpoint, int64, error) {
	path := filepath.Join(s.lifecycleAuditRoot, lifecycleAuditCheckpointFileName)
	var checkpoint lifecycleAuditCheckpoint
	if err := readJSON(path, &checkpoint); err != nil {
		if errors.Is(err, ErrNotFound) {
			return lifecycleAuditCheckpoint{}, 0, nil
		}
		return lifecycleAuditCheckpoint{}, 0, fmt.Errorf("read lifecycle audit checkpoint: %w", err)
	}
	info, err := os.Lstat(path)
	if err != nil || !info.Mode().IsRegular() || info.Mode()&os.ModeSymlink != 0 || info.Mode().Perm() != 0o600 ||
		info.Size() > s.maxLifecycleCheckpointBytes() {
		return lifecycleAuditCheckpoint{}, 0, fmt.Errorf("%w: lifecycle audit checkpoint is not a private bounded file", ErrInvalid)
	}
	if err := validateLifecycleAuditCheckpoint(checkpoint); err != nil {
		return lifecycleAuditCheckpoint{}, 0, err
	}
	return checkpoint, info.Size(), nil
}

func (s *Store) loadLifecycleCreationBindingsUnlocked() (map[string]lifecycleAuditRecord, error) {
	entries, err := os.ReadDir(s.lifecycleBindingRoot)
	if err != nil {
		return nil, fmt.Errorf("list lifecycle creation bindings: %w", err)
	}
	bindings := make(map[string]lifecycleAuditRecord, len(entries))
	for _, entry := range entries {
		if entry.IsDir() || entry.Type()&os.ModeSymlink != 0 || !strings.HasSuffix(entry.Name(), lifecycleBindingFileSuffix) {
			return nil, fmt.Errorf("%w: lifecycle creation binding entry is invalid", ErrInvalid)
		}
		resource, validName := parseLifecycleBindingFileName(entry.Name())
		if !validName {
			return nil, fmt.Errorf("%w: lifecycle creation binding name is invalid", ErrInvalid)
		}
		path := filepath.Join(s.lifecycleBindingRoot, entry.Name())
		info, statErr := os.Lstat(path)
		if statErr != nil || !info.Mode().IsRegular() || info.Mode().Perm() != 0o600 || info.Size() > maxLifecycleRecordSize {
			return nil, fmt.Errorf("%w: lifecycle creation binding is not a private bounded file", ErrInvalid)
		}
		var record lifecycleAuditRecord
		if readErr := readJSON(path, &record); readErr != nil || validateLifecycleAuditRecord(record) != nil ||
			record.Action != "create" || record.Decision != "allowed" ||
			record.ResourceKind != resource.Kind || record.ResourceID != resource.ID {
			return nil, fmt.Errorf("%w: lifecycle creation binding is invalid", ErrInvalid)
		}
		if _, duplicate := bindings[record.Digest]; duplicate {
			return nil, fmt.Errorf("%w: lifecycle creation binding digest is duplicated", ErrInvalid)
		}
		bindings[record.Digest] = record
	}
	return bindings, nil
}

func (s *Store) checkpointLifecycleAuditUnlocked(now time.Time) error {
	if err := s.finishLifecycleCheckpointCleanupUnlocked(); err != nil {
		return err
	}
	if s.lifecycle.activeCount == 0 && s.lifecycle.sequence == 0 {
		return nil
	}
	desiredBindings, checkpointErr := s.lifecycleCreationBindingsForCheckpointUnlocked()
	if checkpointErr != nil {
		return checkpointErr
	}
	if err := s.publishLifecycleCreationBindingsUnlocked(desiredBindings); err != nil {
		return err
	}
	segmentStart, segmentRoot, checkpointErr := s.lifecycleCheckpointSegmentBoundaryUnlocked()
	if checkpointErr != nil {
		return checkpointErr
	}
	checkpointTime := now.UTC().Truncate(time.Microsecond)
	denialWindows := s.activeNotFoundDenialWindows(checkpointTime)
	for _, window := range denialWindows {
		if window.Timestamp.After(checkpointTime) {
			checkpointTime = window.Timestamp
		}
	}
	checkpoint := lifecycleAuditCheckpoint{
		SchemaVersion:            lifecycleAuditCheckpointSchemaVersion,
		Sequence:                 s.lifecycle.sequence,
		SegmentStartSequence:     segmentStart,
		SegmentPreviousDigest:    segmentRoot,
		Timestamp:                checkpointTime,
		HeadDigest:               s.lifecycle.headDigest,
		PreviousCheckpointDigest: s.lifecycle.checkpointDigest,
		NotFoundDenials:          denialWindows,
	}
	checkpoint.Digest = lifecycleCheckpointDigest(checkpoint)
	if err := validateLifecycleAuditCheckpoint(checkpoint); err != nil {
		return err
	}
	encoded, checkpointErr := json.MarshalIndent(checkpoint, "", "  ")
	if checkpointErr != nil || int64(len(encoded)+1) > s.maxLifecycleCheckpointBytes() {
		return fmt.Errorf("%w: lifecycle audit checkpoint exceeds its durable envelope", ErrQuota)
	}
	path := filepath.Join(s.lifecycleAuditRoot, lifecycleAuditCheckpointFileName)
	if err := writeLifecycleAuditCheckpointAtomic(path, encoded); err != nil {
		var visible lifecycleAuditCheckpoint
		if readErr := readJSON(path, &visible); readErr != nil || !reflect.DeepEqual(visible, checkpoint) {
			return err
		}
		s.installLifecycleCheckpointUnlocked(checkpoint, int64(len(encoded)+1), desiredBindings)
		s.lifecycle.checkpointDurabilityPending = true
		return err
	}
	s.installLifecycleCheckpointUnlocked(checkpoint, int64(len(encoded)+1), desiredBindings)
	s.lifecycle.checkpointDurabilityPending = false
	return s.finishLifecycleCheckpointCleanupUnlocked()
}

func writeLifecycleAuditCheckpointAtomic(path string, encoded []byte) error {
	encoded = append(encoded, '\n')
	temporary, err := os.CreateTemp(filepath.Dir(path), lifecycleAuditTempPrefix+"*")
	if err != nil {
		return fmt.Errorf("stage lifecycle audit checkpoint: %w", err)
	}
	temporaryPath := temporary.Name()
	defer func() { _ = os.Remove(temporaryPath) }()
	if err := temporary.Chmod(0o600); err != nil {
		_ = temporary.Close()
		return fmt.Errorf("protect lifecycle audit checkpoint: %w", err)
	}
	if _, err := temporary.Write(encoded); err != nil {
		_ = temporary.Close()
		return fmt.Errorf("write lifecycle audit checkpoint: %w", err)
	}
	if err := temporary.Sync(); err != nil {
		_ = temporary.Close()
		return fmt.Errorf("sync lifecycle audit checkpoint: %w", err)
	}
	if err := temporary.Close(); err != nil {
		return fmt.Errorf("close lifecycle audit checkpoint: %w", err)
	}
	if err := os.Rename(temporaryPath, path); err != nil {
		return fmt.Errorf("publish lifecycle audit checkpoint: %w", err)
	}
	return syncEvaluationDirectory(filepath.Dir(path), "lifecycle audit checkpoint")
}

func (s *Store) lifecycleCreationBindingsForCheckpointUnlocked() (map[lifecycleResourceRef]lifecycleAuditRecord, error) {
	desired := make(map[lifecycleResourceRef]lifecycleAuditRecord)
	durable := make(map[lifecycleResourceRef]bool)
	for _, run := range s.runIndex.allRuns() {
		lifecycle, err := s.readRunLifecycle(run)
		if err != nil {
			return nil, fmt.Errorf("checkpoint lifecycle audit run binding: %w", err)
		}
		record, exists := s.lifecycle.records[lifecycle.CreationAuditDigest]
		if !exists {
			record, exists = s.lifecycle.creationBindings[lifecycle.CreationAuditDigest]
		}
		resource := lifecycleResourceRef{Kind: lifecycleResourceRun, ID: run.ID}
		if !exists || !lifecycleCreationRecordMatches(record, resource, lifecycle.OwnerPrincipalDigest) {
			return nil, fmt.Errorf("%w: lifecycle checkpoint cannot verify run creation", ErrInvalid)
		}
		desired[resource] = record
		durable[resource] = true
	}
	campaigns, err := s.loadStoredCampaignsUnlocked()
	if err != nil {
		return nil, fmt.Errorf("checkpoint campaign lifecycle bindings: %w", err)
	}
	for _, campaign := range campaigns {
		lifecycle, lifecycleErr := s.readCampaignLifecycle(campaign)
		if lifecycleErr != nil {
			return nil, fmt.Errorf("checkpoint campaign lifecycle binding: %w", lifecycleErr)
		}
		record, exists := s.lifecycle.records[lifecycle.CreationAuditDigest]
		if !exists {
			record, exists = s.lifecycle.creationBindings[lifecycle.CreationAuditDigest]
		}
		resource := lifecycleResourceRef{Kind: lifecycleResourceCampaign, ID: campaign.ID}
		if !exists || !lifecycleCreationRecordMatches(record, resource, lifecycle.OwnerPrincipalDigest) {
			return nil, fmt.Errorf("%w: lifecycle checkpoint cannot verify campaign creation", ErrInvalid)
		}
		desired[resource] = record
		durable[resource] = true
	}
	for _, record := range s.lifecycle.records {
		if record.Action == "create" && record.Decision == "allowed" {
			resource := lifecycleResourceRef{Kind: record.ResourceKind, ID: record.ResourceID}
			current, exists := desired[resource]
			if !durable[resource] && (!exists || current.Sequence < record.Sequence) {
				desired[resource] = record
			}
		}
	}
	return desired, nil
}

func (s *Store) publishLifecycleCreationBindingsUnlocked(
	desired map[lifecycleResourceRef]lifecycleAuditRecord,
) error {
	if err := s.resolvePendingLifecycleCreationBindingUnlocked(desired); err != nil {
		return err
	}
	for resource, record := range desired {
		path := filepath.Join(s.lifecycleBindingRoot, lifecycleBindingFileName(resource))
		var existing lifecycleAuditRecord
		if readErr := readJSON(path, &existing); readErr == nil {
			if existing != record {
				known, exists := s.lifecycle.creationBindings[existing.Digest]
				if !exists || known != existing {
					return fmt.Errorf("%w: lifecycle creation binding changed", ErrInvalid)
				}
				if err := os.Remove(path); err != nil {
					return fmt.Errorf("replace superseded lifecycle creation binding: %w", err)
				}
				if err := syncEvaluationDirectory(s.lifecycleBindingRoot, "superseded lifecycle creation binding"); err != nil {
					return err
				}
			} else {
				continue
			}
		} else if !errors.Is(readErr, ErrNotFound) {
			return readErr
		}
		s.lifecycle.pendingLifecycleBindings[resource] = record
		if err := s.lifecycleAuditWriter.WriteExclusive(path, record); err != nil {
			var visible lifecycleAuditRecord
			if readErr := readJSON(path, &visible); errors.Is(readErr, ErrNotFound) {
				delete(s.lifecycle.pendingLifecycleBindings, resource)
			}
			return fmt.Errorf("publish lifecycle creation binding: %w", err)
		}
		delete(s.lifecycle.pendingLifecycleBindings, resource)
	}
	return nil
}

func (s *Store) resolvePendingLifecycleCreationBindingUnlocked(
	desired map[lifecycleResourceRef]lifecycleAuditRecord,
) error {
	if len(s.lifecycle.pendingLifecycleBindings) == 0 {
		return nil
	}
	if len(s.lifecycle.pendingLifecycleBindings) != 1 {
		return fmt.Errorf("%w: lifecycle creation binding publication is ambiguous", ErrConflict)
	}
	var resource lifecycleResourceRef
	var pending lifecycleAuditRecord
	for candidate, record := range s.lifecycle.pendingLifecycleBindings {
		resource, pending = candidate, record
		break
	}
	if expected, exists := desired[resource]; !exists || expected != pending {
		return fmt.Errorf("%w: lifecycle creation binding retry changed identity", ErrConflict)
	}
	path := filepath.Join(s.lifecycleBindingRoot, lifecycleBindingFileName(resource))
	var visible lifecycleAuditRecord
	if err := readJSON(path, &visible); err != nil {
		if errors.Is(err, ErrNotFound) {
			delete(s.lifecycle.pendingLifecycleBindings, resource)
			return nil
		}
		return err
	}
	if visible != pending {
		return fmt.Errorf("%w: lifecycle creation binding changed", ErrInvalid)
	}
	if err := s.lifecycleAuditWriter.SyncDirectory(
		s.lifecycleBindingRoot, "lifecycle creation binding commit retry",
	); err != nil {
		return fmt.Errorf("lifecycle creation binding durability is uncertain: %w", err)
	}
	delete(s.lifecycle.pendingLifecycleBindings, resource)
	return nil
}

func (s *Store) requireNoPendingLifecycleCreationBindings() error {
	if len(s.lifecycle.pendingLifecycleBindings) != 0 {
		return fmt.Errorf("%w: lifecycle creation binding requires the startup owner or explicit checkpoint retry", ErrConflict)
	}
	return nil
}

func (s *Store) installLifecycleCheckpointUnlocked(
	checkpoint lifecycleAuditCheckpoint,
	checkpointBytes int64,
	bindings map[lifecycleResourceRef]lifecycleAuditRecord,
) {
	s.lifecycle.sequence = checkpoint.Sequence
	s.lifecycle.checkpointSequence = checkpoint.Sequence
	s.lifecycle.headDigest = checkpoint.HeadDigest
	s.lifecycle.checkpointDigest = checkpoint.Digest
	s.lifecycle.checkpointSegmentStart = checkpoint.SegmentStartSequence
	s.lifecycle.checkpointSegmentRoot = checkpoint.SegmentPreviousDigest
	s.lifecycle.checkpointCleanup = true
	s.lifecycle.checkpointBindings = bindings
	s.lifecycle.activeCount = 0
	s.lifecycle.bytes = checkpointBytes
	s.lifecycle.records = make(map[string]lifecycleAuditRecord)
	s.lifecycle.creationBindings = lifecycleCreationBindingDigestIndex(bindings)
}

func (s *Store) finishLifecycleCheckpointCleanup() error {
	s.lifecycle.mu.Lock()
	defer s.lifecycle.mu.Unlock()
	return s.finishLifecycleCheckpointCleanupUnlocked()
}

func (s *Store) finishLifecycleCheckpointCleanupUnlocked() error {
	if !s.lifecycle.checkpointCleanup {
		return nil
	}
	if s.lifecycle.checkpointDurabilityPending {
		if err := s.lifecycleAuditWriter.SyncDirectory(
			s.lifecycleAuditRoot, "evaluation lifecycle checkpoint commit retry",
		); err != nil {
			return fmt.Errorf("evaluation lifecycle checkpoint durability is uncertain: %w", err)
		}
		s.lifecycle.checkpointDurabilityPending = false
	}
	bindings := s.lifecycle.checkpointBindings
	if bindings == nil {
		var err error
		bindings, err = s.lifecycleCreationBindingsForCheckpointUnlocked()
		if err != nil {
			return err
		}
	}
	if err := s.removeCompactedLifecycleAuditRecordsUnlocked(s.lifecycle.checkpointSequence); err != nil {
		return err
	}
	if err := s.removeStaleLifecycleCreationBindingsUnlocked(bindings); err != nil {
		return err
	}
	s.lifecycle.creationBindings = lifecycleCreationBindingDigestIndex(bindings)
	s.lifecycle.checkpointBindings = nil
	s.lifecycle.checkpointCleanup = false
	return nil
}

func lifecycleCreationBindingDigestIndex(
	bindings map[lifecycleResourceRef]lifecycleAuditRecord,
) map[string]lifecycleAuditRecord {
	indexed := make(map[string]lifecycleAuditRecord, len(bindings))
	for _, record := range bindings {
		indexed[record.Digest] = record
	}
	return indexed
}

func (s *Store) lifecycleCheckpointSegmentBoundaryUnlocked() (uint64, string, error) {
	if s.lifecycle.activeCount == 0 {
		if s.lifecycle.checkpointSegmentStart == 0 {
			return 0, "", fmt.Errorf("%w: lifecycle checkpoint segment boundary is missing", ErrInvalid)
		}
		return s.lifecycle.checkpointSegmentStart, s.lifecycle.checkpointSegmentRoot, nil
	}
	start := s.lifecycle.sequence - s.lifecycle.activeCount + 1
	for _, record := range s.lifecycle.records {
		if record.Sequence == start {
			return start, record.PreviousDigest, nil
		}
	}
	return 0, "", fmt.Errorf("%w: lifecycle checkpoint segment start is missing", ErrInvalid)
}

func validateCompactedLifecycleAuditSuffix(
	checkpoint lifecycleAuditCheckpoint,
	records []lifecycleAuditRecord,
) error {
	if len(records) == 0 {
		return nil
	}
	first := records[0]
	if first.Sequence < checkpoint.SegmentStartSequence {
		return fmt.Errorf("%w: compacted lifecycle audit precedes its checkpoint segment", ErrInvalid)
	}
	previous := first.PreviousDigest
	if first.Sequence == checkpoint.SegmentStartSequence && previous != checkpoint.SegmentPreviousDigest {
		return fmt.Errorf("%w: compacted lifecycle audit segment root is invalid", ErrInvalid)
	}
	expectedSequence := first.Sequence
	for _, record := range records {
		if record.Sequence != expectedSequence || record.PreviousDigest != previous {
			return fmt.Errorf("%w: compacted lifecycle audit suffix is not contiguous", ErrInvalid)
		}
		expectedSequence++
		previous = record.Digest
	}
	last := records[len(records)-1]
	if last.Sequence != checkpoint.Sequence || last.Digest != checkpoint.HeadDigest {
		return fmt.Errorf("%w: compacted lifecycle audit suffix is not checkpoint-bound", ErrInvalid)
	}
	return nil
}

func (s *Store) removeCompactedLifecycleAuditRecordsUnlocked(sequence uint64) error {
	entries, err := os.ReadDir(s.lifecycleAuditRoot)
	if err != nil {
		return fmt.Errorf("list compacted lifecycle audit records: %w", err)
	}
	for _, entry := range entries {
		match := lifecycleAuditFilePattern.FindStringSubmatch(entry.Name())
		if match == nil {
			continue
		}
		parsed, parseErr := parseLifecycleAuditSequence(match[1])
		if parseErr != nil {
			return parseErr
		}
		if parsed <= sequence {
			if err := s.lifecycleCleaner.Remove(filepath.Join(s.lifecycleAuditRoot, entry.Name())); err != nil {
				return fmt.Errorf("remove compacted lifecycle audit record: %w", err)
			}
		}
	}
	return s.lifecycleCleaner.Sync(s.lifecycleAuditRoot, "lifecycle audit checkpoint cleanup")
}

func (s *Store) removeStaleLifecycleCreationBindingsUnlocked(
	desired map[lifecycleResourceRef]lifecycleAuditRecord,
) error {
	entries, err := os.ReadDir(s.lifecycleBindingRoot)
	if err != nil {
		return fmt.Errorf("list lifecycle creation bindings for cleanup: %w", err)
	}
	for _, entry := range entries {
		if entry.IsDir() || entry.Type()&os.ModeSymlink != 0 ||
			!strings.HasSuffix(entry.Name(), lifecycleBindingFileSuffix) {
			return fmt.Errorf("%w: lifecycle creation binding cleanup found an invalid entry", ErrInvalid)
		}
		resource, validName := parseLifecycleBindingFileName(entry.Name())
		if !validName {
			return fmt.Errorf("%w: lifecycle creation binding cleanup found an invalid name", ErrInvalid)
		}
		if _, retained := desired[resource]; retained {
			continue
		}
		if err := s.lifecycleCleaner.Remove(filepath.Join(s.lifecycleBindingRoot, entry.Name())); err != nil {
			return fmt.Errorf("remove stale lifecycle creation binding: %w", err)
		}
	}
	return s.lifecycleCleaner.Sync(s.lifecycleBindingRoot, "lifecycle creation binding cleanup")
}

func (s *Store) activeNotFoundDenialWindows(now time.Time) []lifecycleAuditDenialWindow {
	s.pruneExpiredLifecycleNotFoundDenials(now)
	windows := make([]lifecycleAuditDenialWindow, 0, len(s.lifecycle.notFoundDenials))
	for key, timestamp := range s.lifecycle.notFoundDenials {
		parts := strings.SplitN(key, "\x00", 3)
		windows = append(windows, lifecycleAuditDenialWindow{
			ActorDigest: parts[0], ResourceKind: parts[1], Action: parts[2], Timestamp: timestamp,
		})
	}
	sort.Slice(windows, func(i, j int) bool {
		return lifecycleNotFoundDenialKey(windows[i].ActorDigest, windows[i].ResourceKind, windows[i].Action) <
			lifecycleNotFoundDenialKey(windows[j].ActorDigest, windows[j].ResourceKind, windows[j].Action)
	})
	return windows
}

func (s *Store) pruneExpiredLifecycleNotFoundDenials(now time.Time) {
	for key, timestamp := range s.lifecycle.notFoundDenials {
		if !now.Before(timestamp.Add(lifecycleNotFoundDedupeWindow)) {
			delete(s.lifecycle.notFoundDenials, key)
		}
	}
}

func trackLifecycleNotFoundDenial(denials map[string]time.Time, key string, timestamp time.Time) {
	if previous, exists := denials[key]; exists {
		if timestamp.After(previous) {
			denials[key] = timestamp
		}
		return
	}
	if len(denials) < maxLifecycleNotFoundDenialWindows {
		denials[key] = timestamp
	}
}

func (s *Store) maxLifecycleCheckpointBytes() int64 {
	budget := s.lifecyclePolicy.Limits.MaxAuditBytes - lifecycleAuditAppendReserveBytes
	if budget < maxLifecycleRecordSize {
		return budget
	}
	return maxLifecycleRecordSize
}

func lifecycleNotFoundDenialKey(actorDigest, resourceKind, action string) string {
	return actorDigest + "\x00" + resourceKind + "\x00" + action
}

func lifecycleCreationRecordMatches(
	record lifecycleAuditRecord,
	resource lifecycleResourceRef,
	ownerDigest string,
) bool {
	return record.Action == "create" && record.Decision == "allowed" &&
		record.ResourceKind == resource.Kind && record.ResourceID == resource.ID &&
		record.OwnerDigest == ownerDigest
}

func lifecycleBindingFileName(resource lifecycleResourceRef) string {
	return resource.Kind + "-" + resource.ID + lifecycleBindingFileSuffix
}

func parseLifecycleBindingFileName(name string) (lifecycleResourceRef, bool) {
	stem := strings.TrimSuffix(name, lifecycleBindingFileSuffix)
	kind, id, found := strings.Cut(stem, "-")
	resource := lifecycleResourceRef{Kind: kind, ID: id}
	return resource, found && strings.HasSuffix(name, lifecycleBindingFileSuffix) &&
		(kind == lifecycleResourceRun || kind == lifecycleResourceCampaign) && validClientRequestID(id)
}

func parseLifecycleAuditSequence(value string) (uint64, error) {
	sequence, err := strconv.ParseUint(value, 10, 64)
	if err != nil {
		return 0, fmt.Errorf("%w: lifecycle audit sequence is invalid", ErrInvalid)
	}
	return sequence, nil
}
