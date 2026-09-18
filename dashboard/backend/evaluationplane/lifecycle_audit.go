package evaluationplane

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"regexp"
	"sort"
	"strings"
	"time"
)

const lifecycleAuditDirectoryName = "audit"

const lifecycleAuditTempPrefix = ".tmp-lifecycle-audit-"

var lifecycleAuditFilePattern = regexp.MustCompile(`^([0-9]{20})-([0-9a-f]{64})\.json$`)

var lifecycleAuditTempPattern = regexp.MustCompile(`^\.tmp-lifecycle-audit-[A-Za-z0-9]+$`)

type lifecycleAuditRecord struct {
	SchemaVersion  string    `json:"schema_version"`
	Sequence       uint64    `json:"sequence"`
	Timestamp      time.Time `json:"timestamp"`
	Action         string    `json:"action"`
	Decision       string    `json:"decision"`
	ReasonCode     string    `json:"reason_code"`
	ActorDigest    string    `json:"actor_digest"`
	ResourceKind   string    `json:"resource_kind"`
	ResourceID     string    `json:"resource_id,omitempty"`
	OwnerDigest    string    `json:"owner_digest,omitempty"`
	PreviousDigest string    `json:"previous_digest,omitempty"`
	Digest         string    `json:"digest"`
}

const (
	lifecycleResourceRun      = "run"
	lifecycleResourceCampaign = "campaign"
	lifecycleResourceStore    = "store"
)

type lifecycleResourceRef struct {
	Kind string
	ID   string
}

type lifecycleAuditWriter interface {
	WriteExclusive(path string, value any) error
	SyncDirectory(path, description string) error
}

type atomicLifecycleAuditWriter struct{}

func (atomicLifecycleAuditWriter) WriteExclusive(path string, value any) error {
	encoded, err := json.MarshalIndent(value, "", "  ")
	if err != nil {
		return fmt.Errorf("encode lifecycle audit record: %w", err)
	}
	encoded = append(encoded, '\n')
	if int64(len(encoded)) > maxLifecycleRecordSize {
		return fmt.Errorf("lifecycle audit record exceeds its durable envelope")
	}
	temporary, err := os.CreateTemp(filepath.Dir(path), lifecycleAuditTempPrefix+"*")
	if err != nil {
		return fmt.Errorf("stage lifecycle audit record: %w", err)
	}
	temporaryPath := temporary.Name()
	defer func() { _ = os.Remove(temporaryPath) }()
	if err := temporary.Chmod(0o600); err != nil {
		_ = temporary.Close()
		return fmt.Errorf("protect lifecycle audit record: %w", err)
	}
	if _, err := temporary.Write(encoded); err != nil {
		_ = temporary.Close()
		return fmt.Errorf("write lifecycle audit record: %w", err)
	}
	if err := temporary.Sync(); err != nil {
		_ = temporary.Close()
		return fmt.Errorf("sync lifecycle audit record: %w", err)
	}
	if err := temporary.Close(); err != nil {
		return fmt.Errorf("close lifecycle audit record: %w", err)
	}
	if err := os.Link(temporaryPath, path); err != nil {
		return fmt.Errorf("publish lifecycle audit record: %w", err)
	}
	return syncEvaluationDirectory(filepath.Dir(path), "evaluation lifecycle audit")
}

func (atomicLifecycleAuditWriter) SyncDirectory(path, description string) error {
	return syncEvaluationDirectory(path, description)
}

func recoverLifecycleAuditTemps(root string) error {
	entries, err := os.ReadDir(root)
	if err != nil {
		return fmt.Errorf("list staged lifecycle audit records: %w", err)
	}
	removed := false
	for _, entry := range entries {
		if !strings.HasPrefix(entry.Name(), lifecycleAuditTempPrefix) {
			continue
		}
		if !lifecycleAuditTempPattern.MatchString(entry.Name()) {
			return fmt.Errorf("%w: staged lifecycle audit record name is invalid", ErrInvalid)
		}
		path := filepath.Join(root, entry.Name())
		info, statErr := os.Lstat(path)
		if statErr != nil || !info.Mode().IsRegular() || info.Mode()&os.ModeSymlink != 0 || info.Mode().Perm() != 0o600 {
			return fmt.Errorf("%w: staged lifecycle audit record is invalid", ErrInvalid)
		}
		if err := os.Remove(path); err != nil {
			return fmt.Errorf("remove staged lifecycle audit record: %w", err)
		}
		removed = true
	}
	if removed {
		return syncEvaluationDirectory(root, "lifecycle audit recovery")
	}
	return nil
}

func requireNoLifecycleAuditTemps(root string) error {
	entries, err := os.ReadDir(root)
	if err != nil {
		return fmt.Errorf("list staged lifecycle audit records: %w", err)
	}
	for _, entry := range entries {
		if !strings.HasPrefix(entry.Name(), lifecycleAuditTempPrefix) {
			continue
		}
		if !lifecycleAuditTempPattern.MatchString(entry.Name()) {
			return fmt.Errorf("%w: staged lifecycle audit record name is invalid", ErrInvalid)
		}
		path := filepath.Join(root, entry.Name())
		info, statErr := os.Lstat(path)
		if statErr != nil || !info.Mode().IsRegular() || info.Mode()&os.ModeSymlink != 0 || info.Mode().Perm() != 0o600 {
			return fmt.Errorf("%w: staged lifecycle audit record is invalid", ErrInvalid)
		}
		return fmt.Errorf("%w: lifecycle audit recovery requires the startup owner", ErrConflict)
	}
	return nil
}

func (s *Store) validateLifecycleAuditUnlocked() error {
	return s.validateLifecycleAuditForOpenUnlocked(true)
}

func (s *Store) validatePeerLifecycleAuditUnlocked() error {
	return s.validateLifecycleAuditForOpenUnlocked(false)
}

type lifecycleAuditProjection struct {
	sequence     uint64
	activeCount  uint64
	headDigest   string
	bytes        int64
	records      map[string]lifecycleAuditRecord
	denials      map[string]time.Time
	staleRecords []lifecycleAuditRecord
}

func (s *Store) validateLifecycleAuditForOpenUnlocked(startupAuthority bool) error {
	if !startupAuthority && (s.lifecycle.checkpointCleanup || s.lifecycle.checkpointDurabilityPending) {
		return fmt.Errorf("%w: lifecycle checkpoint cleanup requires the startup owner or explicit audit retry", ErrConflict)
	}
	if !startupAuthority {
		if err := s.requireNoPendingLifecycleCreationBindings(); err != nil {
			return err
		}
	}
	checkpoint, checkpointBytes, err := s.loadLifecycleAuditCheckpointUnlocked()
	if err != nil {
		return err
	}
	bindings, err := s.loadLifecycleCreationBindingsUnlocked()
	if err != nil {
		return err
	}
	projection, err := s.loadLifecycleAuditProjectionUnlocked(checkpoint, checkpointBytes)
	if err != nil {
		return err
	}
	if checkpoint.Sequence > 0 {
		if err := validateCompactedLifecycleAuditSuffix(checkpoint, projection.staleRecords); err != nil {
			return err
		}
	}
	if !startupAuthority && len(projection.staleRecords) != 0 {
		return fmt.Errorf("%w: lifecycle checkpoint cleanup requires the startup owner", ErrConflict)
	}
	// Startup is also an idempotent retry boundary. Close any prior process's
	// visible-link uncertainty before installing the recovered chain in memory.
	if startupAuthority {
		for _, directory := range []struct {
			path        string
			description string
		}{
			{s.lifecycleAuditRoot, "evaluation lifecycle audit startup recovery"},
			{s.lifecycleBindingRoot, "evaluation lifecycle binding startup recovery"},
		} {
			if err := s.lifecycleAuditWriter.SyncDirectory(directory.path, directory.description); err != nil {
				return fmt.Errorf("evaluation lifecycle audit durability is uncertain: %w", err)
			}
		}
		clear(s.lifecycle.pendingLifecycleBindings)
	}
	if !startupAuthority {
		if !s.lifecycle.loaded || s.lifecycle.sequence != projection.sequence ||
			s.lifecycle.activeCount != projection.activeCount || s.lifecycle.headDigest != projection.headDigest ||
			s.lifecycle.checkpointDigest != checkpoint.Digest || s.lifecycle.bytes != projection.bytes ||
			s.lifecycle.checkpointSequence != checkpoint.Sequence ||
			s.lifecycle.checkpointSegmentStart != checkpoint.SegmentStartSequence ||
			s.lifecycle.checkpointSegmentRoot != checkpoint.SegmentPreviousDigest ||
			!reflect.DeepEqual(s.lifecycle.records, projection.records) ||
			!reflect.DeepEqual(s.lifecycle.creationBindings, bindings) ||
			!reflect.DeepEqual(s.lifecycle.notFoundDenials, projection.denials) {
			return fmt.Errorf("%w: lifecycle audit projection does not match the active root owner", ErrConflict)
		}
		return nil
	}
	s.lifecycle.sequence, s.lifecycle.activeCount = projection.sequence, projection.activeCount
	s.lifecycle.headDigest, s.lifecycle.checkpointDigest, s.lifecycle.bytes = projection.headDigest, checkpoint.Digest, projection.bytes
	s.lifecycle.checkpointSequence = checkpoint.Sequence
	s.lifecycle.checkpointSegmentStart = checkpoint.SegmentStartSequence
	s.lifecycle.checkpointSegmentRoot = checkpoint.SegmentPreviousDigest
	s.lifecycle.records, s.lifecycle.creationBindings = projection.records, bindings
	s.lifecycle.notFoundDenials, s.lifecycle.loaded = projection.denials, true
	s.lifecycle.checkpointCleanup = checkpoint.Sequence > 0
	s.lifecycle.checkpointDurabilityPending = false
	s.lifecycle.checkpointBindings = nil
	return nil
}

func (s *Store) loadLifecycleAuditProjectionUnlocked(
	checkpoint lifecycleAuditCheckpoint,
	checkpointBytes int64,
) (lifecycleAuditProjection, error) {
	entries, err := os.ReadDir(s.lifecycleAuditRoot)
	if err != nil {
		return lifecycleAuditProjection{}, fmt.Errorf("list evaluation lifecycle audit: %w", err)
	}
	sort.Slice(entries, func(i, j int) bool { return entries[i].Name() < entries[j].Name() })
	projection := lifecycleAuditProjection{
		sequence: checkpoint.Sequence, headDigest: checkpoint.HeadDigest, bytes: checkpointBytes,
		records: make(map[string]lifecycleAuditRecord, len(entries)),
		denials: make(map[string]time.Time, len(checkpoint.NotFoundDenials)),
	}
	for _, denial := range checkpoint.NotFoundDenials {
		projection.denials[lifecycleNotFoundDenialKey(
			denial.ActorDigest, denial.ResourceKind, denial.Action,
		)] = denial.Timestamp
	}
	for _, entry := range entries {
		if entry.Name() == lifecycleAuditCheckpointFileName {
			continue
		}
		match := lifecycleAuditFilePattern.FindStringSubmatch(entry.Name())
		if entry.IsDir() || entry.Type()&os.ModeSymlink != 0 || match == nil {
			return lifecycleAuditProjection{}, fmt.Errorf("%w: lifecycle audit contains an invalid entry", ErrInvalid)
		}
		parsedSequence, parseErr := parseLifecycleAuditSequence(match[1])
		if parseErr != nil {
			return lifecycleAuditProjection{}, parseErr
		}
		path := filepath.Join(s.lifecycleAuditRoot, entry.Name())
		info, statErr := os.Lstat(path)
		if statErr != nil || !info.Mode().IsRegular() || info.Mode().Perm() != 0o600 || info.Size() > maxLifecycleRecordSize {
			return lifecycleAuditProjection{}, fmt.Errorf("%w: lifecycle audit record is not a private bounded file", ErrInvalid)
		}
		var record lifecycleAuditRecord
		if readErr := readJSON(path, &record); readErr != nil {
			return lifecycleAuditProjection{}, fmt.Errorf("%w: lifecycle audit record is unreadable: %w", ErrInvalid, readErr)
		}
		if record.Sequence != parsedSequence || match[2] != trimSHA256(record.Digest) || validateLifecycleAuditRecord(record) != nil {
			return lifecycleAuditProjection{}, fmt.Errorf("%w: lifecycle audit hash chain is invalid", ErrInvalid)
		}
		if parsedSequence <= checkpoint.Sequence {
			projection.staleRecords = append(projection.staleRecords, record)
			continue
		}
		projection.activeCount++
		if parsedSequence != projection.sequence+1 || record.PreviousDigest != projection.headDigest ||
			projection.activeCount > maxLifecycleAuditCount {
			return lifecycleAuditProjection{}, fmt.Errorf("%w: lifecycle audit sequence is invalid", ErrInvalid)
		}
		projection.bytes += info.Size()
		if projection.bytes > s.lifecyclePolicy.Limits.MaxAuditBytes {
			return lifecycleAuditProjection{}, fmt.Errorf("%w: lifecycle audit exceeds its configured bound", ErrInvalid)
		}
		projection.sequence, projection.headDigest = record.Sequence, record.Digest
		projection.records[record.Digest] = record
		if record.Decision == "denied" && record.ReasonCode == "not_found" {
			trackLifecycleNotFoundDenial(projection.denials, lifecycleNotFoundDenialKey(
				record.ActorDigest, record.ResourceKind, record.Action,
			), record.Timestamp)
		}
	}
	return projection, nil
}

func validateLifecycleAuditRecord(record lifecycleAuditRecord) error {
	if record.SchemaVersion != lifecycleAuditSchemaVersion || record.Sequence == 0 ||
		record.Timestamp.IsZero() || !validLifecycleAction(record.Action) ||
		(record.Decision != "allowed" && record.Decision != "denied") ||
		!validLifecycleReason(record.ReasonCode) || !digestPattern.MatchString(record.ActorDigest) ||
		(record.OwnerDigest != "" && !digestPattern.MatchString(record.OwnerDigest)) ||
		!validLifecycleResource(
			record.ResourceKind, record.ResourceID, record.Action, record.Decision, record.ReasonCode,
		) ||
		(record.PreviousDigest != "" && !digestPattern.MatchString(record.PreviousDigest)) ||
		record.Digest != lifecycleAuditDigest(record) {
		return fmt.Errorf("invalid lifecycle audit record")
	}
	return nil
}

func validLifecycleResource(kind, id, action, decision, reason string) bool {
	switch kind {
	case lifecycleResourceStore:
		return id == "" && action == "gc"
	case lifecycleResourceRun:
		return action != "gc" &&
			(validClientRequestID(id) || (id == "" && decision == "denied" && reason == "not_found"))
	case lifecycleResourceCampaign:
		if action == "start" || action == "cancel" || action == "gc" {
			return false
		}
		return validClientRequestID(id) || (id == "" && decision == "denied" && reason == "not_found")
	default:
		return false
	}
}

func lifecycleAuditDigest(record lifecycleAuditRecord) string {
	record.Digest = ""
	encoded, err := json.Marshal(record)
	if err != nil {
		panic(err)
	}
	return digestBytes(encoded)
}

func trimSHA256(digest string) string {
	if len(digest) == len("sha256:")+64 {
		return digest[len("sha256:"):]
	}
	return ""
}

func validLifecycleAction(action string) bool {
	switch action {
	case "create", "start", "cancel", "hold", "release", "retention", "delete", "gc":
		return true
	default:
		return false
	}
}

func validLifecycleReason(reason string) bool {
	switch reason {
	case "owner", "administrator", "system", "not_owner", "not_administrator", "not_found",
		"conflict", "invalid_request", "invalid_evidence",
		"quota_owner_bytes", "quota_owner_runs", "quota_owner_campaigns", "quota_store_bytes", "evidence_hold",
		"protected_retention", "referenced", "dry_run", "apply", "delete_cascade", "startup_recovery":
		return true
	default:
		return false
	}
}

func (s *Store) appendLifecycleAuditUnlocked(
	actor Actor,
	resourceKind, action, decision, reasonCode, resourceID, ownerDigest string,
) (lifecycleAuditRecord, error) {
	if err := validateActor(actor); err != nil {
		return lifecycleAuditRecord{}, err
	}
	if !s.lifecycle.loaded {
		return lifecycleAuditRecord{}, fmt.Errorf("evaluation lifecycle audit is not initialized")
	}
	if err := s.finishLifecycleCheckpointCleanupUnlocked(); err != nil {
		return lifecycleAuditRecord{}, err
	}
	now := s.lifecycleNow().UTC().Truncate(time.Microsecond)
	resourceID, denialKey, deduplicated, auditErr := s.prepareLifecycleNotFoundDenialUnlocked(
		actor, resourceKind, action, decision, reasonCode, resourceID, now,
	)
	if auditErr != nil || deduplicated {
		return lifecycleAuditRecord{}, auditErr
	}
	if s.lifecycle.activeCount >= maxLifecycleAuditCount {
		if err := s.checkpointLifecycleAuditUnlocked(now); err != nil {
			return lifecycleAuditRecord{}, err
		}
	}
	record, auditErr := s.newLifecycleAuditRecord(
		actor, resourceKind, action, decision, reasonCode, resourceID, ownerDigest, now,
	)
	if auditErr != nil {
		return lifecycleAuditRecord{}, auditErr
	}
	name := fmt.Sprintf("%020d-%s.json", record.Sequence, trimSHA256(record.Digest))
	encoded, auditErr := encodeLifecycleAuditRecord(record)
	if auditErr != nil {
		return lifecycleAuditRecord{}, auditErr
	}
	projected := s.lifecycle.bytes + int64(len(encoded)+1)
	if projected > s.lifecyclePolicy.Limits.MaxAuditBytes {
		if err := s.checkpointLifecycleAuditUnlocked(now); err != nil {
			return lifecycleAuditRecord{}, err
		}
		record, auditErr = s.newLifecycleAuditRecord(
			actor, resourceKind, action, decision, reasonCode, resourceID, ownerDigest, now,
		)
		if auditErr != nil {
			return lifecycleAuditRecord{}, auditErr
		}
		name = fmt.Sprintf("%020d-%s.json", record.Sequence, trimSHA256(record.Digest))
		encoded, auditErr = encodeLifecycleAuditRecord(record)
		if auditErr != nil {
			return lifecycleAuditRecord{}, auditErr
		}
		projected = s.lifecycle.bytes + int64(len(encoded)+1)
		if projected > s.lifecyclePolicy.Limits.MaxAuditBytes {
			return lifecycleAuditRecord{}, fmt.Errorf("%w: lifecycle audit byte bound cannot fit checkpoint plus one record", ErrQuota)
		}
	}
	path := filepath.Join(s.lifecycleAuditRoot, name)
	if err := s.lifecycleAuditWriter.WriteExclusive(path, record); err != nil {
		// Link publication can succeed before the audit-directory fsync reports
		// failure. Reconcile that exact immutable path/content into the in-memory
		// chain while still returning the durability error to the operation. A
		// retry therefore advances from the committed sequence instead of writing
		// a different timestamp at the same sequence.
		var visible lifecycleAuditRecord
		if readErr := readJSON(path, &visible); readErr == nil && reflect.DeepEqual(visible, record) {
			s.lifecycle.sequence, s.lifecycle.headDigest, s.lifecycle.bytes = record.Sequence, record.Digest, projected
			s.lifecycle.activeCount++
			s.lifecycle.records[record.Digest] = record
			if denialKey != "" {
				s.lifecycle.notFoundDenials[denialKey] = now
			}
			return record, err
		}
		return lifecycleAuditRecord{}, err
	}
	s.lifecycle.sequence, s.lifecycle.headDigest, s.lifecycle.bytes = record.Sequence, record.Digest, projected
	s.lifecycle.activeCount++
	s.lifecycle.records[record.Digest] = record
	if denialKey != "" {
		s.lifecycle.notFoundDenials[denialKey] = now
	}
	return record, nil
}

func (s *Store) prepareLifecycleNotFoundDenialUnlocked(
	actor Actor,
	resourceKind string,
	action string,
	decision string,
	reasonCode string,
	resourceID string,
	now time.Time,
) (string, string, bool, error) {
	if decision != "denied" || reasonCode != "not_found" {
		return resourceID, "", false, nil
	}
	s.pruneExpiredLifecycleNotFoundDenials(now)
	candidateKey := lifecycleNotFoundDenialKey(actor.principalDigest, resourceKind, action)
	if previous, exists := s.lifecycle.notFoundDenials[candidateKey]; exists &&
		now.Before(previous.Add(lifecycleNotFoundDedupeWindow)) {
		// A previous immutable link can be visible even though its parent fsync
		// failed. Dedupe must close that boundary before acknowledging the retry.
		if err := s.lifecycleAuditWriter.SyncDirectory(
			s.lifecycleAuditRoot, "evaluation lifecycle audit dedupe retry",
		); err != nil {
			return "", "", false, fmt.Errorf(
				"evaluation lifecycle audit durability is uncertain: %w", err,
			)
		}
		return "", "", true, nil
	}
	denialKey := ""
	if len(s.lifecycle.notFoundDenials) < maxLifecycleNotFoundDenialWindows {
		denialKey = candidateKey
	}
	// A random not-found identifier carries no durable forensic value. The
	// actor/action/time window remains audited without a UUID-cardinality attack.
	return "", denialKey, false, nil
}

func encodeLifecycleAuditRecord(record lifecycleAuditRecord) ([]byte, error) {
	encoded, err := json.MarshalIndent(record, "", "  ")
	if err != nil {
		return nil, err
	}
	if int64(len(encoded)+1) > lifecycleAuditAppendReserveBytes {
		return nil, fmt.Errorf("%w: lifecycle audit record exceeds its append reserve", ErrInvalid)
	}
	return encoded, nil
}

func (s *Store) newLifecycleAuditRecord(
	actor Actor,
	resourceKind, action, decision, reasonCode, resourceID, ownerDigest string,
	now time.Time,
) (lifecycleAuditRecord, error) {
	record := lifecycleAuditRecord{
		SchemaVersion: lifecycleAuditSchemaVersion, Sequence: s.lifecycle.sequence + 1,
		Timestamp: now, Action: action, Decision: decision, ReasonCode: reasonCode,
		ActorDigest: actor.principalDigest, ResourceKind: resourceKind, ResourceID: resourceID,
		OwnerDigest:    ownerDigest,
		PreviousDigest: s.lifecycle.headDigest,
	}
	record.Digest = lifecycleAuditDigest(record)
	if err := validateLifecycleAuditRecord(record); err != nil {
		return lifecycleAuditRecord{}, fmt.Errorf("encode lifecycle audit decision: %w", err)
	}
	return record, nil
}
