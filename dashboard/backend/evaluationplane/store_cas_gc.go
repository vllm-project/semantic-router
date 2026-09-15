package evaluationplane

import (
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"
)

// DeleteRun and worker evidence import share one publication lock. Once the
// run directory removal is durable, a complete mark precedes every CAS sweep;
// failures therefore retain garbage instead of risking a live object.
func (s *Store) DeleteRunAs(actor Actor, id string) error {
	s.lifecycle.mu.Lock()
	defer s.lifecycle.mu.Unlock()
	if resumed, err := s.resumeRunDeletionAsUnlocked(actor, id); resumed || err != nil {
		return err
	}
	if err := s.authorizeRunActionUnlocked(actor, id, "delete"); err != nil {
		return err
	}
	return s.deleteRunAuthorizedUnlocked(actor, id)
}

func (s *Store) deleteRunAuthorizedUnlocked(actor Actor, id string) error {
	if err := validateResourceID(id); err != nil {
		return err
	}
	s.lifecycle.evidenceMu.Lock()
	defer s.lifecycle.evidenceMu.Unlock()
	s.runIndex.coordinator.Lock()
	defer s.runIndex.coordinator.Unlock()
	s.mu.Lock()
	defer s.mu.Unlock()
	run, runErr := s.getRunUnlocked(id)
	if runErr != nil {
		return runErr
	}
	lifecycle, lifecycleErr := s.readRunLifecycle(run)
	if lifecycleErr != nil {
		return lifecycleErr
	}
	if err := s.ensureRunNotBaselineReferencedUnlocked(id); err != nil {
		if _, auditErr := s.appendLifecycleAuditUnlocked(
			actor, lifecycleResourceRun, "delete", "denied", "referenced", id, lifecycle.OwnerPrincipalDigest,
		); auditErr != nil {
			return auditErr
		}
		return err
	}
	if err := s.ensureRunNotCampaignReferencedUnlocked(id); err != nil {
		if _, auditErr := s.appendLifecycleAuditUnlocked(
			actor, lifecycleResourceRun, "delete", "denied", "referenced", id, lifecycle.OwnerPrincipalDigest,
		); auditErr != nil {
			return auditErr
		}
		return err
	}
	if err := s.ensureRunNotControlledPairReferencedUnlocked(id); err != nil {
		if _, auditErr := s.appendLifecycleAuditUnlocked(
			actor, lifecycleResourceRun, "delete", "denied", "referenced", id, lifecycle.OwnerPrincipalDigest,
		); auditErr != nil {
			return auditErr
		}
		return err
	}

	runDir, runDirErr := s.checkedRunDir(id)
	if runDirErr != nil {
		return runDirErr
	}
	if _, auditErr := s.appendLifecycleAuditUnlocked(
		actor, lifecycleResourceRun, "delete", "allowed", lifecycleOwnerAuthorizationReason(actor, lifecycle.OwnerPrincipalDigest), id, lifecycle.OwnerPrincipalDigest,
	); auditErr != nil {
		return auditErr
	}
	return s.publishRunDeletionUnlocked(actor, run, runDir)
}

// recoverCASGarbage finishes a deletion whose process stopped after the run
// directory commit but before its sweep. Collection is opportunistic: a
// malformed remaining bundle prevents all object deletion while still allowing
// the ledger to quarantine and expose that bundle.
func (s *Store) recoverCASGarbage() {
	s.lifecycle.mu.Lock()
	defer s.lifecycle.mu.Unlock()
	s.lifecycle.evidenceMu.Lock()
	defer s.lifecycle.evidenceMu.Unlock()
	s.runIndex.coordinator.Lock()
	defer s.runIndex.coordinator.Unlock()
	s.mu.Lock()
	defer s.mu.Unlock()
	candidates, planErr := s.unreferencedCASCandidatesUnlocked()
	if planErr != nil {
		log.Printf("evaluationplane: startup CAS collection deferred: %v", planErr)
		return
	}
	if len(candidates) == 0 {
		return
	}
	if _, auditErr := s.appendLifecycleAuditUnlocked(
		SystemActor(), lifecycleResourceStore, "gc", "allowed", "startup_recovery", "", "",
	); auditErr != nil {
		log.Printf("evaluationplane: startup CAS collection audit failed closed: %v", auditErr)
		return
	}
	if err := removeCASCandidates(candidates); err != nil {
		log.Printf("evaluationplane: startup CAS collection deferred: %v", err)
	}
}

func (s *Store) sweepUnreferencedCASUnlocked() error {
	candidates, err := s.unreferencedCASCandidatesUnlocked()
	if err != nil {
		return err
	}
	return removeCASCandidates(candidates)
}

func (s *Store) unreferencedCASCandidatesUnlocked() ([]string, error) {
	references, markErr := s.markCASReferencesUnlocked()
	if markErr != nil {
		return nil, markErr
	}
	casRoot := filepath.Join(s.root, "objects", "sha256")
	if validationErr := requirePrivateDirectory(casRoot); validationErr != nil {
		return nil, fmt.Errorf("validate evaluation CAS directory: %w", validationErr)
	}
	entries, readErr := os.ReadDir(casRoot)
	if readErr != nil {
		return nil, fmt.Errorf("list evaluation CAS objects: %w", readErr)
	}
	candidates := make([]string, 0)
	for _, entry := range entries {
		if entry.IsDir() || !casObjectNamePattern.MatchString(entry.Name()) {
			return nil, fmt.Errorf("evaluation CAS contains an invalid object entry")
		}
		path := filepath.Join(casRoot, entry.Name())
		info, statErr := os.Lstat(path)
		if statErr != nil || !info.Mode().IsRegular() || info.Mode()&os.ModeSymlink != 0 || info.Mode().Perm() != 0o600 {
			return nil, fmt.Errorf("evaluation CAS object %s is not a private regular file", entry.Name())
		}
		if !references[entry.Name()] {
			candidates = append(candidates, path)
		}
	}
	return candidates, nil
}

func removeCASCandidates(candidates []string) error {
	if len(candidates) == 0 {
		return nil
	}
	for _, path := range candidates {
		if removeErr := os.Remove(path); removeErr != nil && !os.IsNotExist(removeErr) {
			return fmt.Errorf("remove unreferenced evaluation CAS object: %w", removeErr)
		}
	}
	return syncEvaluationDirectory(filepath.Dir(candidates[0]), "evaluation CAS collection")
}

func (s *Store) markCASReferencesUnlocked() (map[string]bool, error) {
	if err := s.requireNoPendingRunPublications(); err != nil {
		return nil, err
	}
	if err := s.requireNoRunDeletionIntentsUnlocked(); err != nil {
		return nil, err
	}
	references := make(map[string]bool)
	entries, err := os.ReadDir(s.runsRoot)
	if err != nil {
		return nil, fmt.Errorf("list evaluation runs for CAS collection: %w", err)
	}
	for _, entry := range entries {
		if !entry.IsDir() || !validClientRequestID(entry.Name()) {
			return nil, fmt.Errorf("evaluation runs contain an entry with unknown CAS ownership")
		}
		if err := s.markRunCASReferences(entry.Name(), references); err != nil {
			return nil, fmt.Errorf("mark evaluation run %s CAS references: %w", entry.Name(), err)
		}
	}
	return references, nil
}

func (s *Store) markRunCASReferences(runID string, references map[string]bool) error {
	runDir, runDirErr := s.checkedRunDir(runID)
	if runDirErr != nil {
		return runDirErr
	}
	if _, runErr := s.GetRun(runID); runErr != nil {
		return fmt.Errorf("read run before CAS reference scan: %w", runErr)
	}
	allowed := map[string]bool{
		runFileName: true, eventsFileName: true, reportAnchorFileName: true, lifecycleFileName: true,
		controlledPairMembershipFile: true,
	}
	for _, name := range workerRunArtifactNames {
		allowed[name] = true
	}
	entries, err := os.ReadDir(runDir)
	if err != nil {
		return err
	}
	present := make(map[string]bool, len(entries))
	for _, entry := range entries {
		if entry.IsDir() || !allowed[entry.Name()] {
			return fmt.Errorf("run bundle contains an entry with unknown CAS ownership")
		}
		present[entry.Name()] = true
	}

	artifactDigests := make(map[string]string)
	for _, name := range workerRunArtifactNames {
		if !present[name] {
			continue
		}
		digest, _, digestErr := privateFileHexDigest(filepath.Join(runDir, name), workerArtifactLimit(name))
		if digestErr != nil {
			return fmt.Errorf("hash run artifact %s: %w", name, digestErr)
		}
		artifactDigests[name] = digest
		references[digest] = true
	}
	if present["lineage.json"] {
		lineage, readErr := readEvidenceBytes(filepath.Join(runDir, "lineage.json"), maxStructuredArtifactBytes)
		if readErr != nil {
			return readErr
		}
		if _, lineageErr := decodeLineageDocument(lineage); lineageErr != nil {
			return lineageErr
		}
		value, decodeErr := decodeJSONValue(lineage)
		if decodeErr != nil {
			return decodeErr
		}
		collectCASReferences(value, references)
	}
	if present[privateChecksumArtifactName] {
		if err := validateGCPrivateReceipt(runDir, present, artifactDigests); err != nil {
			return err
		}
	}
	if present[reportAnchorFileName] {
		anchor, anchorErr := s.readReportAnchor(runID)
		if anchorErr != nil {
			return anchorErr
		}
		for _, evidence := range anchor.EvidenceFiles {
			references[strings.TrimPrefix(evidence.Digest, "sha256:")] = true
		}
	}
	return nil
}

func validateGCPrivateReceipt(runDir string, present map[string]bool, artifactDigests map[string]string) error {
	receipt, err := readEvidenceBytes(filepath.Join(runDir, privateChecksumArtifactName), maxStructuredArtifactBytes)
	if err != nil {
		return err
	}
	allowed := make(map[string]bool, len(workerRunArtifactNames))
	for _, name := range workerRunArtifactNames {
		allowed[name] = true
	}
	checksums, err := parseChecksumReceipt(receipt, allowed)
	if err != nil {
		return err
	}
	excluded := map[string]bool{privateChecksumArtifactName: true, reportFileName: true}
	expected := 0
	for _, name := range workerRunArtifactNames {
		if excluded[name] || !present[name] {
			continue
		}
		expected++
		if checksums[name] != artifactDigests[name] {
			return fmt.Errorf("private checksum receipt does not match run artifact %s", name)
		}
	}
	if len(checksums) != expected {
		return fmt.Errorf("private checksum receipt does not match the run bundle")
	}
	return nil
}
