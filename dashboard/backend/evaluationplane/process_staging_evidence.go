package evaluationplane

import (
	"bytes"
	"fmt"
	"os"
	"path/filepath"
)

const (
	maxStructuredArtifactBytes = int64(16 * 1024 * 1024)
	maxWorkerArtifactBytes     = int64(256 * 1024 * 1024)
	maxWorkerBundleBytes       = int64(1024 * 1024 * 1024)
)

var workerRunArtifactNames = []string{
	manifestFileName,
	"cases.jsonl",
	"grading-cases.jsonl",
	"records.jsonl",
	"metrics.json",
	"gates.json",
	"lineage.json",
	"provenance.json",
	"failure-summary.json",
	"routing-traces.jsonl",
	"capacity-profile.json",
	"checksums.sha256",
	"private-checksums.sha256",
	reportFileName,
}

var requiredWorkerRunArtifacts = map[string]bool{
	manifestFileName:           true,
	"cases.jsonl":              true,
	"grading-cases.jsonl":      true,
	"records.jsonl":            true,
	"metrics.json":             true,
	"gates.json":               true,
	"lineage.json":             true,
	"provenance.json":          true,
	"failure-summary.json":     true,
	"checksums.sha256":         true,
	"private-checksums.sha256": true,
	reportFileName:             true,
}

func (staging *workerStaging) importEvidence() error {
	return staging.importEvidenceWithBudget(maxWorkerBundleBytes)
}

func (staging *workerStaging) importEvidenceWithBudget(bundleLimit int64) error {
	if staging.evidencePublication == nil {
		return fmt.Errorf("worker evidence publication coordinator is not configured")
	}
	return staging.evidencePublication(func() error {
		return staging.importEvidenceDuringPublication(bundleLimit, nil)
	})
}

func (staging *workerStaging) importEvidenceDuringPublication(
	bundleLimit int64,
	quotaCheck func(runID string, runBytes, logicalCASBytes, physicalCASBytes int64) error,
) error {
	destinationRun := filepath.Join(staging.destinationStore, "runs", staging.runID)
	if err := requirePrivateDirectory(destinationRun); err != nil {
		return fmt.Errorf("validate destination run bundle: %w", err)
	}
	if err := staging.validateEvidenceManifests(destinationRun); err != nil {
		return err
	}

	sourceRun := filepath.Join(staging.storePath, "runs", staging.runID)
	seen, err := discoverWorkerRunArtifacts(sourceRun)
	if err != nil {
		return err
	}
	artifactDigests, casReferences, runBytes, err := workerCASReferences(sourceRun, seen)
	if err != nil {
		return err
	}
	casPlan, casBytes, err := staging.planCASObjects(casReferences)
	if err != nil {
		return err
	}
	if runBytes > bundleLimit || casBytes > bundleLimit-runBytes {
		return fmt.Errorf("worker evidence bundle exceeds the import limit")
	}
	var physicalCASBytes int64
	for _, object := range casPlan {
		if !object.exists {
			physicalCASBytes += object.size
		}
	}
	if quotaCheck != nil {
		if err := quotaCheck(staging.runID, runBytes, casBytes, physicalCASBytes); err != nil {
			return err
		}
	}
	destinationCAS := filepath.Join(staging.destinationStore, "objects", "sha256")
	created := make([]string, 0, len(casPlan)+len(seen))
	rollback := func() {
		for index := len(created) - 1; index >= 0; index-- {
			_ = os.Remove(created[index])
		}
		_ = syncEvaluationDirectory(destinationCAS, "evaluation CAS rollback")
		_ = syncEvaluationDirectory(destinationRun, "evaluation run evidence rollback")
	}
	for _, object := range casPlan {
		if object.exists {
			continue
		}
		if _, copyErr := copyPrivateFileExclusive(
			object.source, object.destination, object.digest, maxWorkerArtifactBytes,
		); copyErr != nil {
			rollback()
			return fmt.Errorf("import worker CAS object: %w", copyErr)
		}
		created = append(created, object.destination)
	}
	for _, name := range workerRunArtifactNames {
		if name == manifestFileName || !seen[name] {
			continue
		}
		destination := filepath.Join(destinationRun, name)
		_, copyErr := copyPrivateFileExclusive(
			filepath.Join(sourceRun, name), destination, artifactDigests[name], workerArtifactLimit(name),
		)
		if copyErr != nil {
			rollback()
			return fmt.Errorf("import worker artifact %s: %w", name, copyErr)
		}
		created = append(created, destination)
	}
	if err := syncEvaluationDirectory(destinationCAS, "evaluation CAS"); err != nil {
		rollback()
		return err
	}
	if err := syncEvaluationDirectory(destinationRun, "evaluation run evidence"); err != nil {
		rollback()
		return err
	}
	return nil
}

func (staging *workerStaging) validateEvidenceManifests(destinationRun string) error {
	realManifest, err := readBundleFile(filepath.Join(destinationRun, manifestFileName))
	if err != nil {
		return fmt.Errorf("read destination run manifest: %w", err)
	}
	stagedManifest, err := readBundleFile(staging.manifestPath)
	if err != nil {
		return fmt.Errorf("read worker run manifest: %w", err)
	}
	if !bytes.Equal(realManifest, staging.manifestBytes) || !bytes.Equal(stagedManifest, staging.manifestBytes) {
		return fmt.Errorf("worker or destination run manifest changed during execution")
	}
	return nil
}

func discoverWorkerRunArtifacts(sourceRun string) (map[string]bool, error) {
	entries, err := os.ReadDir(sourceRun)
	if err != nil {
		return nil, fmt.Errorf("list worker run evidence: %w", err)
	}
	allowed := make(map[string]bool, len(workerRunArtifactNames))
	seen := make(map[string]bool, len(entries))
	for _, name := range workerRunArtifactNames {
		allowed[name] = true
	}
	for _, entry := range entries {
		if entry.IsDir() || !allowed[entry.Name()] {
			return nil, fmt.Errorf("worker produced unsupported run artifact %q", entry.Name())
		}
		seen[entry.Name()] = true
	}
	for name := range requiredWorkerRunArtifacts {
		if !seen[name] {
			return nil, fmt.Errorf("worker omitted required run artifact %q", name)
		}
	}
	return seen, nil
}

func workerArtifactLimit(name string) int64 {
	switch name {
	case "cases.jsonl", "grading-cases.jsonl", "records.jsonl", "routing-traces.jsonl":
		return maxWorkerArtifactBytes
	default:
		return maxStructuredArtifactBytes
	}
}
