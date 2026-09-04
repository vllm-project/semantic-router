package evaluationplane

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
)

const (
	maxStructuredArtifactBytes = int64(16 * 1024 * 1024)
	maxWorkerArtifactBytes     = int64(256 * 1024 * 1024)
	maxWorkerBundleBytes       = int64(1024 * 1024 * 1024)
)

var casObjectNamePattern = regexp.MustCompile(`^[0-9a-f]{64}$`)

// Evidence publication is serialized because canonical CAS objects are shared
// across runs. This keeps validation, publication, and rollback atomic when
// concurrent workers produce the same object digest.
var workerEvidenceImportMu sync.Mutex

var workerRunArtifactNames = []string{
	manifestFileName,
	"events.jsonl",
	"cases.jsonl",
	"grading-cases.jsonl",
	"records.jsonl",
	"metrics.json",
	"gates.json",
	"lineage.json",
	"provenance.json",
	"failure-cases.jsonl",
	"failure-summary.json",
	"routing-traces.jsonl",
	"capacity-profile.json",
	"report.md",
	"report.html",
	"checksums.sha256",
	"private-checksums.sha256",
	reportFileName,
}

var requiredWorkerRunArtifacts = map[string]bool{
	manifestFileName:           true,
	"events.jsonl":             true,
	"cases.jsonl":              true,
	"grading-cases.jsonl":      true,
	"records.jsonl":            true,
	"metrics.json":             true,
	"gates.json":               true,
	"lineage.json":             true,
	"provenance.json":          true,
	"failure-cases.jsonl":      true,
	"failure-summary.json":     true,
	"report.md":                true,
	"report.html":              true,
	"checksums.sha256":         true,
	"private-checksums.sha256": true,
	reportFileName:             true,
}

type workerStaging struct {
	root             string
	storePath        string
	manifestPath     string
	destinationStore string
	runID            string
	manifestBytes    []byte
}

func prepareWorkerStaging(spec ProcessSpec) (*workerStaging, error) {
	manifestPath, err := filepath.Abs(spec.ManifestPath)
	if err != nil {
		return nil, fmt.Errorf("resolve worker manifest: %w", err)
	}
	storePath, err := filepath.Abs(spec.StorePath)
	if err != nil {
		return nil, fmt.Errorf("resolve worker store: %w", err)
	}
	if validationErr := requirePrivateDirectory(storePath); validationErr != nil {
		return nil, fmt.Errorf("validate worker destination store: %w", validationErr)
	}
	manifest, raw, err := readRunManifestStrict(manifestPath)
	if err != nil {
		return nil, fmt.Errorf("read worker manifest: %w", err)
	}
	expectedManifest := filepath.Join(storePath, "runs", manifest.RunID, manifestFileName)
	if manifestPath != expectedManifest {
		return nil, fmt.Errorf("worker manifest is outside its canonical run bundle")
	}

	root, err := os.MkdirTemp("", "vllm-sr-evaluation-worker-")
	if err != nil {
		return nil, fmt.Errorf("create private worker staging root: %w", err)
	}
	staging := &workerStaging{
		root: root, storePath: filepath.Join(root, "store"), destinationStore: storePath,
		runID: manifest.RunID, manifestBytes: raw,
	}
	if err := staging.initialize(); err != nil {
		staging.cleanup()
		return nil, err
	}
	return staging, nil
}

func (staging *workerStaging) initialize() error {
	if err := os.Chmod(staging.root, 0o700); err != nil {
		return fmt.Errorf("protect worker staging root: %w", err)
	}
	paths := []string{
		staging.storePath,
		filepath.Join(staging.storePath, "runs"),
		filepath.Join(staging.storePath, "runs", staging.runID),
	}
	for _, path := range paths {
		if err := os.Mkdir(path, 0o700); err != nil {
			return fmt.Errorf("create worker staging directory: %w", err)
		}
		if err := requirePrivateDirectory(path); err != nil {
			return fmt.Errorf("protect worker staging directory: %w", err)
		}
	}
	staging.manifestPath = filepath.Join(paths[len(paths)-1], manifestFileName)
	//nolint:gosec // The canonical path is constructed under a private mktemp root.
	file, err := os.OpenFile(staging.manifestPath, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o600)
	if err != nil {
		return fmt.Errorf("stage immutable worker manifest: %w", err)
	}
	if _, err = file.Write(staging.manifestBytes); err == nil {
		err = file.Sync()
	}
	closeErr := file.Close()
	if err != nil {
		return fmt.Errorf("write immutable worker manifest: %w", err)
	}
	if closeErr != nil {
		return fmt.Errorf("close immutable worker manifest: %w", closeErr)
	}
	return nil
}

func (staging *workerStaging) cleanup() {
	if staging != nil && staging.root != "" {
		_ = os.RemoveAll(staging.root)
	}
}

func (staging *workerStaging) importEvidence() error {
	return staging.importEvidenceWithBudget(maxWorkerBundleBytes)
}

func (staging *workerStaging) importEvidenceWithBudget(bundleLimit int64) error {
	workerEvidenceImportMu.Lock()
	defer workerEvidenceImportMu.Unlock()
	return staging.importEvidenceUnlocked(bundleLimit)
}

func (staging *workerStaging) importEvidenceUnlocked(bundleLimit int64) error {
	destinationRun := filepath.Join(staging.destinationStore, "runs", staging.runID)
	if err := requirePrivateDirectory(destinationRun); err != nil {
		return fmt.Errorf("validate destination run bundle: %w", err)
	}
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

	sourceRun := filepath.Join(staging.storePath, "runs", staging.runID)
	entries, err := os.ReadDir(sourceRun)
	if err != nil {
		return fmt.Errorf("list worker run evidence: %w", err)
	}
	allowed := make(map[string]bool, len(workerRunArtifactNames))
	seen := make(map[string]bool, len(entries))
	for _, name := range workerRunArtifactNames {
		allowed[name] = true
	}
	for _, entry := range entries {
		if entry.IsDir() || !allowed[entry.Name()] {
			return fmt.Errorf("worker produced unsupported run artifact %q", entry.Name())
		}
		seen[entry.Name()] = true
	}
	for name := range requiredWorkerRunArtifacts {
		if !seen[name] {
			return fmt.Errorf("worker omitted required run artifact %q", name)
		}
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
	created := make([]string, 0, len(casPlan)+len(seen))
	rollback := func() {
		for index := len(created) - 1; index >= 0; index-- {
			_ = os.Remove(created[index])
		}
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
	return nil
}

func workerCASReferences(sourceRun string, seen map[string]bool) (map[string]string, map[string]bool, int64, error) {
	artifactDigests := make(map[string]string, len(seen))
	references := make(map[string]bool, len(seen))
	var total int64
	for _, name := range workerRunArtifactNames {
		if !seen[name] {
			continue
		}
		digest, size, err := privateFileHexDigest(filepath.Join(sourceRun, name), workerArtifactLimit(name))
		if err != nil {
			return nil, nil, 0, fmt.Errorf("verify worker artifact %s: %w", name, err)
		}
		artifactDigests[name] = digest
		total += size
		if total > maxWorkerBundleBytes {
			return nil, nil, 0, fmt.Errorf("worker evidence bundle exceeds the import limit")
		}
		if name != "events.jsonl" {
			references[digest] = true
		}
	}
	lineage, err := readEvidenceBytes(filepath.Join(sourceRun, "lineage.json"), maxStructuredArtifactBytes)
	if err != nil {
		return nil, nil, 0, fmt.Errorf("read worker lineage CAS references: %w", err)
	}
	if _, lineageErr := resolvedLineage(lineage); lineageErr != nil {
		return nil, nil, 0, lineageErr
	}
	value, err := decodeJSONValue(lineage)
	if err != nil {
		return nil, nil, 0, err
	}
	collectCASReferences(value, references)
	return artifactDigests, references, total, nil
}

func collectCASReferences(value any, references map[string]bool) {
	switch typed := value.(type) {
	case map[string]any:
		for key, child := range typed {
			if key == "digest" {
				if digest, ok := child.(string); ok && digestPattern.MatchString(digest) {
					references[strings.TrimPrefix(digest, "sha256:")] = true
				}
			}
			collectCASReferences(child, references)
		}
	case []any:
		for _, child := range typed {
			collectCASReferences(child, references)
		}
	}
}

type casImport struct {
	source      string
	destination string
	digest      string
	exists      bool
}

func (staging *workerStaging) planCASObjects(references map[string]bool) ([]casImport, int64, error) {
	source := filepath.Join(staging.storePath, "objects", "sha256")
	destination := filepath.Join(staging.destinationStore, "objects", "sha256")
	if err := requirePrivateDirectory(source); err != nil {
		return nil, 0, fmt.Errorf("validate worker CAS: %w", err)
	}
	if err := requirePrivateDirectory(destination); err != nil {
		return nil, 0, fmt.Errorf("validate destination CAS: %w", err)
	}
	entries, err := os.ReadDir(source)
	if err != nil {
		return nil, 0, fmt.Errorf("list worker CAS: %w", err)
	}
	seen := make(map[string]bool, len(entries))
	plan := make([]casImport, 0, len(entries))
	var total int64
	for _, entry := range entries {
		if entry.IsDir() || !casObjectNamePattern.MatchString(entry.Name()) || !references[entry.Name()] {
			return nil, 0, fmt.Errorf("worker produced an invalid or unreferenced CAS object")
		}
		seen[entry.Name()] = true
		sourcePath := filepath.Join(source, entry.Name())
		_, size, verifyErr := privateFileHexDigest(sourcePath, maxWorkerArtifactBytes)
		if verifyErr != nil {
			return nil, 0, fmt.Errorf("validate worker CAS object: %w", verifyErr)
		}
		total += size
		if total > maxWorkerBundleBytes {
			return nil, 0, fmt.Errorf("worker CAS exceeds the import limit")
		}
		if verifyErr := verifyPrivateFileDigest(sourcePath, entry.Name()); verifyErr != nil {
			return nil, 0, fmt.Errorf("validate worker CAS object: %w", verifyErr)
		}
		target := filepath.Join(destination, entry.Name())
		exists := false
		if _, statErr := os.Lstat(target); statErr == nil {
			if verifyErr := verifyPrivateFileDigest(target, entry.Name()); verifyErr != nil {
				return nil, 0, fmt.Errorf("validate existing CAS object: %w", verifyErr)
			}
			exists = true
		} else if !os.IsNotExist(statErr) {
			return nil, 0, fmt.Errorf("stat destination CAS object: %w", statErr)
		}
		plan = append(plan, casImport{source: sourcePath, destination: target, digest: entry.Name(), exists: exists})
	}
	for digest := range references {
		if !seen[digest] {
			return nil, 0, fmt.Errorf("worker CAS omitted a referenced object")
		}
	}
	return plan, total, nil
}

func copyPrivateFileExclusive(source, destination, expectedHexDigest string, limit int64) (int64, error) {
	input, err := openBundleFile(source, os.O_RDONLY)
	if err != nil {
		return 0, err
	}
	defer func() { _ = input.Close() }()
	info, err := input.Stat()
	if err != nil {
		return 0, err
	}
	if info.Size() > limit {
		return 0, fmt.Errorf("artifact exceeds the per-file import limit")
	}
	temporary, err := os.CreateTemp(filepath.Dir(destination), ".tmp-worker-import-*")
	if err != nil {
		return 0, err
	}
	temporaryPath := temporary.Name()
	defer func() { _ = os.Remove(temporaryPath) }()
	if err := temporary.Chmod(0o600); err != nil {
		_ = temporary.Close()
		return 0, err
	}
	hash := sha256.New()
	written, copyErr := io.Copy(io.MultiWriter(temporary, hash), io.LimitReader(input, limit+1))
	if copyErr == nil && written > limit {
		copyErr = fmt.Errorf("artifact exceeds the per-file import limit")
	}
	if copyErr == nil && expectedHexDigest != "" && fmt.Sprintf("%x", hash.Sum(nil)) != expectedHexDigest {
		copyErr = fmt.Errorf("CAS object content does not match its identity")
	}
	if copyErr == nil {
		copyErr = temporary.Sync()
	}
	closeErr := temporary.Close()
	if copyErr != nil {
		return 0, copyErr
	}
	if closeErr != nil {
		return 0, closeErr
	}
	if err := os.Link(temporaryPath, destination); err != nil {
		return 0, err
	}
	return written, nil
}

func verifyPrivateFileDigest(path, expectedHexDigest string) error {
	digest, _, err := privateFileHexDigest(path, maxWorkerArtifactBytes)
	if err != nil {
		return err
	}
	if digest != expectedHexDigest {
		return fmt.Errorf("private file digest mismatch")
	}
	return nil
}

func privateFileHexDigest(path string, limit int64) (string, int64, error) {
	file, err := openBundleFile(path, os.O_RDONLY)
	if err != nil {
		return "", 0, err
	}
	defer func() { _ = file.Close() }()
	hash := sha256.New()
	written, err := io.Copy(hash, io.LimitReader(file, limit+1))
	if err != nil {
		return "", 0, err
	}
	if written > limit {
		return "", 0, fmt.Errorf("private file exceeds the per-file limit")
	}
	return fmt.Sprintf("%x", hash.Sum(nil)), written, nil
}

func workerArtifactLimit(name string) int64 {
	switch name {
	case "events.jsonl", "cases.jsonl", "grading-cases.jsonl", "records.jsonl",
		"failure-cases.jsonl", "routing-traces.jsonl":
		return maxWorkerArtifactBytes
	default:
		return maxStructuredArtifactBytes
	}
}

func readRunManifestStrict(path string) (RunManifest, []byte, error) {
	data, err := readBundleFile(path)
	if err != nil {
		return RunManifest{}, nil, err
	}
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	var manifest RunManifest
	if decodeErr := decoder.Decode(&manifest); decodeErr != nil {
		return RunManifest{}, nil, fmt.Errorf("decode evaluation manifest: %w", decodeErr)
	}
	if trailingErr := ensureJSONEOF(decoder); trailingErr != nil {
		return RunManifest{}, nil, trailingErr
	}
	if manifest.SchemaVersion != SchemaVersion || manifest.Target.SchemaVersion != SchemaVersion {
		return RunManifest{}, nil, fmt.Errorf("evaluation manifest schema_version must be %q", SchemaVersion)
	}
	if validationErr := validateResourceID(manifest.RunID); validationErr != nil {
		return RunManifest{}, nil, validationErr
	}
	if !digestPattern.MatchString(manifest.PolicySnapshotDigest) {
		return RunManifest{}, nil, fmt.Errorf("evaluation manifest policy_snapshot_digest is invalid")
	}
	if !digestPattern.MatchString(manifest.ManifestDigest) {
		return RunManifest{}, nil, fmt.Errorf("evaluation manifest manifest_digest is invalid")
	}
	recomputedManifestDigest, err := manifestSemanticDigest(manifest)
	if err != nil || recomputedManifestDigest != manifest.ManifestDigest {
		return RunManifest{}, nil, fmt.Errorf("evaluation manifest manifest_digest does not match its server-owned semantic value")
	}
	if !sourceRevisionPattern.MatchString(manifest.CodeRevision) {
		return RunManifest{}, nil, fmt.Errorf("evaluation manifest code_revision is not immutable")
	}
	if strings.TrimSpace(manifest.GateContractVersion) == "" || !validSuiteRevisionSnapshot(manifest.SuiteIDs, manifest.SuiteRevisions) {
		return RunManifest{}, nil, fmt.Errorf("evaluation manifest suite and gate contract revisions are invalid")
	}
	if err := validateTargetContract(
		manifest.Target.RouterAPIKey,
		manifest.Target.EnvoyAPIKey,
		manifest.Target.ModelArms,
		manifest.Target.BackendTopologyDigest,
	); err != nil {
		return RunManifest{}, nil, fmt.Errorf("evaluation manifest target is invalid: %w", err)
	}
	return manifest, data, nil
}

func validSuiteRevisionSnapshot(suiteIDs []string, revisions map[string]string) bool {
	if len(suiteIDs) == 0 || len(revisions) != len(suiteIDs) {
		return false
	}
	seen := make(map[string]bool, len(suiteIDs))
	for _, suiteID := range suiteIDs {
		revision, ok := revisions[suiteID]
		if suiteID == "" || seen[suiteID] || !ok || strings.TrimSpace(revision) == "" || len(revision) > 160 {
			return false
		}
		seen[suiteID] = true
	}
	return true
}
