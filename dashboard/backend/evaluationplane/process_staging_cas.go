package evaluationplane

import (
	"crypto/sha256"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"strings"
)

var casObjectNamePattern = regexp.MustCompile(`^[0-9a-f]{64}$`)

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
		references[digest] = true
	}
	lineage, err := readEvidenceBytes(filepath.Join(sourceRun, "lineage.json"), maxStructuredArtifactBytes)
	if err != nil {
		return nil, nil, 0, fmt.Errorf("read worker lineage CAS references: %w", err)
	}
	if _, lineageErr := decodeLineageDocument(lineage); lineageErr != nil {
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
	size        int64
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
		plan = append(plan, casImport{
			source: sourcePath, destination: target, digest: entry.Name(), exists: exists, size: size,
		})
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
