package evaluationplane

import (
	"crypto/sha256"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
)

const maxSealedEvidenceFiles = 4096

type sealedEvidenceFile struct {
	Scope       string `json:"scope"`
	Name        string `json:"name"`
	Digest      string `json:"digest"`
	SizeBytes   int64  `json:"size_bytes"`
	FileVersion string `json:"file_version"`
}

func (s *Service) buildSealedEvidenceSnapshot(runID string, checksums map[string]string) ([]sealedEvidenceFile, error) {
	runDir, err := s.store.checkedRunDir(runID)
	if err != nil {
		return nil, err
	}
	if len(checksums) == 0 || len(checksums) > maxSealedEvidenceFiles {
		return nil, fmt.Errorf("%w: private evidence set is empty or too large", ErrInvalid)
	}

	entries := make([]sealedEvidenceFile, 0, len(checksums)*2)
	casDigests := make(map[string]bool)
	names := make([]string, 0, len(checksums))
	for name := range checksums {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		hexDigest := checksums[name]
		entry, sealErr := sealEvidenceFile(
			"run", name, filepath.Join(runDir, name), "sha256:"+hexDigest, workerArtifactLimit(name),
		)
		if sealErr != nil {
			return nil, fmt.Errorf("seal run evidence %s: %w", name, sealErr)
		}
		entries = append(entries, entry)
	}
	if lineageDigest := checksums["lineage.json"]; lineageDigest != "" {
		lineage, readErr := readEvidenceBytes(filepath.Join(runDir, "lineage.json"), maxStructuredArtifactBytes)
		if readErr != nil || strings.TrimPrefix(digestBytes(lineage), "sha256:") != lineageDigest {
			return nil, fmt.Errorf("%w: sealed lineage evidence changed", ErrInvalid)
		}
		value, decodeErr := decodeJSONValue(lineage)
		if decodeErr != nil {
			return nil, decodeErr
		}
		collectCASReferences(value, casDigests)
	}
	if len(entries)+len(casDigests) > maxSealedEvidenceFiles {
		return nil, fmt.Errorf("%w: sealed evidence set is too large", ErrInvalid)
	}
	casNames := make([]string, 0, len(casDigests))
	for digest := range casDigests {
		casNames = append(casNames, digest)
	}
	sort.Strings(casNames)
	for _, digest := range casNames {
		if !casObjectNamePattern.MatchString(digest) {
			return nil, fmt.Errorf("%w: sealed CAS identity is invalid", ErrInvalid)
		}
		entry, sealErr := sealEvidenceFile(
			"cas", digest, filepath.Join(s.store.root, "objects", "sha256", digest), "sha256:"+digest, maxWorkerArtifactBytes,
		)
		if sealErr != nil {
			return nil, fmt.Errorf("seal CAS evidence %s: %w", digest, sealErr)
		}
		entries = append(entries, entry)
	}
	sort.Slice(entries, func(left, right int) bool {
		return entries[left].Scope+":"+entries[left].Name < entries[right].Scope+":"+entries[right].Name
	})
	return entries, nil
}

func (s *Service) verifySealedEvidenceSnapshot(runID string, entries []sealedEvidenceFile, receipt []byte) error {
	if err := validateSealedEvidenceMetadata(entries); err != nil {
		return err
	}
	runDir, err := s.store.checkedRunDir(runID)
	if err != nil {
		return err
	}
	allowed := make(map[string]bool, len(workerRunArtifactNames))
	for _, name := range workerRunArtifactNames {
		allowed[name] = true
	}
	checksums, err := parseChecksumReceipt(receipt, allowed)
	if err != nil {
		return fmt.Errorf("%w: sealed private checksum receipt is invalid", ErrInvalid)
	}
	runEntries := make(map[string]sealedEvidenceFile)
	casEntries := make(map[string]sealedEvidenceFile)
	for _, entry := range entries {
		if entry.Scope == "run" {
			runEntries[entry.Name] = entry
		} else {
			casEntries[entry.Name] = entry
		}
	}
	if len(runEntries) != len(checksums) {
		return fmt.Errorf("%w: sealed run evidence set changed", ErrInvalid)
	}
	casDigests := make(map[string]bool)
	for name, hexDigest := range checksums {
		entry, ok := runEntries[name]
		if !ok || entry.Digest != "sha256:"+hexDigest {
			return fmt.Errorf("%w: sealed run evidence receipt changed", ErrInvalid)
		}
		if err := verifyEvidenceFileMetadata(entry, filepath.Join(runDir, name)); err != nil {
			return err
		}
	}
	if err := verifyRunEvidenceSet(runDir, runEntries); err != nil {
		return err
	}
	if lineageDigest := checksums["lineage.json"]; lineageDigest != "" {
		lineage, readErr := readEvidenceBytes(filepath.Join(runDir, "lineage.json"), maxStructuredArtifactBytes)
		if readErr != nil || strings.TrimPrefix(digestBytes(lineage), "sha256:") != lineageDigest {
			return fmt.Errorf("%w: sealed lineage evidence changed", ErrInvalid)
		}
		value, decodeErr := decodeJSONValue(lineage)
		if decodeErr != nil {
			return decodeErr
		}
		collectCASReferences(value, casDigests)
	}
	if len(casEntries) != len(casDigests) {
		return fmt.Errorf("%w: sealed CAS evidence set changed", ErrInvalid)
	}
	for digest := range casDigests {
		entry, ok := casEntries[digest]
		if !ok || entry.Digest != "sha256:"+digest {
			return fmt.Errorf("%w: sealed CAS evidence receipt changed", ErrInvalid)
		}
		if err := verifyEvidenceFileMetadata(entry, filepath.Join(s.store.root, "objects", "sha256", digest)); err != nil {
			return err
		}
	}
	return nil
}

func sealEvidenceFile(scope, name, path, expectedDigest string, limit int64) (sealedEvidenceFile, error) {
	file, err := openBundleFile(path, os.O_RDONLY)
	if err != nil {
		return sealedEvidenceFile{}, err
	}
	defer func() { _ = file.Close() }()
	before, err := file.Stat()
	if err != nil || before.Size() > limit {
		return sealedEvidenceFile{}, fmt.Errorf("evidence file exceeds its limit")
	}
	hash := sha256.New()
	written, err := io.Copy(hash, io.LimitReader(file, limit+1))
	if err != nil {
		return sealedEvidenceFile{}, fmt.Errorf("hash sealed evidence: %w", err)
	}
	if written > limit {
		return sealedEvidenceFile{}, fmt.Errorf("evidence file exceeds its limit")
	}
	after, err := file.Stat()
	if err != nil || !os.SameFile(before, after) || bundleFileVersion(before) != bundleFileVersion(after) {
		return sealedEvidenceFile{}, fmt.Errorf("evidence file changed while sealing")
	}
	digest := fmt.Sprintf("sha256:%x", hash.Sum(nil))
	if written != after.Size() || digest != expectedDigest {
		return sealedEvidenceFile{}, fmt.Errorf("evidence file does not match its receipt")
	}
	return sealedEvidenceFile{
		Scope: scope, Name: name, Digest: digest, SizeBytes: written, FileVersion: bundleFileVersion(after),
	}, nil
}

func verifyEvidenceFileMetadata(expected sealedEvidenceFile, path string) error {
	file, err := openBundleFile(path, os.O_RDONLY)
	if err != nil {
		return fmt.Errorf("%w: sealed evidence file is unavailable", ErrInvalid)
	}
	defer func() { _ = file.Close() }()
	info, err := file.Stat()
	limit := maxWorkerArtifactBytes
	if expected.Scope == "run" {
		limit = workerArtifactLimit(expected.Name)
	}
	if err != nil || expected.SizeBytes > limit || info.Size() != expected.SizeBytes {
		return fmt.Errorf("%w: sealed evidence file changed after publication", ErrInvalid)
	}
	if bundleFileVersion(info) == expected.FileVersion {
		return nil
	}

	// Container startup may repair an unchanged private tree's ownership and
	// permissions, which legitimately changes filesystem version metadata. Old
	// anchors also include ctime in FileVersion. Fall back to the sealed content
	// digest so those durable reports survive a restart without weakening the
	// evidence identity check.
	hash := sha256.New()
	written, hashErr := io.Copy(hash, io.LimitReader(file, limit+1))
	if hashErr != nil || written != expected.SizeBytes || written > limit ||
		fmt.Sprintf("sha256:%x", hash.Sum(nil)) != expected.Digest {
		return fmt.Errorf("%w: sealed evidence file changed after publication", ErrInvalid)
	}
	return nil
}

func verifyRunEvidenceSet(runDir string, expected map[string]sealedEvidenceFile) error {
	excluded := map[string]bool{"events.jsonl": true, privateChecksumArtifactName: true, reportFileName: true}
	for _, name := range workerRunArtifactNames {
		if excluded[name] {
			continue
		}
		_, err := os.Lstat(filepath.Join(runDir, name))
		if err == nil && expected[name].Name == "" {
			return fmt.Errorf("%w: unsealed evidence file appeared after publication", ErrInvalid)
		}
		if err != nil && !os.IsNotExist(err) {
			return fmt.Errorf("stat sealed evidence file: %w", err)
		}
	}
	return nil
}

func validateSealedEvidenceMetadata(entries []sealedEvidenceFile) error {
	if len(entries) == 0 || len(entries) > maxSealedEvidenceFiles {
		return fmt.Errorf("%w: report anchor evidence set is invalid", ErrInvalid)
	}
	last := ""
	for _, entry := range entries {
		key := entry.Scope + ":" + entry.Name
		validName := entry.Scope == "run" && filepath.Base(entry.Name) == entry.Name
		if entry.Scope == "cas" {
			validName = casObjectNamePattern.MatchString(entry.Name) && entry.Digest == "sha256:"+entry.Name
		}
		if !validName || !digestPattern.MatchString(entry.Digest) || !digestPattern.MatchString(entry.FileVersion) ||
			entry.SizeBytes < 0 || (last != "" && key <= last) {
			return fmt.Errorf("%w: report anchor evidence metadata is invalid", ErrInvalid)
		}
		last = key
	}
	return nil
}

func bundleFileVersion(info os.FileInfo) string {
	parts := []string{
		fmt.Sprintf("size=%d", info.Size()),
		fmt.Sprintf("mode=%d", info.Mode()),
		fmt.Sprintf("mtime=%d", info.ModTime().UnixNano()),
	}
	value := reflect.Indirect(reflect.ValueOf(info.Sys()))
	if value.IsValid() && value.Kind() == reflect.Struct {
		// Ownership repair changes ctime even when bytes, mode, mtime, inode, and
		// device are unchanged. Keep only metadata stable across a normal pod
		// restart; content is independently anchored by its SHA-256 digest.
		for _, name := range []string{"Dev", "Ino", "FileIndexHigh", "FileIndexLow"} {
			field := value.FieldByName(name)
			if field.IsValid() && field.CanInterface() {
				parts = append(parts, name+"="+fmt.Sprint(field.Interface()))
			}
		}
	}
	return digestString(strings.Join(parts, "\n"))
}
