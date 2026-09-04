package evaluationplane

import (
	"crypto/sha256"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
)

const privateChecksumArtifactName = "private-checksums.sha256"

func (s *Service) validatePrivateReceipt(runID string) (map[string]string, error) {
	runDir, err := s.store.checkedRunDir(runID)
	if err != nil {
		return nil, err
	}
	receipt, err := readEvidenceBytes(filepath.Join(runDir, privateChecksumArtifactName), maxStructuredArtifactBytes)
	if err != nil {
		return nil, fmt.Errorf("read private artifact checksum receipt: %w", err)
	}
	allowed := make(map[string]bool, len(workerRunArtifactNames))
	for _, name := range workerRunArtifactNames {
		allowed[name] = true
	}
	checksums, err := parseChecksumReceipt(receipt, allowed)
	if err != nil {
		return nil, fmt.Errorf("%w: private artifact checksum receipt is invalid", ErrInvalid)
	}
	excluded := map[string]bool{
		"events.jsonl": true, privateChecksumArtifactName: true, reportFileName: true,
	}
	expected := make(map[string]bool)
	for _, name := range workerRunArtifactNames {
		if excluded[name] {
			continue
		}
		_, statErr := os.Lstat(filepath.Join(runDir, name))
		if statErr == nil {
			expected[name] = true
			continue
		}
		if !os.IsNotExist(statErr) {
			return nil, fmt.Errorf("stat evidence artifact %s: %w", name, statErr)
		}
	}
	if len(checksums) != len(expected) {
		return nil, fmt.Errorf("%w: private checksum set does not match the run bundle", ErrInvalid)
	}
	for name := range expected {
		digest, ok := checksums[name]
		if !ok {
			return nil, fmt.Errorf("%w: private checksum receipt omits %s", ErrInvalid, name)
		}
		actual, _, digestErr := privateFileHexDigest(filepath.Join(runDir, name), workerArtifactLimit(name))
		if digestErr != nil {
			return nil, fmt.Errorf("verify evidence artifact %s: %w", name, digestErr)
		}
		if actual != digest {
			return nil, fmt.Errorf("%w: evidence artifact %s does not match its private checksum", ErrInvalid, name)
		}
	}
	return checksums, nil
}

func parseChecksumReceipt(data []byte, allowed map[string]bool) (map[string]string, error) {
	if len(data) == 0 || data[len(data)-1] != '\n' {
		return nil, fmt.Errorf("checksum receipt must end with a newline")
	}
	checksums := make(map[string]string)
	for _, line := range strings.Split(strings.TrimSuffix(string(data), "\n"), "\n") {
		digest, name, found := strings.Cut(line, "  ")
		if !found || !digestPattern.MatchString("sha256:"+digest) || filepath.Base(name) != name ||
			!allowed[name] || checksums[name] != "" {
			return nil, fmt.Errorf("checksum receipt contains an invalid row")
		}
		checksums[name] = digest
	}
	return checksums, nil
}

func readEvidenceBytes(path string, limit int64) ([]byte, error) {
	file, err := openBundleFile(path, os.O_RDONLY)
	if err != nil {
		return nil, err
	}
	defer func() { _ = file.Close() }()
	info, err := file.Stat()
	if err != nil {
		return nil, err
	}
	if info.Size() > limit {
		return nil, fmt.Errorf("evidence file exceeds its limit")
	}
	data, err := io.ReadAll(io.LimitReader(file, limit+1))
	if err != nil {
		return nil, err
	}
	if int64(len(data)) > limit {
		return nil, fmt.Errorf("evidence file exceeds its limit")
	}
	return data, nil
}

func digestAndSize(data []byte) (string, int64) {
	return fmt.Sprintf("sha256:%x", sha256.Sum256(data)), int64(len(data))
}
