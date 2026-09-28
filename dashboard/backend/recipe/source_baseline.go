package recipe

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
)

const sourceBaselineSchema = "vllm-sr/recipe-source-baseline/v1"

type sourceBaselineRecord struct {
	SchemaVersion string `json:"schema_version"`
	ConfigDigest  string `json:"config_digest"`
}

// RefreshSourceBaseline writes immutable digest-addressed config bytes before
// atomically publishing the small pointer record. Package-to-package activation
// never calls this method; each new source-to-package cycle does.
func (s *Store) RefreshSourceBaseline(config []byte) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.ensureLayout(); err != nil {
		return err
	}
	digest := digestBytes(config)
	configPath := s.sourceBaselineObjectPath(digest)
	recordPath := filepath.Join(s.root, "source-baseline.json")
	_, statErr := os.Lstat(configPath)
	if errors.Is(statErr, os.ErrNotExist) {
		if writeErr := writeFileAtomically(configPath, config, 0o600); writeErr != nil {
			return writeErr
		}
	} else if statErr != nil {
		return statErr
	} else if existing, readErr := readBoundedFile(configPath, maxConfigBytes); readErr != nil || digestBytes(existing) != digest {
		return errors.New("source Recipe baseline object is inconsistent")
	}
	record := sourceBaselineRecord{SchemaVersion: sourceBaselineSchema, ConfigDigest: digest}
	return writeJSONAtomically(recordPath, record, 0o600)
}

func (s *Store) SourceBaseline() ([]byte, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.readSourceBaselineLocked()
}

func (s *Store) readSourceBaselineLocked() ([]byte, error) {
	var record sourceBaselineRecord
	if err := readStrictJSONFile(filepath.Join(s.root, "source-baseline.json"), 64<<10, &record); err != nil {
		return nil, err
	}
	if record.SchemaVersion != sourceBaselineSchema || !validDigest(record.ConfigDigest) {
		return nil, errors.New("source Recipe baseline record is invalid")
	}
	config, err := readBoundedFile(s.sourceBaselineObjectPath(record.ConfigDigest), maxConfigBytes)
	if err != nil {
		return nil, err
	}
	if digestBytes(config) != record.ConfigDigest {
		return nil, errors.New("source Recipe baseline digest mismatch")
	}
	return config, nil
}

func (s *Store) sourceBaselineObjectPath(digest string) string {
	return filepath.Join(s.root, "source-baselines", strings.TrimPrefix(digest, "sha256:")+".yaml")
}
