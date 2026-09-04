package evaluationplane

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

var downloadableArtifactNames = map[string]bool{
	"routing-traces.jsonl":  true,
	"capacity-profile.json": true,
	"metrics.json":          true,
	"gates.json":            true,
	"comparison.json":       true,
	"failure-summary.json":  true,
	"provenance.json":       true,
	"checksums.sha256":      true,
}

type OpenedArtifact struct {
	File      *os.File
	Name      string
	MediaType string
	Size      int64
}

func (s *Store) OpenArtifact(runID, artifactPath string) (*OpenedArtifact, error) {
	runDir, err := s.checkedRunDir(runID)
	if err != nil {
		return nil, err
	}
	relative, err := cleanArtifactPath(artifactPath)
	if err != nil {
		return nil, err
	}
	if !downloadableArtifactNames[filepath.ToSlash(relative)] {
		return nil, fmt.Errorf("%w: artifact is not downloadable", ErrInvalid)
	}
	candidate := filepath.Join(runDir, relative)
	info, err := os.Lstat(candidate)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, fmt.Errorf("%w: evaluation artifact", ErrNotFound)
		}
		return nil, fmt.Errorf("stat evaluation artifact: %w", err)
	}
	if !info.Mode().IsRegular() || info.Mode()&os.ModeSymlink != 0 {
		return nil, fmt.Errorf("%w: artifact is not a private regular file", ErrInvalid)
	}
	file, err := openBundleFile(candidate, os.O_RDONLY)
	if err != nil {
		return nil, fmt.Errorf("open evaluation artifact: %w", err)
	}
	openedInfo, err := file.Stat()
	if err != nil {
		_ = file.Close()
		return nil, fmt.Errorf("stat evaluation artifact: %w", err)
	}
	return &OpenedArtifact{File: file, Name: filepath.Base(candidate), Size: openedInfo.Size()}, nil
}

func cleanArtifactPath(raw string) (string, error) {
	raw = strings.TrimSpace(strings.ReplaceAll(raw, "\\", "/"))
	if raw == "" || strings.HasPrefix(raw, "/") {
		return "", fmt.Errorf("%w: invalid artifact path", ErrInvalid)
	}
	cleaned := filepath.Clean(filepath.FromSlash(raw))
	if cleaned == "." || cleaned == ".." || strings.HasPrefix(cleaned, ".."+string(filepath.Separator)) || filepath.IsAbs(cleaned) || filepath.ToSlash(cleaned) != raw {
		return "", fmt.Errorf("%w: invalid artifact path", ErrInvalid)
	}
	return cleaned, nil
}
