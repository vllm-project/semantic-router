package evaluationplane

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
)

type publicArtifactContract struct {
	Kind      string
	MediaType string
}

var publicArtifactContracts = map[string]publicArtifactContract{
	"capacity-profile.json": {Kind: "json", MediaType: "application/json"},
	"metrics.json":          {Kind: "json", MediaType: "application/json"},
	"gates.json":            {Kind: "json", MediaType: "application/json"},
	"failure-summary.json":  {Kind: "json", MediaType: "application/json"},
	"provenance.json":       {Kind: "json", MediaType: "application/json"},
	"checksums.sha256":      {Kind: "sha256", MediaType: "text/plain"},
}

type OpenedArtifact struct {
	File      io.ReadSeekCloser
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
	contract, ok := publicArtifactContracts[filepath.ToSlash(relative)]
	if !ok {
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
	return &OpenedArtifact{
		File: file, Name: filepath.Base(candidate), MediaType: contract.MediaType, Size: openedInfo.Size(),
	}, nil
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
