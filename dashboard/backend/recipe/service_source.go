package recipe

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"unicode/utf8"
)

const runtimeConfigProvenanceSchema = "vllm-sr/runtime-config-provenance/v1"

type runtimeConfigProvenance struct {
	SchemaVersion                string `json:"schema_version"`
	SourceDigest                 string `json:"source_digest"`
	LastMaterializedActiveDigest string `json:"last_materialized_active_digest"`
}

func (s *Service) resolveDirectory() (string, string, string, error) {
	if s.store == nil {
		return s.directory, ActivationNone, "", nil
	}
	pointer, state, err := s.store.ActivationStatus()
	if state == ActivationNone && err == nil {
		return s.directory, state, "", nil
	}
	if pointer.RecipeDigest != "" {
		if directory, pathErr := s.store.ObjectDirectory(pointer.RecipeDigest); pathErr == nil {
			if state == ActivationActive && err == nil {
				return directory, state, pointer.RecipeDigest, nil
			}
			return directory, state, pointer.RecipeDigest, ErrActivationState
		}
	}
	return s.directory, state, pointer.RecipeDigest, ErrActivationState
}

func activationIssue(state string) string {
	if state == ActivationRecovering {
		return "active Recipe activation recovery is pending"
	}
	return "active Recipe and runtime configuration are inconsistent"
}

func (s *Service) validateSourceRuntimeBinding(sourceConfig []byte) error {
	_, err := s.boundRuntimeConfigDigest(sourceConfig)
	return err
}

// boundRuntimeConfigDigest verifies the CLI provenance receipt when platform
// realization makes the runtime document differ from the distributable Recipe
// source. It returns only a digest; runtime-normalized bytes and expanded
// environment values are never retained in the parsed snapshot cache.
func (s *Service) boundRuntimeConfigDigest(sourceConfig []byte) (string, error) {
	if s.runtimeConfigPath == "" || s.runtimeConfigPath == "." {
		return digestBytes(sourceConfig), nil
	}
	runtimeConfig, err := readBoundedFile(s.runtimeConfigPath, maxConfigBytes)
	if err != nil {
		return "", err
	}
	runtimeDigest := digestBytes(runtimeConfig)
	sourceDigest := digestBytes(sourceConfig)
	if runtimeDigest == sourceDigest {
		return runtimeDigest, nil
	}
	if s.activePointerBindsRuntime(sourceDigest, runtimeDigest) {
		return runtimeDigest, nil
	}
	provenancePath := strings.TrimSuffix(s.runtimeConfigPath, filepath.Ext(s.runtimeConfigPath)) + ".provenance.json"
	provenance, err := readRuntimeConfigProvenance(provenancePath)
	if err != nil {
		return "", err
	}
	if provenance.SchemaVersion != runtimeConfigProvenanceSchema ||
		provenance.SourceDigest != sourceDigest ||
		provenance.LastMaterializedActiveDigest != runtimeDigest {
		return "", errors.New("runtime config provenance does not bind this source Recipe")
	}
	return runtimeDigest, nil
}

func (s *Service) activePointerBindsRuntime(sourceDigest, runtimeDigest string) bool {
	if s.store == nil {
		return false
	}
	pointer, state, err := s.store.ActivationStatus()
	return err == nil && state == ActivationActive &&
		pointer.ConfigDigest == sourceDigest &&
		pointer.RealizedConfigDigest == runtimeDigest
}

func readRuntimeConfigProvenance(path string) (runtimeConfigProvenance, error) {
	data, err := readBoundedFile(path, 64<<10)
	if err != nil {
		return runtimeConfigProvenance{}, err
	}
	var provenance runtimeConfigProvenance
	decoder := json.NewDecoder(strings.NewReader(string(data)))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&provenance); err != nil {
		return runtimeConfigProvenance{}, err
	}
	var extra any
	if err := decoder.Decode(&extra); !errors.Is(err, io.EOF) {
		return runtimeConfigProvenance{}, errors.New("runtime config provenance contains trailing data")
	}
	return provenance, nil
}

func (s *Service) readFixedFiles(directory string) (map[string][]byte, map[string]FileHealth, Digests, []string) {
	files := make(map[string][]byte, len(recipeFiles))
	health := make(map[string]FileHealth, len(recipeFiles))
	digests := Digests{}
	issues := []string{}
	for _, file := range recipeFiles {
		data, err := readBoundedFile(filepath.Join(directory, file.name), file.maxBytes)
		if errors.Is(err, os.ErrNotExist) {
			health[file.key] = FileHealth{Present: false}
			continue
		}
		if err != nil {
			health[file.key] = FileHealth{Present: true}
			issues = append(issues, fmt.Sprintf("%s: %s", file.name, safeFileError(err)))
			continue
		}
		digest := digestBytes(data)
		files[file.key] = data
		health[file.key] = FileHealth{Present: true, Digest: digest, SizeBytes: int64(len(data))}
		setFileDigest(&digests, file.key, digest)
	}
	return files, health, digests, issues
}

func setFileDigest(digests *Digests, key, digest string) {
	switch key {
	case "metadata":
		digests.Metadata = digest
	case "config":
		digests.Config = digest
	case "probes":
		digests.Probes = digest
	case "dsl":
		digests.DSL = digest
	case "readme":
		digests.README = digest
	}
}

func safeFileError(err error) string {
	message := err.Error()
	if errors.Is(err, os.ErrPermission) {
		return "cannot be read"
	}
	if strings.Contains(message, "exceeds ") || strings.Contains(message, "must be ") || strings.Contains(message, "must not be ") {
		return message
	}
	return "cannot be read"
}

func readBoundedFile(path string, maxBytes int64) ([]byte, error) {
	pathInfo, err := os.Lstat(path)
	if err != nil {
		return nil, err
	}
	if validationErr := validateBoundedPathInfo(pathInfo, maxBytes); validationErr != nil {
		return nil, validationErr
	}

	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer func() { _ = file.Close() }()
	info, err := file.Stat()
	if err != nil {
		return nil, err
	}
	if validationErr := validateOpenedFileInfo(pathInfo, info, maxBytes); validationErr != nil {
		return nil, validationErr
	}
	data, err := io.ReadAll(io.LimitReader(file, maxBytes+1))
	if err != nil {
		return nil, err
	}
	if err := validateBoundedFileData(data, maxBytes); err != nil {
		return nil, err
	}
	return data, nil
}

func validateBoundedPathInfo(info os.FileInfo, maxBytes int64) error {
	if info.Mode()&os.ModeSymlink != 0 {
		return errors.New("must not be a symbolic link")
	}
	if !info.Mode().IsRegular() {
		return errors.New("must be a regular file")
	}
	if info.Size() > maxBytes {
		return fmt.Errorf("exceeds %d byte limit", maxBytes)
	}
	return nil
}

func validateOpenedFileInfo(pathInfo, info os.FileInfo, maxBytes int64) error {
	if !info.Mode().IsRegular() || !os.SameFile(pathInfo, info) {
		return errors.New("must be a regular file")
	}
	if info.Size() > maxBytes {
		return fmt.Errorf("exceeds %d byte limit", maxBytes)
	}
	return nil
}

func validateBoundedFileData(data []byte, maxBytes int64) error {
	if int64(len(data)) > maxBytes {
		return fmt.Errorf("exceeds %d byte limit", maxBytes)
	}
	if !utf8.Valid(data) {
		return errors.New("must be valid UTF-8")
	}
	return nil
}
