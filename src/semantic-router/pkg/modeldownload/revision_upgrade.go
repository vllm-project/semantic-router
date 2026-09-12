package modeldownload

import (
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
)

// HF local-dir downloads retain files deleted in a later revision. Archive only
// unchanged, HF-managed ONNX exports from older immutable revisions. Native
// weights, tokenizer files, local additions and edits are never retired here.
// Verification must succeed without the archived exports, otherwise restore
// them before returning the download error.
func recordDownloadedModelRevision(spec ModelSpec) (result error) {
	if !immutableModelRevision(spec) {
		return recordModelRevision(spec)
	}
	// An offline CLI may report success without refreshing anything. Establish
	// the requested revision for every required artifact before moving files.
	required := spec.RequiredFiles
	if len(required) == 0 {
		required = DefaultRequiredFiles
	}
	for _, name := range required {
		if !safeRevisionArtifactPath(name) {
			return fmt.Errorf("%w: invalid required artifact path %q", errUnverifiedModelRevision, name)
		}
		etag, err := hfArtifactETag(spec, name)
		if err != nil {
			return err
		}
		if err := verifyHFArtifactBytes(filepath.Join(spec.LocalPath, name), etag); err != nil {
			return fmt.Errorf("%w: %s: %w", errUnverifiedModelRevision, name, err)
		}
	}
	stale, err := supersededONNXArtifacts(spec)
	if err != nil {
		return err
	}
	if len(stale) == 0 {
		return recordModelRevision(spec)
	}
	archive, err := os.MkdirTemp(spec.LocalPath, ".router-superseded-")
	if err != nil {
		return err
	}
	moved := make([]string, 0, len(stale))
	defer func() {
		if result == nil {
			return
		}
		for i := len(moved) - 1; i >= 0; i-- {
			name := moved[i]
			if err := os.Rename(filepath.Join(archive, name), filepath.Join(spec.LocalPath, name)); err != nil {
				result = errors.Join(result, fmt.Errorf("restore superseded artifact %s: %w", name, err))
			}
		}
	}()
	for _, name := range stale {
		destination := filepath.Join(archive, name)
		if err := os.MkdirAll(filepath.Dir(destination), 0o700); err != nil {
			return err
		}
		if err := os.Rename(filepath.Join(spec.LocalPath, name), destination); err != nil {
			return err
		}
		moved = append(moved, name)
	}
	return recordModelRevision(spec)
}

func supersededONNXArtifacts(spec ModelSpec) ([]string, error) {
	root, err := filepath.EvalSymlinks(spec.LocalPath)
	if err != nil {
		return nil, err
	}
	var stale []string
	err = filepath.WalkDir(root, func(path string, entry fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if path == root {
			return nil
		}
		if strings.HasPrefix(entry.Name(), ".") {
			if entry.IsDir() {
				return filepath.SkipDir
			}
			return nil
		}
		// Our portable and accelerated exports use .onnx and .onnx.data.
		// Other formats remain subject to ordinary revision verification;
		// optional new variants must not replace missing/stale native weights.
		if entry.IsDir() || (!strings.HasSuffix(entry.Name(), ".onnx") && !strings.HasSuffix(entry.Name(), ".onnx.data")) {
			return nil
		}
		relative, err := filepath.Rel(root, path)
		if err != nil {
			return err
		}
		name := filepath.ToSlash(relative)
		if revisionArtifactExcluded(name, spec.ExcludePatterns) {
			return nil
		}
		metadata := filepath.Join(root, ".cache/huggingface/download", relative+".metadata")
		data, err := os.ReadFile(metadata)
		if os.IsNotExist(err) {
			return nil // An unmanaged file must remain visible to revision validation.
		}
		if err != nil {
			return err
		}
		fields := strings.Fields(string(data))
		if len(fields) != 3 || strings.EqualFold(fields[0], spec.Revision) {
			return nil // The ordinary verifier handles malformed/current metadata.
		}
		previous := spec
		previous.Revision = fields[0]
		if !immutableModelRevision(previous) {
			return nil
		}
		etag, err := hfArtifactETag(previous, name)
		if err != nil {
			return err
		}
		if err := verifyHFArtifactBytes(path, etag); err != nil {
			return fmt.Errorf("%w: modified superseded artifact %s: %w", errUnverifiedModelRevision, name, err)
		}
		stale = append(stale, relative)
		return nil
	})
	return stale, err
}
