package modeldownload

import (
	"encoding/hex"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// ValidateReloadArtifacts prevents provisioning from modifying directories used
// by the live generation. DownloadModel additionally rejects reusing populated
// explicit artifact paths without matching immutable HF cache metadata, which
// also protects retired generations still draining requests.
func ValidateReloadArtifacts(current, next *config.RouterConfig) error {
	if current == nil {
		return nil
	}
	previous, err := BuildModelSpecs(current)
	if err != nil {
		return err
	}
	candidate, err := BuildModelSpecs(next)
	if err != nil {
		return err
	}
	live := map[string]ModelSpec{}
	for _, spec := range previous {
		live[artifactDirectory(spec.LocalPath)] = spec
	}
	missing, err := GetMissingModels(candidate)
	if err != nil {
		return err
	}
	refresh := map[string]bool{}
	for _, spec := range missing {
		refresh[artifactDirectory(spec.LocalPath)] = true
	}
	for _, spec := range candidate {
		path := artifactDirectory(spec.LocalPath)
		old, exists := live[path]
		if !exists {
			continue
		}
		if old.Revision != spec.Revision || refresh[path] {
			return fmt.Errorf("artifact %q is in use and cannot be refreshed during reload; choose a separate local artifact directory", spec.LocalPath)
		}
	}
	return nil
}

func artifactDirectory(path string) string {
	absolute, err := filepath.Abs(path)
	if err != nil {
		return filepath.Clean(path)
	}
	if resolved, err := filepath.EvalSymlinks(absolute); err == nil {
		return resolved
	}
	return absolute
}

func immutableRevision(revision string) bool {
	if len(revision) != 40 {
		return false
	}
	_, err := hex.DecodeString(revision)
	return err == nil
}

// cachedRevisionMatches reads the HF client's existing commit/etag/timestamp
// metadata. No router receipt or registration requirement is introduced.
func cachedRevisionMatches(spec ModelSpec) (bool, error) {
	files, err := snapshotFiles(spec.LocalPath)
	if err != nil {
		return false, err
	}
	if len(files) == 0 || !immutableRevision(spec.Revision) {
		return false, nil
	}
	for _, path := range files {
		relative, err := filepath.Rel(spec.LocalPath, path)
		if err != nil {
			return false, err
		}
		data, err := os.ReadFile(filepath.Join(spec.LocalPath, ".cache", "huggingface", "download", relative+".metadata"))
		if os.IsNotExist(err) {
			return false, nil
		}
		if err != nil {
			return false, err
		}
		lines := strings.Split(strings.TrimSpace(string(data)), "\n")
		if len(lines) < 3 || !strings.EqualFold(strings.TrimSpace(lines[0]), spec.Revision) || strings.TrimSpace(lines[1]) == "" {
			return false, nil
		}
		timestamp, err := strconv.ParseFloat(strings.TrimSpace(lines[2]), 64)
		if err != nil {
			return false, nil
		}
		info, err := os.Stat(path)
		if err != nil {
			return false, err
		}
		if float64(info.ModTime().UnixNano())/1e9 > timestamp {
			return false, nil
		}
	}
	return true, nil
}

func snapshotFiles(root string) ([]string, error) {
	var files []string
	err := filepath.WalkDir(root, func(path string, entry fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if entry.IsDir() {
			if path != root && (entry.Name() == ".cache" || entry.Name() == ".git") {
				return fs.SkipDir
			}
			return nil
		}
		files = append(files, path)
		return nil
	})
	if os.IsNotExist(err) {
		return nil, nil
	}
	return files, err
}

func validateArtifactDownload(spec ModelSpec) error {
	if !spec.Strict {
		return nil
	}
	files, err := snapshotFiles(spec.LocalPath)
	if err != nil {
		return err
	}
	if len(files) == 0 {
		return nil
	}
	matches, err := cachedRevisionMatches(spec)
	if err != nil {
		return err
	}
	if !matches {
		return fmt.Errorf("artifact %q already contains files without matching immutable revision %q; use a separate empty artifact directory to preserve existing generations", spec.LocalPath, spec.Revision)
	}
	return nil
}
