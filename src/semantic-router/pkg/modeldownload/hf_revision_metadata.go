package modeldownload

import (
	"crypto/sha1" //nolint:gosec // HF uses Git's SHA-1 blob identity for non-LFS files.
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"hash"
	"io"
	"io/fs"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"sort"
	"strconv"
	"strings"
)

var errUnverifiedModelRevision = errors.New("model artifacts do not establish the pinned HF revision")

func verifyHFModelRevision(spec ModelSpec) (map[string]string, error) {
	files, err := revisionArtifactFiles(spec)
	if err != nil {
		return nil, err
	}
	verified := make(map[string]string, len(files))
	hasWeights := false
	for _, name := range files {
		etag, err := hfArtifactETag(spec, name)
		if err != nil {
			return nil, err
		}
		if err := verifyHFArtifactBytes(filepath.Join(spec.LocalPath, filepath.FromSlash(name)), etag); err != nil {
			return nil, fmt.Errorf("%w: %s: %w", errUnverifiedModelRevision, name, err)
		}
		verified[name] = etag
		hasWeights = hasWeights || primaryRevisionWeight(spec, name)
	}
	if !hasWeights {
		return nil, fmt.Errorf("%w: no verified runtime weights", errUnverifiedModelRevision)
	}
	return verified, nil
}

func primaryRevisionWeight(spec ModelSpec, name string) bool {
	// A bundled LoRA or auxiliary dense head cannot establish the root model's
	// completeness. Sharded native weights are rooted by their verified index;
	// nonstandard layouts can declare their primary weights explicitly.
	if filepath.Dir(name) != "." && !strings.HasPrefix(name, "onnx/") && !slices.Contains(spec.RequiredFiles, name) {
		return false
	}
	for _, pattern := range modelWeightPatterns {
		if matched, _ := filepath.Match(pattern, filepath.Base(name)); matched {
			return true
		}
	}
	return false
}

// Validate present weights, tokenizer/config companions, external ONNX data,
// indexed shards and explicit runtime requirements. A current config/index
// cannot launder stale weights. Explicitly excluded exports are not consumed by
// this runtime and do not invalidate its native model snapshot.
func revisionArtifactFiles(spec ModelSpec) ([]string, error) {
	root, err := filepath.EvalSymlinks(spec.LocalPath)
	if err != nil {
		return nil, err
	}
	files := map[string]bool{}
	required := spec.RequiredFiles
	if len(required) == 0 {
		required = DefaultRequiredFiles
	}
	for _, name := range required {
		if !safeRevisionArtifactPath(name) {
			return nil, fmt.Errorf("%w: invalid required artifact path %q", errUnverifiedModelRevision, name)
		}
		files[filepath.ToSlash(name)] = true
	}
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
		if entry.IsDir() {
			return nil
		}
		relative, relativeErr := filepath.Rel(root, path)
		if relativeErr != nil {
			return relativeErr
		}
		name := filepath.ToSlash(relative)
		if revisionArtifactExcluded(name, spec.ExcludePatterns) && !files[name] {
			return nil
		}
		if runtimeRevisionArtifact(entry.Name()) {
			files[name] = true
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	for name := range files {
		if strings.HasSuffix(name, ".onnx") {
			locations, err := hfONNXExternalFiles(filepath.Join(root, filepath.FromSlash(name)))
			if err != nil {
				return nil, fmt.Errorf("%w: invalid ONNX artifact %s: %w", errUnverifiedModelRevision, name, err)
			}
			for _, location := range locations {
				if !safeRevisionArtifactPath(location) {
					return nil, fmt.Errorf("%w: invalid external data path in %s", errUnverifiedModelRevision, name)
				}
				files[filepath.ToSlash(filepath.Join(filepath.Dir(name), location))] = true
			}
		}
		if !strings.HasSuffix(name, ".index.json") {
			continue
		}
		data, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(name)))
		if err != nil {
			return nil, err
		}
		var index struct {
			WeightMap map[string]string `json:"weight_map"`
		}
		if json.Unmarshal(data, &index) != nil || len(index.WeightMap) == 0 {
			return nil, fmt.Errorf("%w: invalid weight index %s", errUnverifiedModelRevision, name)
		}
		for _, shard := range index.WeightMap {
			if !safeRevisionArtifactPath(shard) {
				return nil, fmt.Errorf("%w: invalid shard path in %s", errUnverifiedModelRevision, name)
			}
			files[filepath.ToSlash(filepath.Join(filepath.Dir(name), shard))] = true
		}
	}
	result := make([]string, 0, len(files))
	for name := range files {
		result = append(result, name)
	}
	sort.Strings(result)
	return result, nil
}

func safeRevisionArtifactPath(name string) bool {
	if name == "" || filepath.IsAbs(name) || strings.Contains(name, "\\") {
		return false
	}
	for _, part := range strings.Split(filepath.ToSlash(name), "/") {
		if part == ".." || part == "." || part == "" {
			return false
		}
	}
	return true
}

func runtimeRevisionArtifact(name string) bool {
	for _, pattern := range modelWeightPatterns {
		if match, _ := filepath.Match(pattern, name); match {
			return true
		}
	}
	if strings.HasSuffix(name, ".data") || strings.HasSuffix(name, ".index.json") || strings.HasSuffix(name, "_mapping.json") {
		return true
	}
	switch name {
	case "config.json", "adapter_config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json",
		"vocab.txt", "vocab.json", "merges.txt", "tokenizer.model", "sentencepiece.model", "spiece.model", "modules.json",
		"sentence_bert_config.json", "config_sentence_transformers.json", "matryoshka_config.json":
		return true
	}
	return false
}

func revisionArtifactExcluded(name string, patterns []string) bool {
	for _, pattern := range patterns {
		// HF's fnmatch '*' crosses directory separators, unlike filepath.Match.
		quoted := regexp.QuoteMeta(pattern)
		quoted = strings.ReplaceAll(quoted, `\*`, ".*")
		quoted = strings.ReplaceAll(quoted, `\?`, ".")
		if matched, _ := regexp.MatchString("^(?:"+quoted+")$", name); pattern != "" && matched {
			return true
		}
	}
	return false
}

func hfArtifactETag(spec ModelSpec, name string) (string, error) {
	metadata := filepath.Join(spec.LocalPath, ".cache", "huggingface", "download", filepath.FromSlash(name)+".metadata")
	data, err := os.ReadFile(metadata)
	if os.IsNotExist(err) {
		if etag, ok := hfSnapshotBlobETag(spec, name); ok {
			return etag, nil
		}
		return "", fmt.Errorf("%w: missing HF metadata for %s", errUnverifiedModelRevision, name)
	}
	if err != nil {
		return "", err
	}
	fields := strings.Fields(string(data))
	if len(fields) != 3 || !strings.EqualFold(fields[0], spec.Revision) {
		return "", fmt.Errorf("%w: stale or malformed HF metadata for %s", errUnverifiedModelRevision, name)
	}
	timestamp, err := strconv.ParseFloat(fields[2], 64)
	if err != nil || timestamp <= 0 || math.IsNaN(timestamp) || math.IsInf(timestamp, 0) || !validHFETag(fields[1]) {
		return "", fmt.Errorf("%w: malformed HF metadata for %s", errUnverifiedModelRevision, name)
	}
	return strings.ToLower(fields[1]), nil
}

// Standard HF cache paths encode both repo and revision. Each snapshot file
// points to an etag-named blob, whose bytes are still independently verified.
func hfSnapshotBlobETag(spec ModelSpec, name string) (string, bool) {
	root, err := filepath.EvalSymlinks(spec.LocalPath)
	if err != nil || filepath.Base(root) != spec.Revision || filepath.Base(filepath.Dir(root)) != "snapshots" {
		return "", false
	}
	repo := filepath.Dir(filepath.Dir(root))
	if filepath.Base(repo) != "models--"+strings.ReplaceAll(spec.RepoID, "/", "--") {
		return "", false
	}
	blob, err := filepath.EvalSymlinks(filepath.Join(root, filepath.FromSlash(name)))
	if err != nil || filepath.Dir(blob) != filepath.Join(repo, "blobs") || !validHFETag(filepath.Base(blob)) {
		return "", false
	}
	return strings.ToLower(filepath.Base(blob)), true
}

func validHFETag(value string) bool {
	if len(value) != 40 && len(value) != 64 {
		return false
	}
	_, err := hex.DecodeString(value)
	return err == nil
}

func verifyHFArtifactBytes(path, etag string) error {
	file, err := os.Open(path)
	if err != nil {
		return err
	}
	defer file.Close()
	before, err := file.Stat()
	if err != nil {
		return err
	}
	if !before.Mode().IsRegular() {
		return fmt.Errorf("artifact is not a regular file")
	}
	var hasher hash.Hash
	if len(etag) == 64 {
		hasher = sha256.New()
	} else {
		hasher = sha1.New() //nolint:gosec // HF's Git blob etag, not a security signature.
		fmt.Fprintf(hasher, "blob %d\x00", before.Size())
	}
	if _, copyErr := io.Copy(hasher, file); copyErr != nil {
		return copyErr
	}
	after, err := file.Stat()
	if err != nil {
		return err
	}
	if before.Size() != after.Size() || !before.ModTime().Equal(after.ModTime()) {
		return fmt.Errorf("artifact changed during verification")
	}
	if hex.EncodeToString(hasher.Sum(nil)) != etag {
		return fmt.Errorf("content does not match HF etag")
	}
	return nil
}
