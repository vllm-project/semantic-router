package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"strings"
)

const omniManifestFile = "vela_omni_manifest.json"

var (
	sha256Hex = regexp.MustCompile(`^[0-9a-f]{64}$`)
	sha1Hex   = regexp.MustCompile(`^[0-9a-f]{40}$`)
)

type artifactEvidence struct {
	Repository    string
	MaxTextLength int
	Files         map[string]string
}

// modelArtifact verifies the derived deployment's content-addressed manifest.
// Native preparation subsequently validates every task, graph and processor field.
func modelArtifact(dir, revision string) (artifactEvidence, error) {
	result := artifactEvidence{Files: map[string]string{}}
	data, err := os.ReadFile(filepath.Join(dir, omniManifestFile))
	if err != nil {
		return result, err
	}
	var manifest struct {
		FormatVersion int    `json:"format_version"`
		Adapter       string `json:"adapter"`
		Source        struct {
			RepoID   string `json:"repo_id"`
			Revision string `json:"revision"`
		} `json:"source"`
		MaxTextLength int               `json:"max_text_length"`
		Files         map[string]string `json:"files"`
	}
	if err := json.Unmarshal(data, &manifest); err != nil {
		return result, err
	}
	if manifest.FormatVersion != 1 || manifest.Adapter != "vela_omni" || !sha1Hex.MatchString(manifest.Source.Revision) || manifest.Source.Revision != revision || manifest.Source.RepoID == "" || manifest.MaxTextLength <= 0 || len(manifest.Files) == 0 {
		return result, fmt.Errorf("invalid Omni manifest or mismatched source revision")
	}
	result.Repository, result.MaxTextLength = manifest.Source.RepoID, manifest.MaxTextLength
	for name, expected := range manifest.Files {
		cleaned := filepath.Clean(name)
		if filepath.IsAbs(name) || cleaned != name || cleaned == ".." || strings.HasPrefix(cleaned, "../") || !sha256Hex.MatchString(expected) {
			return result, fmt.Errorf("invalid manifest file identity %q", name)
		}
		path := filepath.Join(dir, name)
		real, err := filepath.EvalSymlinks(path)
		if err != nil {
			return result, err
		}
		root, err := filepath.EvalSymlinks(dir)
		if err != nil {
			return result, err
		}
		rel, err := filepath.Rel(root, real)
		if err != nil || rel == ".." || strings.HasPrefix(rel, "../") {
			return result, fmt.Errorf("artifact file escapes model directory: %s", name)
		}
		file, err := os.Open(real)
		if err != nil {
			return result, err
		}
		hash := sha256.New()
		_, copyErr := io.Copy(hash, file)
		file.Close()
		if copyErr != nil {
			return result, copyErr
		}
		actual := hex.EncodeToString(hash.Sum(nil))
		if actual != expected {
			return result, fmt.Errorf("manifest checksum mismatch: %s", name)
		}
		result.Files[name] = "sha256:" + actual
	}
	digest := sha256.Sum256(data)
	result.Files[omniManifestFile] = "sha256:" + hex.EncodeToString(digest[:])
	return result, nil
}
