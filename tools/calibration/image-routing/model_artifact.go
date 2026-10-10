package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"slices"
)

var sha1Hex = regexp.MustCompile(`^[0-9a-f]{40}$`)

type artifactEvidence struct {
	Repository    string
	MaxTextLength int
	Files         map[string]string
}

// modelArtifact attests a published Vela Omni snapshot by the SHA-256 of
// every file in it; the caller compares them with the runtime's pins, and the
// model runtime verifies the same files again when it loads them.
func modelArtifact(dir, repository, revision string) (artifactEvidence, error) {
	result := artifactEvidence{Repository: repository, Files: map[string]string{}}
	if repository == "" || !sha1Hex.MatchString(revision) {
		return result, fmt.Errorf("a published repository and a 40-character revision are required")
	}
	data, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return result, err
	}
	var config struct {
		Architectures []string `json:"architectures"`
		MaxTextLength int      `json:"max_text_length"`
	}
	if err = json.Unmarshal(data, &config); err != nil {
		return result, err
	}
	if !slices.Equal(config.Architectures, []string{"VelaOmni"}) || config.MaxTextLength <= 0 {
		return result, fmt.Errorf("%s is not a Vela Omni snapshot", dir)
	}
	result.MaxTextLength = config.MaxTextLength
	root, err := filepath.EvalSymlinks(dir)
	if err != nil {
		return result, err
	}
	err = filepath.WalkDir(root, func(path string, entry fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		name, relErr := filepath.Rel(root, path)
		if relErr != nil {
			return relErr
		}
		// The Hugging Face client keeps its download metadata here.
		if entry.IsDir() && name == ".cache" {
			return filepath.SkipDir
		}
		if entry.IsDir() {
			return nil
		}
		if !entry.Type().IsRegular() {
			return fmt.Errorf("snapshot file is not a regular file: %s", name)
		}
		digest, hashErr := fileSHA256(path)
		if hashErr != nil {
			return hashErr
		}
		result.Files[filepath.ToSlash(name)] = "sha256:" + digest
		return nil
	})
	return result, err
}

func fileSHA256(path string) (string, error) {
	file, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer file.Close()
	hash := sha256.New()
	if _, err := io.Copy(hash, file); err != nil {
		return "", err
	}
	return hex.EncodeToString(hash.Sum(nil)), nil
}
