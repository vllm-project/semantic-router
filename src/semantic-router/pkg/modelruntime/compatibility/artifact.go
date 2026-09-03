package compatibility

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
)

const candleArtifactSchemaVersionV1 = "semantic-router.candle-artifact/v1"

type candleArtifactIdentity struct {
	SchemaVersion string                       `json:"schema_version"`
	Files         []candleArtifactFileIdentity `json:"files"`
}

type candleArtifactFileIdentity struct {
	Path   string `json:"path"`
	SHA256 string `json:"sha256"`
}

// DigestLocalCandleArtifact identifies the exact files selected by the local
// traditional BERT loader. A caller must still provide the artifact revision:
// a filesystem path alone cannot prove where those bytes came from.
func DigestLocalCandleArtifact(modelPath string) (string, error) {
	info, err := os.Stat(modelPath)
	if err != nil {
		return "", fmt.Errorf("inspect local Candle artifact: %w", err)
	}
	if !info.IsDir() {
		return "", fmt.Errorf("local Candle artifact %q must be a directory", modelPath)
	}

	files := []string{"config.json", "tokenizer.json"}
	weights, err := selectedCandleWeights(modelPath)
	if err != nil {
		return "", err
	}
	files = append(files, weights)

	identity := candleArtifactIdentity{
		SchemaVersion: candleArtifactSchemaVersionV1,
		Files:         make([]candleArtifactFileIdentity, 0, len(files)),
	}
	for _, name := range files {
		digest, err := digestArtifactFile(filepath.Join(modelPath, name))
		if err != nil {
			return "", fmt.Errorf("digest local Candle artifact %s: %w", name, err)
		}
		identity.Files = append(identity.Files, candleArtifactFileIdentity{
			Path:   name,
			SHA256: digest,
		})
	}

	encoded, err := json.Marshal(identity)
	if err != nil {
		return "", fmt.Errorf("encode local Candle artifact identity: %w", err)
	}
	digest := sha256.Sum256(encoded)
	return "sha256:" + hex.EncodeToString(digest[:]), nil
}

func selectedCandleWeights(modelPath string) (string, error) {
	for _, name := range []string{"model.safetensors", "pytorch_model.bin"} {
		info, err := os.Stat(filepath.Join(modelPath, name))
		switch {
		case err == nil && info.Mode().IsRegular():
			return name, nil
		case err == nil:
			continue
		case !os.IsNotExist(err):
			return "", fmt.Errorf("inspect local Candle artifact %s: %w", name, err)
		}
	}
	return "", fmt.Errorf(
		"local Candle artifact requires model.safetensors or pytorch_model.bin",
	)
}

func digestArtifactFile(path string) (string, error) {
	file, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer file.Close()

	digest := sha256.New()
	if _, err := io.Copy(digest, file); err != nil {
		return "", err
	}
	return "sha256:" + hex.EncodeToString(digest.Sum(nil)), nil
}
