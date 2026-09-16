package extproc

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"os"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

var errConfigReloadSuperseded = errors.New("config reload superseded by a newer source document")

func checkFileReloadCandidate(path string, candidate *config.RouterConfig) error {
	hash, err := reloadDocumentHash(path)
	if err != nil {
		return fmt.Errorf("verify config reload source: %w", err)
	}
	if candidate == nil || candidate.DocumentHash != hash {
		return errConfigReloadSuperseded
	}
	return nil
}

func reloadDocumentHash(path string) (string, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}
	digest := sha256.Sum256(data)
	return hex.EncodeToString(digest[:]), nil
}
