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
	data, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("verify config reload source: %w", err)
	}
	digest := sha256.Sum256(data)
	if candidate == nil || candidate.DocumentHash != hex.EncodeToString(digest[:]) {
		return errConfigReloadSuperseded
	}
	return nil
}
