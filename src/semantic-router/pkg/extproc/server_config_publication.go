package extproc

import (
	"fmt"
	"os"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/configsnapshot"
)

var errConfigReloadSuperseded = configsnapshot.ErrSuperseded

func checkFileReloadCandidate(path string, candidate *config.RouterConfig) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("verify config reload source: %w", err)
	}
	if candidate == nil || candidate.DocumentHash != documentDigest(data) {
		return errConfigReloadSuperseded
	}
	return nil
}
