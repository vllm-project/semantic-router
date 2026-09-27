package framework

import (
	"os"
	"path/filepath"
)

// reportPath keeps batch profiles' evidence separate while preserving local defaults.
func reportPath(name string) string {
	return filepath.Join(os.Getenv("E2E_REPORT_DIR"), name)
}
