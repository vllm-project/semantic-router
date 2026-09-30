package config

import (
	"encoding/hex"
	"fmt"
	"path/filepath"
	"strings"
)

// OperatingPointReference selects an immutable interpretation of model scores.
// Relative paths resolve inside the deployment artifact. No policy is discovered
// implicitly; its model, tokenizer, execution and thresholds are checked at load.
type OperatingPointReference struct {
	Path   string `yaml:"path" json:"path" jsonschema:"required"`
	SHA256 string `yaml:"sha256" json:"sha256" jsonschema:"required"`
}

func (r OperatingPointReference) Validate() error {
	if strings.TrimSpace(r.Path) == "" || strings.TrimSpace(r.Path) != r.Path {
		return fmt.Errorf("operating_point.path must be nonempty and trimmed")
	}
	if !filepath.IsAbs(r.Path) && (filepath.Clean(r.Path) == ".." || strings.HasPrefix(filepath.Clean(r.Path), ".."+string(filepath.Separator))) {
		return fmt.Errorf("operating_point.path must not escape the deployment artifact")
	}
	decoded, err := hex.DecodeString(r.SHA256)
	if err != nil || len(decoded) != 32 || strings.ToLower(r.SHA256) != r.SHA256 {
		return fmt.Errorf("operating_point.sha256 must contain 64 lowercase hexadecimal characters")
	}
	return nil
}

func (r OperatingPointReference) ResolvePath(artifact string) string {
	if filepath.IsAbs(r.Path) {
		return r.Path
	}
	return filepath.Join(ResolveModelPath(artifact), r.Path)
}
