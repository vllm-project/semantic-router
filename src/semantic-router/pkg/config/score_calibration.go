package config

import (
	"encoding/hex"
	"fmt"
	"strings"
)

// ScoreCalibrationReference pins a reviewed signal calibration artifact by
// content. Promotion is this configuration diff; nothing is discovered or
// refreshed at request time.
type ScoreCalibrationReference struct {
	Path   string `yaml:"path" json:"path" jsonschema:"required"`
	SHA256 string `yaml:"sha256" json:"sha256" jsonschema:"required"`
}

func (r ScoreCalibrationReference) validate() error {
	if strings.TrimSpace(r.Path) == "" || strings.TrimSpace(r.Path) != r.Path {
		return fmt.Errorf("calibration.path must be nonempty and trimmed")
	}
	decoded, err := hex.DecodeString(r.SHA256)
	if err != nil || len(decoded) != 32 || strings.ToLower(r.SHA256) != r.SHA256 {
		return fmt.Errorf("calibration.sha256 must contain 64 lowercase hexadecimal characters")
	}
	return nil
}

// validateCategoryCalibration admits a calibration only where its scale holds.
// The artifact binds the model the runtime serves, which a remote backend does
// not report, and it maps the top label's probability, which is the only label
// that can match at a threshold of 0.5 or more.
func validateCategoryCalibration(model *CategoryModel) error {
	if model.Calibration == nil {
		return nil
	}
	if err := model.Calibration.validate(); err != nil {
		return fmt.Errorf("classifier.domain.%w", err)
	}
	if model.Backend != nil {
		return fmt.Errorf("classifier.domain.calibration binds the model the runtime serves and cannot be used with a backend")
	}
	if model.Threshold < 0.5 {
		return fmt.Errorf("classifier.domain.calibration maps the top label's probability and requires a threshold of at least 0.5")
	}
	return nil
}

// CalibratedScoreFamilies names the signal types whose scores this
// configuration declares on a calibrated scale.
func (c *RouterConfig) CalibratedScoreFamilies() []string {
	if c == nil || c.CategoryModel.Calibration == nil {
		return nil
	}
	return []string{SignalTypeDomain}
}
