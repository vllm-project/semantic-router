package compatibility

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"slices"
	"strings"
	"time"
)

const QualificationSuiteSchemaVersionV1 = "semantic-router.local-candle-qualification-suite/v1"

// QualificationSuite contains explicit expected observations for one model.
type QualificationSuite struct {
	SchemaVersion    string               `json:"schema_version"`
	LabelParity      []LabelProbe         `json:"label_parity"`
	InputBounds      []InputBoundaryProbe `json:"input_bounds"`
	DeadlineBehavior DeadlineProbe        `json:"deadline_behavior"`
}

type LabelProbe struct {
	Name      string `json:"name"`
	Input     string `json:"input"`
	WantLabel string `json:"want_label"`
}

type InputBoundaryProbe struct {
	Name      string `json:"name"`
	Input     string `json:"input"`
	WantError bool   `json:"want_error"`
}

type DeadlineProbe struct {
	Input             string `json:"input"`
	MaxDurationMillis int64  `json:"max_duration_ms"`
}

// Validate rejects an ambiguous or incomplete model-specific probe suite.
func (s QualificationSuite) Validate(labels []string) error {
	if s.SchemaVersion != QualificationSuiteSchemaVersionV1 {
		return fmt.Errorf("unsupported Candle qualification suite schema %q", s.SchemaVersion)
	}
	if len(s.LabelParity) == 0 {
		return fmt.Errorf("Candle qualification label_parity probes must not be empty")
	}
	if len(s.InputBounds) == 0 {
		return fmt.Errorf("Candle qualification input_bounds probes must not be empty")
	}
	if s.DeadlineBehavior.MaxDurationMillis <= 0 {
		return fmt.Errorf("Candle qualification deadline max_duration_ms must be positive")
	}
	if s.DeadlineBehavior.MaxDurationMillis > math.MaxInt64/int64(time.Millisecond) {
		return fmt.Errorf("Candle qualification deadline max_duration_ms exceeds time.Duration")
	}
	if err := validateNamedProbes("label_parity", len(s.LabelParity), func(index int) string {
		return s.LabelParity[index].Name
	}); err != nil {
		return err
	}
	for index, probe := range s.LabelParity {
		if !slices.Contains(labels, probe.WantLabel) {
			return fmt.Errorf(
				"Candle qualification label_parity[%d].want_label %q is not in subject labels",
				index,
				probe.WantLabel,
			)
		}
	}
	return validateNamedProbes("input_bounds", len(s.InputBounds), func(index int) string {
		return s.InputBounds[index].Name
	})
}

// Digest identifies the exact versioned probes used to produce a receipt.
func (s QualificationSuite) Digest(labels []string) (string, error) {
	if err := s.Validate(labels); err != nil {
		return "", err
	}
	encoded, err := json.Marshal(s)
	if err != nil {
		return "", fmt.Errorf("encode Candle qualification suite: %w", err)
	}
	digest := sha256.Sum256(encoded)
	return "sha256:" + hex.EncodeToString(digest[:]), nil
}

// ParseQualificationSuite strictly decodes and validates one probe suite.
func ParseQualificationSuite(data []byte, labels []string) (QualificationSuite, error) {
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	var suite QualificationSuite
	if err := decoder.Decode(&suite); err != nil {
		return QualificationSuite{}, fmt.Errorf("decode Candle qualification suite: %w", err)
	}
	if err := decoder.Decode(&struct{}{}); err != io.EOF {
		return QualificationSuite{}, fmt.Errorf("decode Candle qualification suite: trailing content")
	}
	if err := suite.Validate(labels); err != nil {
		return QualificationSuite{}, err
	}
	return suite, nil
}

func validateNamedProbes(kind string, count int, nameAt func(int) string) error {
	seen := make(map[string]struct{}, count)
	for index := range count {
		name := nameAt(index)
		if strings.TrimSpace(name) == "" {
			return fmt.Errorf("Candle qualification %s[%d].name is required", kind, index)
		}
		if strings.TrimSpace(name) != name {
			return fmt.Errorf(
				"Candle qualification %s[%d].name must not have surrounding whitespace",
				kind,
				index,
			)
		}
		if _, exists := seen[name]; exists {
			return fmt.Errorf("Candle qualification %s probe %q is duplicated", kind, name)
		}
		seen[name] = struct{}{}
	}
	return nil
}
