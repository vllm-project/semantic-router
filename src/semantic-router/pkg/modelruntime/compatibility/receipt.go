// Package compatibility defines offline compatibility evidence for router model runtimes.
package compatibility

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"strings"
)

const (
	SubjectSchemaVersionV1      = "semantic-router.compatibility-subject/v1"
	ReceiptSchemaVersionV1      = "semantic-router.compatibility-receipt/v1"
	ExecutionPredicateType      = "semantic-router.execution"
	ExecutionPredicateVersionV1 = "v1"
	CandleConnectorV1           = "sr.candle.embedded.v1"
	LabelDistributionContractV1 = "label_distribution.v1"
	CheckLabelParity            = "label_parity"
	CheckInputBounds            = "input_bounds"
	CheckDeadlineBehavior       = "deadline_behavior"
	CheckUnavailableBehavior    = "unavailable_behavior"
)

var requiredLabelDistributionChecks = [...]string{
	CheckLabelParity,
	CheckInputBounds,
	CheckDeadlineBehavior,
	CheckUnavailableBehavior,
}

// Subject identifies one exact qualification target independently of its connector.
type Subject struct {
	SchemaVersion    string   `json:"schema_version"`
	ArtifactRevision string   `json:"artifact_revision"`
	ArtifactDigest   string   `json:"artifact_digest"`
	TaskContract     string   `json:"task_contract"`
	Connector        string   `json:"connector"`
	Precision        string   `json:"precision"`
	Provider         string   `json:"provider"`
	DeviceProfile    string   `json:"device_profile"`
	RouterRevision   string   `json:"router_revision"`
	Labels           []string `json:"labels,omitempty"`
}

// CheckOutcome records one observed conformance check without assigning support status.
type CheckOutcome struct {
	Name    string `json:"name"`
	Passed  bool   `json:"passed"`
	Details string `json:"details,omitempty"`
}

// Receipt is unsigned offline evidence. Its content digest does not authenticate
// an issuer or imply that the recorded checks passed.
type Receipt struct {
	SchemaVersion            string         `json:"schema_version"`
	Subject                  Subject        `json:"subject"`
	SubjectDigest            string         `json:"subject_digest"`
	PredicateType            string         `json:"predicate_type"`
	PredicateVersion         string         `json:"predicate_version"`
	QualificationSuiteDigest string         `json:"qualification_suite_digest"`
	Checks                   []CheckOutcome `json:"checks"`
}

// NewReceipt constructs an evidence payload for offline conformance results.
func NewReceipt(
	subject Subject,
	qualificationSuiteDigest string,
	checks []CheckOutcome,
) (Receipt, error) {
	subject.Labels = append([]string(nil), subject.Labels...)
	digest, err := subject.Digest()
	if err != nil {
		return Receipt{}, err
	}
	receipt := Receipt{
		SchemaVersion:            ReceiptSchemaVersionV1,
		Subject:                  subject,
		SubjectDigest:            digest,
		PredicateType:            ExecutionPredicateType,
		PredicateVersion:         ExecutionPredicateVersionV1,
		QualificationSuiteDigest: qualificationSuiteDigest,
		Checks:                   append([]CheckOutcome(nil), checks...),
	}
	if err := receipt.Validate(); err != nil {
		return Receipt{}, err
	}
	return receipt, nil
}

// Digest returns the deterministic identity of the validated subject.
func (s Subject) Digest() (string, error) {
	if err := s.Validate(); err != nil {
		return "", err
	}
	encoded, err := canonicalJSON(s)
	if err != nil {
		return "", err
	}
	digest := sha256.Sum256(encoded)
	return "sha256:" + hex.EncodeToString(digest[:]), nil
}

// Validate rejects incomplete or ambiguous subject identities.
func (s Subject) Validate() error {
	required := []struct {
		name  string
		value string
	}{
		{"schema_version", s.SchemaVersion},
		{"artifact_revision", s.ArtifactRevision},
		{"artifact_digest", s.ArtifactDigest},
		{"task_contract", s.TaskContract},
		{"connector", s.Connector},
		{"precision", s.Precision},
		{"provider", s.Provider},
		{"device_profile", s.DeviceProfile},
		{"router_revision", s.RouterRevision},
	}
	for _, field := range required {
		if strings.TrimSpace(field.value) == "" {
			return fmt.Errorf("compatibility subject %s is required", field.name)
		}
		if strings.TrimSpace(field.value) != field.value {
			return fmt.Errorf("compatibility subject %s must not have surrounding whitespace", field.name)
		}
	}
	if s.SchemaVersion != SubjectSchemaVersionV1 {
		return fmt.Errorf("unsupported compatibility subject schema %q", s.SchemaVersion)
	}
	if err := validateSHA256("compatibility subject", "artifact_digest", s.ArtifactDigest); err != nil {
		return err
	}
	if s.TaskContract == LabelDistributionContractV1 && len(s.Labels) < 2 {
		return fmt.Errorf("compatibility subject labels must contain at least two entries")
	}
	seen := make(map[string]struct{}, len(s.Labels))
	for index, label := range s.Labels {
		if strings.TrimSpace(label) == "" {
			return fmt.Errorf("compatibility subject labels[%d] is required", index)
		}
		if strings.TrimSpace(label) != label {
			return fmt.Errorf("compatibility subject labels[%d] must not have surrounding whitespace", index)
		}
		if _, exists := seen[label]; exists {
			return fmt.Errorf("compatibility subject label %q is duplicated", label)
		}
		seen[label] = struct{}{}
	}
	return nil
}

// Validate checks receipt metadata and proves that the embedded subject matches its digest.
func (r Receipt) Validate() error {
	if r.SchemaVersion != ReceiptSchemaVersionV1 {
		return fmt.Errorf("unsupported compatibility receipt schema %q", r.SchemaVersion)
	}
	if r.PredicateType != ExecutionPredicateType {
		return fmt.Errorf("unsupported compatibility predicate type %q", r.PredicateType)
	}
	if r.PredicateVersion != ExecutionPredicateVersionV1 {
		return fmt.Errorf("unsupported compatibility predicate version %q", r.PredicateVersion)
	}
	if err := validateSHA256(
		"compatibility receipt",
		"qualification_suite_digest",
		r.QualificationSuiteDigest,
	); err != nil {
		return err
	}
	expected, err := r.Subject.Digest()
	if err != nil {
		return err
	}
	if r.SubjectDigest != expected {
		return fmt.Errorf(
			"compatibility receipt subject_digest %q does not match subject %q",
			r.SubjectDigest,
			expected,
		)
	}
	return validateChecks(r.Checks, r.Subject.TaskContract)
}

func validateChecks(checks []CheckOutcome, taskContract string) error {
	if len(checks) == 0 {
		return fmt.Errorf("compatibility receipt checks must not be empty")
	}
	seen := make(map[string]struct{}, len(checks))
	for index, check := range checks {
		if strings.TrimSpace(check.Name) == "" {
			return fmt.Errorf("compatibility receipt checks[%d].name is required", index)
		}
		if strings.TrimSpace(check.Name) != check.Name {
			return fmt.Errorf("compatibility receipt checks[%d].name must not have surrounding whitespace", index)
		}
		if _, exists := seen[check.Name]; exists {
			return fmt.Errorf("compatibility receipt check %q is duplicated", check.Name)
		}
		seen[check.Name] = struct{}{}
	}
	if taskContract != LabelDistributionContractV1 {
		return nil
	}
	for _, name := range requiredLabelDistributionChecks {
		if _, exists := seen[name]; !exists {
			return fmt.Errorf("compatibility receipt check %q is required", name)
		}
	}
	return nil
}

// ParseReceipt strictly decodes and validates one receipt payload.
func ParseReceipt(data []byte) (Receipt, error) {
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	var receipt Receipt
	if err := decoder.Decode(&receipt); err != nil {
		return Receipt{}, fmt.Errorf("decode compatibility receipt: %w", err)
	}
	if err := decoder.Decode(&struct{}{}); err != io.EOF {
		return Receipt{}, fmt.Errorf("decode compatibility receipt: trailing content")
	}
	if err := receipt.Validate(); err != nil {
		return Receipt{}, err
	}
	return receipt, nil
}

// CanonicalJSON returns the v1 evidence encoding: fixed field order, declared
// slice order, no HTML escaping, and no trailing newline. It is not RFC 8785.
func (r Receipt) CanonicalJSON() ([]byte, error) {
	if err := r.Validate(); err != nil {
		return nil, err
	}
	return canonicalJSON(r)
}

// Digest identifies the complete validated evidence, including all outcomes.
// It is kept outside the payload to avoid a self-referential checksum.
func (r Receipt) Digest() (string, error) {
	encoded, err := r.CanonicalJSON()
	if err != nil {
		return "", err
	}
	digest := sha256.Sum256(encoded)
	return "sha256:" + hex.EncodeToString(digest[:]), nil
}

// VerifyDigest checks content integrity against an independently retained digest.
// A digest stored alongside editable evidence is not proof of authenticity.
func (r Receipt) VerifyDigest(expected string) error {
	if err := validateSHA256("compatibility receipt", "expected_digest", expected); err != nil {
		return err
	}
	actual, err := r.Digest()
	if err != nil {
		return err
	}
	if actual != expected {
		return fmt.Errorf("compatibility receipt digest %q does not match expected %q", actual, expected)
	}
	return nil
}

func canonicalJSON(value any) ([]byte, error) {
	var encoded bytes.Buffer
	encoder := json.NewEncoder(&encoded)
	encoder.SetEscapeHTML(false)
	if err := encoder.Encode(value); err != nil {
		return nil, fmt.Errorf("encode compatibility identity: %w", err)
	}
	return bytes.TrimSuffix(encoded.Bytes(), []byte{'\n'}), nil
}

func validateSHA256(scope string, field string, value string) error {
	encoded, ok := strings.CutPrefix(value, "sha256:")
	if !ok || len(encoded) != sha256.Size*2 || encoded != strings.ToLower(encoded) {
		return fmt.Errorf("%s %s must be a lowercase sha256 digest", scope, field)
	}
	if _, err := hex.DecodeString(encoded); err != nil {
		return fmt.Errorf("%s %s must be a lowercase sha256 digest", scope, field)
	}
	return nil
}
