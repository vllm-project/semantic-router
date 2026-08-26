package quotaruntime

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"regexp"
	"sort"
	"strings"
	"time"

	"github.com/google/uuid"
)

const (
	admissionRecoveryVersion   = "quota-admission-recovery.v1"
	maximumRecoveryRecordBytes = 256 << 10
	maximumRecoveryRules       = 512
)

var recoveryCodePattern = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$`)

// AdmissionRecoveryContext is the complete request identity needed to create
// a terminal usage record after an admission lease expires. The shape is
// deliberately closed and contains only identifiers and display snapshots
// already allowed in the usage ledger.
type AdmissionRecoveryContext struct {
	EventID           string
	FenceID           string
	NamespaceID       string
	ExternalRequestID string
	ReplayID          string
	Protocol          string
	Path              string
	OccurredAt        time.Time
	Stream            bool
	Principal         RecoveryPrincipal
	Routing           RecoveryRouting
	FallbackDispatch  RecoveryDispatch
}

type RecoveryPrincipal struct {
	APIKeyID string
	UserID   string
	TeamID   string
}

type RecoveryRouting struct {
	EntrypointID       string
	EntrypointName     string
	EntrypointRuleID   string
	EntrypointRuleName string
	RecipeID           string
	RecipeName         string
	RecipeRevision     int64
	RoutingRevision    int64
	AccessRevision     int64
}

type RecoveryDispatch struct {
	ModelID       string
	ModelName     string
	ModelRevision int64
	Currency      string
}

type admissionRecoveryRecord struct {
	Version string                   `json:"version"`
	Context AdmissionRecoveryContext `json:"context"`
	Rules   []RuleBinding            `json:"rules"`
}

func encodeAdmissionRecoveryRecord(
	context *AdmissionRecoveryContext,
	rules []RuleBinding,
) (string, string, error) {
	if context == nil {
		return "", "", nil
	}
	record := admissionRecoveryRecord{
		Version: admissionRecoveryVersion,
		Context: *context,
		Rules:   cloneRuleBindingsForRecovery(rules),
	}
	if err := record.validate(); err != nil {
		return "", "", err
	}
	payload, err := json.Marshal(record)
	if err != nil {
		return "", "", fmt.Errorf("%w: encode admission recovery record", ErrInvalidRequest)
	}
	if len(payload) > maximumRecoveryRecordBytes {
		return "", "", fmt.Errorf("%w: admission recovery record exceeds %d bytes", ErrInvalidRequest, maximumRecoveryRecordBytes)
	}
	digest := sha256.Sum256(payload)
	return string(payload), hex.EncodeToString(digest[:]), nil
}

func decodeAdmissionRecoveryRecord(payload, expectedDigest string) (admissionRecoveryRecord, error) {
	if payload == "" || len(payload) > maximumRecoveryRecordBytes || strings.ContainsRune(payload, '\x00') {
		return admissionRecoveryRecord{}, fmt.Errorf("%w: admission recovery record is unavailable", ErrRuntimeCorrupt)
	}
	digest := sha256.Sum256([]byte(payload))
	if hex.EncodeToString(digest[:]) != expectedDigest {
		return admissionRecoveryRecord{}, fmt.Errorf("%w: admission recovery record digest differs", ErrRuntimeCorrupt)
	}
	decoder := json.NewDecoder(bytes.NewBufferString(payload))
	decoder.DisallowUnknownFields()
	var record admissionRecoveryRecord
	if err := decoder.Decode(&record); err != nil {
		return admissionRecoveryRecord{}, fmt.Errorf("%w: decode admission recovery record", ErrRuntimeCorrupt)
	}
	if err := decoder.Decode(&struct{}{}); err != io.EOF {
		return admissionRecoveryRecord{}, fmt.Errorf("%w: admission recovery record has trailing data", ErrRuntimeCorrupt)
	}
	if err := record.validate(); err != nil {
		return admissionRecoveryRecord{}, fmt.Errorf("%w: invalid admission recovery record: %v", ErrRuntimeCorrupt, err)
	}
	return record, nil
}

func (record admissionRecoveryRecord) validate() error {
	if record.Version != admissionRecoveryVersion {
		return fmt.Errorf("unsupported recovery record version %q", record.Version)
	}
	if err := record.Context.validate(); err != nil {
		return err
	}
	if len(record.Rules) > maximumRecoveryRules {
		return fmt.Errorf("recovery rule count exceeds %d", maximumRecoveryRules)
	}
	for index, rule := range record.Rules {
		if err := rule.Validate(); err != nil {
			return fmt.Errorf("recovery rule %d: %w", index, err)
		}
	}
	return nil
}

func (context AdmissionRecoveryContext) validate() error {
	for label, value := range map[string]string{
		"event ID": context.EventID, "fence ID": context.FenceID,
		"namespace ID": context.NamespaceID, "API key ID": context.Principal.APIKeyID,
	} {
		if !canonicalRecoveryUUID(value) {
			return fmt.Errorf("%s must be a canonical UUID", label)
		}
	}
	for label, value := range map[string]string{
		"user ID": context.Principal.UserID, "team ID": context.Principal.TeamID,
	} {
		if value != "" && !canonicalRecoveryUUID(value) {
			return fmt.Errorf("%s must be empty or a canonical UUID", label)
		}
	}
	if context.ExternalRequestID != "" && !canonicalRecoveryUUID(context.ExternalRequestID) {
		return fmt.Errorf("external request ID must be empty or a canonical UUID")
	}
	if err := boundedRecoveryIdentity("replay ID", context.ReplayID, 256, true); err != nil {
		return err
	}
	if err := boundedRecoveryCode("protocol", context.Protocol, false); err != nil {
		return err
	}
	if context.Path == "" || len(context.Path) > 2048 || !strings.HasPrefix(context.Path, "/") ||
		strings.ContainsAny(context.Path, "?#\x00") || recoveryLooksSensitive(context.Path) {
		return fmt.Errorf("recovery path is invalid")
	}
	if context.OccurredAt.IsZero() {
		return fmt.Errorf("recovery occurrence time is required")
	}
	if err := context.Routing.validate(); err != nil {
		return err
	}
	return context.FallbackDispatch.validate()
}

func (routing RecoveryRouting) validate() error {
	for label, value := range map[string]string{
		"entrypoint ID":      routing.EntrypointID,
		"entrypoint rule ID": routing.EntrypointRuleID,
		"recipe ID":          routing.RecipeID,
	} {
		if err := boundedRecoveryCode(label, value, true); err != nil {
			return err
		}
	}
	if routing.RecipeRevision < 0 || routing.RoutingRevision < 0 || routing.AccessRevision < 0 {
		return fmt.Errorf("recovery routing revisions cannot be negative")
	}
	for label, value := range map[string]string{
		"entrypoint name":      routing.EntrypointName,
		"entrypoint rule name": routing.EntrypointRuleName,
		"recipe name":          routing.RecipeName,
	} {
		if err := boundedRecoveryIdentity(label, value, 256, true); err != nil {
			return err
		}
	}
	return nil
}

func (dispatch RecoveryDispatch) validate() error {
	if dispatch.ModelID != "" {
		if err := boundedRecoveryCode("fallback Model ID", dispatch.ModelID, true); err != nil {
			return err
		}
		if dispatch.ModelRevision <= 0 {
			return fmt.Errorf("fallback Model revision is required with its ID")
		}
	} else if dispatch.ModelRevision != 0 {
		return fmt.Errorf("fallback Model revision requires an ID")
	}
	if err := boundedRecoveryIdentity("fallback Model name", dispatch.ModelName, 512, true); err != nil {
		return err
	}
	if !validCurrencyCode(dispatch.Currency) {
		return fmt.Errorf("fallback dispatch requires a three-letter uppercase currency")
	}
	return nil
}

func cloneRuleBindingsForRecovery(source []RuleBinding) []RuleBinding {
	result := make([]RuleBinding, len(source))
	copy(result, source)
	for index := range result {
		result[index].CalendarSchedule = append([]CalendarInterval(nil), source[index].CalendarSchedule...)
	}
	sort.Slice(result, func(left, right int) bool {
		if result[left].Rule.Ordinal != result[right].Rule.Ordinal {
			return result[left].Rule.Ordinal < result[right].Rule.Ordinal
		}
		if result[left].BindingID != result[right].BindingID {
			return result[left].BindingID < result[right].BindingID
		}
		return result[left].Rule.ID < result[right].Rule.ID
	})
	return result
}

func canonicalRecoveryUUID(value string) bool {
	parsed, err := uuid.Parse(value)
	return err == nil && parsed.String() == value
}

func boundedRecoveryCode(label, value string, optional bool) error {
	if value == "" && optional {
		return nil
	}
	if value == "" || !recoveryCodePattern.MatchString(value) {
		return fmt.Errorf("%s is not a bounded canonical code", label)
	}
	return nil
}

func boundedRecoveryIdentity(label, value string, maximum int, optional bool) error {
	if value == "" && optional {
		return nil
	}
	if value == "" || len(value) > maximum || strings.TrimSpace(value) != value || strings.ContainsRune(value, '\x00') {
		return fmt.Errorf("%s is not a bounded canonical identity", label)
	}
	if recoveryLooksSensitive(value) {
		return fmt.Errorf("%s contains credential-like material", label)
	}
	return nil
}

func recoveryLooksSensitive(value string) bool {
	lower := strings.ToLower(value)
	return strings.Contains(lower, "bearer ") || strings.Contains(lower, "authorization:") ||
		strings.Contains(lower, "api_key=") || strings.Contains(lower, "vsr_") ||
		strings.Contains(lower, "vsd_") || strings.Contains(lower, "vsm_")
}
