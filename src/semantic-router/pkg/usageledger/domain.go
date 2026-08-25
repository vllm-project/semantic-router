package usageledger

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"regexp"
	"sort"
	"strings"
	"time"

	"github.com/google/uuid"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/quota"
)

const (
	TerminalEventSchema = "usage.v1"
	CostScaleDigits     = 15
	maxEventBytes       = 1 << 20
)

var (
	ErrInvalidEvent = errors.New("invalid terminal usage event")
	ErrConflict     = errors.New("usage settlement conflict")

	codePattern     = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$`)
	metaPattern     = regexp.MustCompile(`^[a-z][a-z0-9_]{0,63}$`)
	currencyPattern = regexp.MustCompile(`^[A-Z]{3}$`)
)

type UsageState string

const (
	UsageKnownZero   UsageState = "known_zero"
	UsageKnownActual UsageState = "known_actual"
	UsageUnknown     UsageState = "unknown"
)

type EvidenceState string

const (
	EvidenceKnown   EvidenceState = "known"
	EvidenceMixed   EvidenceState = "mixed"
	EvidenceUnknown EvidenceState = "unknown"
)

type CostState string

const (
	CostComplete CostState = "complete"
	CostUnknown  CostState = "unknown"
)

type PrincipalSnapshot struct {
	APIKeyID     string `json:"apiKeyId,omitempty"`
	CredentialID string `json:"credentialId,omitempty"`
	UserID       string `json:"userId,omitempty"`
	TeamID       string `json:"teamId,omitempty"`
	APIKeyName   string `json:"apiKeyName,omitempty"`
	UserName     string `json:"userName,omitempty"`
	TeamName     string `json:"teamName,omitempty"`
}

type RoutingSnapshot struct {
	EntrypointID       string `json:"entrypointId,omitempty"`
	EntrypointName     string `json:"entrypointName,omitempty"`
	EntrypointRuleID   string `json:"entrypointRuleId,omitempty"`
	EntrypointRuleName string `json:"entrypointRuleName,omitempty"`
	RecipeID           string `json:"recipeId,omitempty"`
	RecipeName         string `json:"recipeName,omitempty"`
	RecipeRevision     int64  `json:"recipeRevision,omitempty"`
	RoutingRevision    int64  `json:"routingRevision,omitempty"`
	AccessRevision     int64  `json:"accessRevision,omitempty"`
}

type ServedUsage struct {
	InputTokens  string `json:"inputTokens"`
	InputKnown   bool   `json:"inputKnown"`
	OutputTokens string `json:"outputTokens"`
	OutputKnown  bool   `json:"outputKnown"`
}

type Attempt struct {
	AttemptID   string     `json:"attemptId"`
	Ordinal     int        `json:"ordinal"`
	BackendID   string     `json:"backendId,omitempty"`
	ProviderID  string     `json:"providerId,omitempty"`
	State       UsageState `json:"state"`
	StatusCode  int        `json:"statusCode,omitempty"`
	ErrorCode   string     `json:"errorCode,omitempty"`
	StartedAt   time.Time  `json:"startedAt"`
	CompletedAt time.Time  `json:"completedAt"`
}

type DispatchCost struct {
	Currency  string    `json:"currency"`
	State     CostState `json:"state"`
	Numerator string    `json:"numerator"`
	Reason    string    `json:"reason,omitempty"`
}

type Dispatch struct {
	DispatchID       string       `json:"dispatchId"`
	ParentDispatchID string       `json:"parentDispatchId,omitempty"`
	ParallelGroup    string       `json:"parallelGroup,omitempty"`
	Ordinal          int          `json:"ordinal"`
	DispatchType     string       `json:"dispatchType"`
	DecisionID       string       `json:"decisionId,omitempty"`
	DecisionName     string       `json:"decisionName,omitempty"`
	DecisionTier     int          `json:"decisionTier,omitempty"`
	ModelID          string       `json:"modelId,omitempty"`
	ModelName        string       `json:"modelName,omitempty"`
	ModelRevision    int64        `json:"modelRevision,omitempty"`
	BackendID        string       `json:"backendId,omitempty"`
	ProviderID       string       `json:"providerId,omitempty"`
	ProviderModelID  string       `json:"providerModelId,omitempty"`
	PricingRevision  int64        `json:"pricingRevision,omitempty"`
	RetryClass       string       `json:"retryClass,omitempty"`
	CacheState       string       `json:"cacheState,omitempty"`
	InputTokens      string       `json:"inputTokens"`
	CacheReadTokens  string       `json:"cacheReadTokens"`
	CacheWriteTokens string       `json:"cacheWriteTokens"`
	OutputTokens     string       `json:"outputTokens"`
	UsageState       UsageState   `json:"usageState"`
	UnknownReason    string       `json:"unknownReason,omitempty"`
	Cost             DispatchCost `json:"cost"`
	EvidenceDigest   string       `json:"evidenceDigest,omitempty"`
	StartedAt        time.Time    `json:"startedAt"`
	CompletedAt      time.Time    `json:"completedAt"`
	Attempts         []Attempt    `json:"attempts"`
}

type QuotaReceipt struct {
	BindingID string `json:"bindingId"`
	RuleID    string `json:"ruleId"`
	Metric    string `json:"metric"`
	Amount    string `json:"amount"`
}

type FenceBinding struct {
	BindingID      string `json:"bindingId"`
	RuleID         string `json:"ruleId"`
	AdmissionLimit string `json:"admissionLimit,omitempty"`
	MaximumDebit   string `json:"maximumDebit,omitempty"`
}

type UnknownFence struct {
	FenceID  string         `json:"fenceId"`
	Reason   string         `json:"reason"`
	Bindings []FenceBinding `json:"bindings"`
}

// TerminalEvent is the only payload accepted from the quota-runtime usage
// stream. It is intentionally typed and excludes headers, credentials, prompt
// bodies, responses, and arbitrary evidence blobs.
type TerminalEvent struct {
	Schema              string            `json:"schema"`
	EventID             string            `json:"eventId"`
	NamespaceID         string            `json:"namespaceId"`
	AdmissionID         string            `json:"admissionId"`
	FinalizationDigest  string            `json:"finalizationDigest"`
	EvidenceState       EvidenceState     `json:"evidenceState"`
	ExternalRequestID   string            `json:"externalRequestId,omitempty"`
	ReplayID            string            `json:"replayId,omitempty"`
	Protocol            string            `json:"protocol"`
	Path                string            `json:"path"`
	StatusCode          int               `json:"statusCode"`
	ErrorCode           string            `json:"errorCode,omitempty"`
	OccurredAt          time.Time         `json:"occurredAt"`
	CompletedAt         time.Time         `json:"completedAt"`
	LatencyMilliseconds int64             `json:"latencyMilliseconds"`
	TTFTMilliseconds    *int64            `json:"ttftMilliseconds,omitempty"`
	Stream              bool              `json:"stream"`
	ToolCall            bool              `json:"toolCall"`
	CacheState          string            `json:"cacheState,omitempty"`
	Principal           PrincipalSnapshot `json:"principal"`
	Routing             RoutingSnapshot   `json:"routing"`
	Served              ServedUsage       `json:"served"`
	Dispatches          []Dispatch        `json:"dispatches"`
	QuotaReceipts       []QuotaReceipt    `json:"quotaReceipts,omitempty"`
	Metadata            map[string]string `json:"metadata,omitempty"`
	Fence               *UnknownFence     `json:"fence,omitempty"`
}

type CostAggregate struct {
	Currency             string
	KnownNumerator       quota.QuotaInteger
	KnownDispatches      quota.QuotaInteger
	IncompleteDispatches quota.QuotaInteger
}

type EventAggregate struct {
	UsageState           UsageState
	InputTokens          quota.QuotaInteger
	OutputTokens         quota.QuotaInteger
	ServedInputTokens    quota.QuotaInteger
	ServedOutputTokens   quota.QuotaInteger
	IncompleteDispatches quota.QuotaInteger
	Costs                []CostAggregate
}

func DecodeTerminalEvent(payload string) (TerminalEvent, error) {
	if payload == "" || len(payload) > maxEventBytes || strings.ContainsRune(payload, '\x00') {
		return TerminalEvent{}, fmt.Errorf("%w: payload must be non-empty, NUL-free, and at most %d bytes", ErrInvalidEvent, maxEventBytes)
	}
	decoder := json.NewDecoder(strings.NewReader(payload))
	decoder.DisallowUnknownFields()
	var event TerminalEvent
	if err := decoder.Decode(&event); err != nil {
		return TerminalEvent{}, fmt.Errorf("%w: decode: %w", ErrInvalidEvent, err)
	}
	if err := decoder.Decode(&struct{}{}); !errors.Is(err, io.EOF) {
		return TerminalEvent{}, fmt.Errorf("%w: payload has trailing data", ErrInvalidEvent)
	}
	if _, err := event.Validate(); err != nil {
		return TerminalEvent{}, err
	}
	return event, nil
}

func (e TerminalEvent) Validate() (EventAggregate, error) {
	servedInput, servedOutput, err := validateTerminalHeader(e)
	if err != nil {
		return EventAggregate{}, err
	}
	state := eventValidationState{
		aggregate:   EventAggregate{ServedInputTokens: servedInput, ServedOutputTokens: servedOutput},
		incomplete:  !e.Served.InputKnown || !e.Served.OutputKnown,
		costs:       make(map[string]*CostAggregate),
		dispatchIDs: make(map[string]struct{}, len(e.Dispatches)),
	}
	for index, dispatch := range e.Dispatches {
		if err := state.addDispatch(index, dispatch); err != nil {
			return EventAggregate{}, err
		}
	}
	if err := state.validateParents(e.Dispatches); err != nil {
		return EventAggregate{}, err
	}
	if err := state.finalizeEvidence(e.EvidenceState, e.Dispatches); err != nil {
		return EventAggregate{}, err
	}
	if e.Fence != nil {
		if !state.incomplete {
			return EventAggregate{}, invalid("complete accounting cannot open an unknown-usage fence")
		}
		if err := validateFence(*e.Fence); err != nil {
			return EventAggregate{}, err
		}
	}
	if err := validateReceipts(e.QuotaReceipts); err != nil {
		return EventAggregate{}, err
	}
	state.aggregate.Costs = sortedCostAggregates(state.costs)
	return state.aggregate, nil
}

func validateTerminalHeader(e TerminalEvent) (quota.QuotaInteger, quota.QuotaInteger, error) {
	zero := quota.QuotaInteger{}
	if e.Schema != TerminalEventSchema {
		return zero, zero, invalid("schema must be %q", TerminalEventSchema)
	}
	if err := requireUUID("event ID", e.EventID, false); err != nil {
		return zero, zero, err
	}
	if err := requireUUID("namespace ID", e.NamespaceID, false); err != nil {
		return zero, zero, err
	}
	if err := boundedIdentifier("admission ID", e.AdmissionID, 256); err != nil {
		return zero, zero, err
	}
	if !isHexDigest(e.FinalizationDigest) {
		return zero, zero, invalid("finalization digest must be 32-byte lowercase hex")
	}
	if e.ExternalRequestID != "" {
		if err := boundedIdentifier("external request ID", e.ExternalRequestID, 256); err != nil {
			return zero, zero, err
		}
		if looksSensitive(e.ExternalRequestID) {
			return zero, zero, invalid("external request ID is unsafe")
		}
	}
	if e.ReplayID != "" {
		if err := boundedIdentifier("replay ID", e.ReplayID, 256); err != nil {
			return zero, zero, err
		}
		if looksSensitive(e.ReplayID) {
			return zero, zero, invalid("replay ID is unsafe")
		}
	}
	if err := boundedCode("protocol", e.Protocol, true); err != nil {
		return zero, zero, err
	}
	if e.Path == "" || len(e.Path) > 2048 || !strings.HasPrefix(e.Path, "/") || strings.ContainsAny(e.Path, "?#\x00") {
		return zero, zero, invalid("path must be a normalized path without query or fragment")
	}
	if looksSensitive(e.Path) {
		return zero, zero, invalid("path is unsafe")
	}
	if e.StatusCode < 100 || e.StatusCode > 599 {
		return zero, zero, invalid("status code is outside HTTP range")
	}
	if err := boundedCode("error code", e.ErrorCode, false); err != nil {
		return zero, zero, err
	}
	if e.OccurredAt.IsZero() || e.CompletedAt.IsZero() || e.CompletedAt.Before(e.OccurredAt) {
		return zero, zero, invalid("occurred/completed timestamps are required and ordered")
	}
	if e.LatencyMilliseconds < 0 || e.LatencyMilliseconds != e.CompletedAt.Sub(e.OccurredAt).Milliseconds() {
		return zero, zero, invalid("latency must exactly match terminal timestamps")
	}
	if e.TTFTMilliseconds != nil && (*e.TTFTMilliseconds < 0 || *e.TTFTMilliseconds > e.LatencyMilliseconds) {
		return zero, zero, invalid("TTFT is outside request lifetime")
	}
	if err := validatePrincipal(e.Principal); err != nil {
		return zero, zero, err
	}
	if err := validateRouting(e.Routing); err != nil {
		return zero, zero, err
	}
	servedInput, err := parseQuantity("served input tokens", e.Served.InputTokens)
	if err != nil {
		return zero, zero, err
	}
	servedOutput, err := parseQuantity("served output tokens", e.Served.OutputTokens)
	if err != nil {
		return zero, zero, err
	}
	if !e.Served.InputKnown && !servedInput.IsZero() || !e.Served.OutputKnown && !servedOutput.IsZero() {
		return zero, zero, invalid("unknown served usage cannot claim token values")
	}
	if len(e.Dispatches) == 0 || len(e.Dispatches) > 4096 {
		return zero, zero, invalid("dispatch count must be between 1 and 4096")
	}
	if err := validateMetadata(e.Metadata); err != nil {
		return zero, zero, err
	}
	return servedInput, servedOutput, nil
}

type eventValidationState struct {
	aggregate   EventAggregate
	incomplete  bool
	costs       map[string]*CostAggregate
	dispatchIDs map[string]struct{}
	known       int
	unknown     int
}

func (s *eventValidationState) addDispatch(index int, dispatch Dispatch) error {
	input, output, err := validateDispatch(index, dispatch)
	if err != nil {
		return err
	}
	if _, exists := s.dispatchIDs[dispatch.DispatchID]; exists {
		return invalid("duplicate dispatch ID %q", dispatch.DispatchID)
	}
	s.dispatchIDs[dispatch.DispatchID] = struct{}{}
	if dispatch.ParentDispatchID != "" && dispatch.ParentDispatchID == dispatch.DispatchID {
		return invalid("dispatch cannot parent itself")
	}
	if dispatch.UsageState == UsageUnknown {
		s.unknown++
		s.incomplete = true
		s.aggregate.IncompleteDispatches, _ = addOne(s.aggregate.IncompleteDispatches)
	} else {
		s.known++
		s.aggregate.InputTokens, err = s.aggregate.InputTokens.Add(input)
		if err != nil {
			return invalid("input token aggregate overflows")
		}
		s.aggregate.OutputTokens, err = s.aggregate.OutputTokens.Add(output)
		if err != nil {
			return invalid("output token aggregate overflows")
		}
	}
	cost := s.costs[dispatch.Cost.Currency]
	if cost == nil {
		cost = &CostAggregate{Currency: dispatch.Cost.Currency}
		s.costs[dispatch.Cost.Currency] = cost
	}
	if dispatch.Cost.State == CostComplete {
		amount, _ := quota.ParseQuotaInteger(dispatch.Cost.Numerator)
		cost.KnownNumerator, err = cost.KnownNumerator.Add(amount)
		if err != nil {
			return invalid("cost aggregate overflows")
		}
		cost.KnownDispatches, _ = addOne(cost.KnownDispatches)
	} else {
		s.incomplete = true
		cost.IncompleteDispatches, _ = addOne(cost.IncompleteDispatches)
	}
	return nil
}

func (s *eventValidationState) validateParents(dispatches []Dispatch) error {
	for _, dispatch := range dispatches {
		if dispatch.ParentDispatchID != "" {
			if _, exists := s.dispatchIDs[dispatch.ParentDispatchID]; !exists {
				return invalid("dispatch %q references an absent parent", dispatch.DispatchID)
			}
		}
	}
	return nil
}

func (s *eventValidationState) finalizeEvidence(evidence EvidenceState, dispatches []Dispatch) error {
	switch {
	case s.known == 0 && s.unknown > 0:
		if evidence != EvidenceUnknown {
			return invalid("evidence state does not match all-unknown dispatches")
		}
		s.aggregate.UsageState = UsageUnknown
	case s.known > 0 && s.unknown > 0:
		if evidence != EvidenceMixed {
			return invalid("evidence state does not match mixed dispatches")
		}
		s.aggregate.UsageState = UsageUnknown
	default:
		if evidence != EvidenceKnown {
			return invalid("evidence state does not match known dispatches")
		}
		s.aggregate.UsageState = UsageKnownZero
		for _, dispatch := range dispatches {
			if dispatch.UsageState == UsageKnownActual {
				s.aggregate.UsageState = UsageKnownActual
				break
			}
		}
	}
	return nil
}

func sortedCostAggregates(costs map[string]*CostAggregate) []CostAggregate {
	keys := make([]string, 0, len(costs))
	for currency := range costs {
		keys = append(keys, currency)
	}
	sort.Strings(keys)
	result := make([]CostAggregate, 0, len(keys))
	for _, currency := range keys {
		result = append(result, *costs[currency])
	}
	return result
}

func (e TerminalEvent) CanonicalDigest() ([]byte, error) {
	if _, err := e.Validate(); err != nil {
		return nil, err
	}
	payload, err := json.Marshal(e)
	if err != nil {
		return nil, fmt.Errorf("%w: encode canonical event: %w", ErrInvalidEvent, err)
	}
	digest := sha256.Sum256(payload)
	return digest[:], nil
}

func EncodeTerminalEvent(e TerminalEvent) (string, error) {
	if _, err := e.Validate(); err != nil {
		return "", err
	}
	payload, err := json.Marshal(e)
	if err != nil {
		return "", fmt.Errorf("encode terminal usage event: %w", err)
	}
	return string(payload), nil
}

func numeratorToDecimal(value quota.QuotaInteger) string {
	digits := value.String()
	if value.IsZero() {
		return "0"
	}
	if len(digits) <= CostScaleDigits {
		digits = strings.Repeat("0", CostScaleDigits-len(digits)+1) + digits
	}
	cut := len(digits) - CostScaleDigits
	whole, fraction := digits[:cut], strings.TrimRight(digits[cut:], "0")
	if fraction == "" {
		return whole
	}
	return whole + "." + fraction
}

func isHexDigest(value string) bool {
	if len(value) != sha256.Size*2 || strings.ToLower(value) != value {
		return false
	}
	decoded, err := hex.DecodeString(value)
	return err == nil && len(decoded) == sha256.Size
}

func parseQuantity(label, value string) (quota.QuotaInteger, error) {
	parsed, err := quota.ParseQuotaInteger(value)
	if err != nil {
		return quota.QuotaInteger{}, invalid("%s: %v", label, err)
	}
	return parsed, nil
}

func addOne(value quota.QuotaInteger) (quota.QuotaInteger, error) {
	one, _ := quota.ParseQuotaInteger("1")
	return value.Add(one)
}

func requireUUID(label, value string, optional bool) error {
	if value == "" && optional {
		return nil
	}
	parsed, err := uuid.Parse(value)
	if err != nil || parsed.String() != strings.ToLower(value) {
		return invalid("%s must be a canonical UUID", label)
	}
	return nil
}

func boundedIdentifier(label, value string, maximum int) error {
	if value == "" || len(value) > maximum || strings.TrimSpace(value) != value || strings.ContainsRune(value, '\x00') {
		return invalid("%s is required and must be bounded, trim-stable, and NUL-free", label)
	}
	return nil
}

func boundedCode(label, value string, required bool) error {
	if value == "" && !required {
		return nil
	}
	if !codePattern.MatchString(value) {
		return invalid("%s is not a bounded canonical code", label)
	}
	return nil
}

func boundedSafeText(label, value string, maximum int, optional bool) error {
	if value == "" && optional {
		return nil
	}
	if value == "" || len(value) > maximum || strings.ContainsRune(value, '\x00') || looksSensitive(value) {
		return invalid("%s is unsafe or exceeds %d bytes", label, maximum)
	}
	return nil
}

func looksSensitive(value string) bool {
	lower := strings.ToLower(value)
	return strings.Contains(lower, "bearer ") || strings.Contains(lower, "vsr_") ||
		strings.Contains(lower, "vsd_") || strings.Contains(lower, "vsm_") ||
		strings.Contains(lower, "api_key=") || strings.Contains(lower, "authorization:")
}

func invalid(format string, args ...any) error {
	return fmt.Errorf("%w: %s", ErrInvalidEvent, fmt.Sprintf(format, args...))
}

func equalDigest(left, right []byte) bool {
	return bytes.Equal(left, right)
}
