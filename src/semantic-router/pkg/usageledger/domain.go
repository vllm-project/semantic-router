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
	if e.Schema != TerminalEventSchema {
		return EventAggregate{}, invalid("schema must be %q", TerminalEventSchema)
	}
	if err := requireUUID("event ID", e.EventID, false); err != nil {
		return EventAggregate{}, err
	}
	if err := requireUUID("namespace ID", e.NamespaceID, false); err != nil {
		return EventAggregate{}, err
	}
	if err := boundedIdentifier("admission ID", e.AdmissionID, 256); err != nil {
		return EventAggregate{}, err
	}
	if !isHexDigest(e.FinalizationDigest) {
		return EventAggregate{}, invalid("finalization digest must be 32-byte lowercase hex")
	}
	if e.ExternalRequestID != "" {
		if err := boundedIdentifier("external request ID", e.ExternalRequestID, 256); err != nil {
			return EventAggregate{}, err
		}
		if looksSensitive(e.ExternalRequestID) {
			return EventAggregate{}, invalid("external request ID is unsafe")
		}
	}
	if e.ReplayID != "" {
		if err := boundedIdentifier("replay ID", e.ReplayID, 256); err != nil {
			return EventAggregate{}, err
		}
		if looksSensitive(e.ReplayID) {
			return EventAggregate{}, invalid("replay ID is unsafe")
		}
	}
	if err := boundedCode("protocol", e.Protocol, true); err != nil {
		return EventAggregate{}, err
	}
	if e.Path == "" || len(e.Path) > 2048 || !strings.HasPrefix(e.Path, "/") || strings.ContainsAny(e.Path, "?#\x00") {
		return EventAggregate{}, invalid("path must be a normalized path without query or fragment")
	}
	if looksSensitive(e.Path) {
		return EventAggregate{}, invalid("path is unsafe")
	}
	if e.StatusCode < 100 || e.StatusCode > 599 {
		return EventAggregate{}, invalid("status code is outside HTTP range")
	}
	if err := boundedCode("error code", e.ErrorCode, false); err != nil {
		return EventAggregate{}, err
	}
	if e.OccurredAt.IsZero() || e.CompletedAt.IsZero() || e.CompletedAt.Before(e.OccurredAt) {
		return EventAggregate{}, invalid("occurred/completed timestamps are required and ordered")
	}
	if e.LatencyMilliseconds < 0 || e.LatencyMilliseconds != e.CompletedAt.Sub(e.OccurredAt).Milliseconds() {
		return EventAggregate{}, invalid("latency must exactly match terminal timestamps")
	}
	if e.TTFTMilliseconds != nil && (*e.TTFTMilliseconds < 0 || *e.TTFTMilliseconds > e.LatencyMilliseconds) {
		return EventAggregate{}, invalid("TTFT is outside request lifetime")
	}
	if err := validatePrincipal(e.Principal); err != nil {
		return EventAggregate{}, err
	}
	if err := validateRouting(e.Routing); err != nil {
		return EventAggregate{}, err
	}
	servedInput, err := parseQuantity("served input tokens", e.Served.InputTokens)
	if err != nil {
		return EventAggregate{}, err
	}
	servedOutput, err := parseQuantity("served output tokens", e.Served.OutputTokens)
	if err != nil {
		return EventAggregate{}, err
	}
	if !e.Served.InputKnown && !servedInput.IsZero() || !e.Served.OutputKnown && !servedOutput.IsZero() {
		return EventAggregate{}, invalid("unknown served usage cannot claim token values")
	}
	if len(e.Dispatches) == 0 || len(e.Dispatches) > 4096 {
		return EventAggregate{}, invalid("dispatch count must be between 1 and 4096")
	}
	if err := validateMetadata(e.Metadata); err != nil {
		return EventAggregate{}, err
	}

	aggregate := EventAggregate{ServedInputTokens: servedInput, ServedOutputTokens: servedOutput}
	costs := make(map[string]*CostAggregate)
	dispatchIDs := make(map[string]struct{}, len(e.Dispatches))
	known, unknown := 0, 0
	for index, dispatch := range e.Dispatches {
		input, output, err := validateDispatch(index, dispatch)
		if err != nil {
			return EventAggregate{}, err
		}
		if _, exists := dispatchIDs[dispatch.DispatchID]; exists {
			return EventAggregate{}, invalid("duplicate dispatch ID %q", dispatch.DispatchID)
		}
		dispatchIDs[dispatch.DispatchID] = struct{}{}
		if dispatch.ParentDispatchID != "" && dispatch.ParentDispatchID == dispatch.DispatchID {
			return EventAggregate{}, invalid("dispatch cannot parent itself")
		}
		if dispatch.UsageState == UsageUnknown {
			unknown++
			aggregate.IncompleteDispatches, _ = addOne(aggregate.IncompleteDispatches)
		} else {
			known++
			aggregate.InputTokens, err = aggregate.InputTokens.Add(input)
			if err != nil {
				return EventAggregate{}, invalid("input token aggregate overflows")
			}
			aggregate.OutputTokens, err = aggregate.OutputTokens.Add(output)
			if err != nil {
				return EventAggregate{}, invalid("output token aggregate overflows")
			}
		}
		cost := costs[dispatch.Cost.Currency]
		if cost == nil {
			cost = &CostAggregate{Currency: dispatch.Cost.Currency}
			costs[dispatch.Cost.Currency] = cost
		}
		if dispatch.Cost.State == CostComplete {
			amount, _ := quota.ParseQuotaInteger(dispatch.Cost.Numerator)
			cost.KnownNumerator, err = cost.KnownNumerator.Add(amount)
			if err != nil {
				return EventAggregate{}, invalid("cost aggregate overflows")
			}
			cost.KnownDispatches, _ = addOne(cost.KnownDispatches)
		} else {
			cost.IncompleteDispatches, _ = addOne(cost.IncompleteDispatches)
		}
	}
	for _, dispatch := range e.Dispatches {
		if dispatch.ParentDispatchID != "" {
			if _, exists := dispatchIDs[dispatch.ParentDispatchID]; !exists {
				return EventAggregate{}, invalid("dispatch %q references an absent parent", dispatch.DispatchID)
			}
		}
	}
	switch {
	case known == 0 && unknown > 0:
		if e.EvidenceState != EvidenceUnknown {
			return EventAggregate{}, invalid("evidence state does not match all-unknown dispatches")
		}
		aggregate.UsageState = UsageUnknown
	case known > 0 && unknown > 0:
		if e.EvidenceState != EvidenceMixed {
			return EventAggregate{}, invalid("evidence state does not match mixed dispatches")
		}
		aggregate.UsageState = UsageUnknown
	default:
		if e.EvidenceState != EvidenceKnown {
			return EventAggregate{}, invalid("evidence state does not match known dispatches")
		}
		aggregate.UsageState = UsageKnownZero
		for _, dispatch := range e.Dispatches {
			if dispatch.UsageState == UsageKnownActual {
				aggregate.UsageState = UsageKnownActual
				break
			}
		}
	}
	if unknown > 0 && e.Fence != nil {
		if err := validateFence(*e.Fence); err != nil {
			return EventAggregate{}, err
		}
	} else if unknown == 0 && e.Fence != nil {
		return EventAggregate{}, invalid("a known event cannot open an unknown-usage fence")
	}
	if err := validateReceipts(e.QuotaReceipts); err != nil {
		return EventAggregate{}, err
	}
	keys := make([]string, 0, len(costs))
	for currency := range costs {
		keys = append(keys, currency)
	}
	sort.Strings(keys)
	for _, currency := range keys {
		aggregate.Costs = append(aggregate.Costs, *costs[currency])
	}
	return aggregate, nil
}

func validateDispatch(index int, d Dispatch) (quota.QuotaInteger, quota.QuotaInteger, error) {
	zero := quota.QuotaInteger{}
	if err := boundedIdentifier("dispatch ID", d.DispatchID, 256); err != nil {
		return zero, zero, err
	}
	if d.ParentDispatchID != "" {
		if err := boundedIdentifier("parent dispatch ID", d.ParentDispatchID, 256); err != nil {
			return zero, zero, err
		}
	}
	if d.Ordinal < 0 || d.Ordinal != index {
		return zero, zero, invalid("dispatch ordinals must be contiguous and ordered")
	}
	if err := boundedCode("dispatch type", d.DispatchType, true); err != nil {
		return zero, zero, err
	}
	if d.DecisionTier < 0 {
		return zero, zero, invalid("decision tier cannot be negative")
	}
	if err := boundedCode("parallel group", d.ParallelGroup, false); err != nil {
		return zero, zero, err
	}
	for label, value := range map[string]string{
		"decision ID": d.DecisionID, "model ID": d.ModelID,
	} {
		if err := boundedCode(label, value, false); err != nil {
			return zero, zero, err
		}
	}
	if err := requireUUID("backend ID", d.BackendID, true); err != nil {
		return zero, zero, err
	}
	for label, value := range map[string]string{
		"decision display snapshot": d.DecisionName,
		"model display snapshot":    d.ModelName,
		"provider Model ID":         d.ProviderModelID,
	} {
		if err := boundedSafeText(label, value, 512, true); err != nil {
			return zero, zero, err
		}
	}
	if d.ModelID != "" && d.ModelRevision <= 0 {
		return zero, zero, invalid("model revision is required with a model ID")
	}
	if d.PricingRevision < 0 {
		return zero, zero, invalid("pricing revision cannot be negative")
	}
	for label, value := range map[string]string{
		"provider ID": d.ProviderID, "retry class": d.RetryClass, "cache state": d.CacheState,
	} {
		if err := boundedCode(label, value, false); err != nil {
			return zero, zero, err
		}
	}
	input, err := parseQuantity("dispatch input tokens", d.InputTokens)
	if err != nil {
		return zero, zero, err
	}
	cacheRead, err := parseQuantity("dispatch cache-read tokens", d.CacheReadTokens)
	if err != nil {
		return zero, zero, err
	}
	cacheWrite, err := parseQuantity("dispatch cache-write tokens", d.CacheWriteTokens)
	if err != nil {
		return zero, zero, err
	}
	output, err := parseQuantity("dispatch output tokens", d.OutputTokens)
	if err != nil {
		return zero, zero, err
	}
	cacheTotal, err := cacheRead.Add(cacheWrite)
	if err != nil || cacheTotal.Compare(input) > 0 {
		return zero, zero, invalid("dispatch cache buckets exceed input tokens")
	}
	switch d.UsageState {
	case UsageKnownZero:
		if !input.IsZero() || !output.IsZero() || !cacheRead.IsZero() || !cacheWrite.IsZero() || d.UnknownReason != "" {
			return zero, zero, invalid("known-zero dispatch must contain zero usage and no unknown reason")
		}
	case UsageKnownActual:
		if d.UnknownReason != "" {
			return zero, zero, invalid("known-actual dispatch cannot carry an unknown reason")
		}
	case UsageUnknown:
		if !input.IsZero() || !output.IsZero() || !cacheRead.IsZero() || !cacheWrite.IsZero() {
			return zero, zero, invalid("unknown dispatch cannot claim token values")
		}
		if err := boundedCode("unknown reason", d.UnknownReason, true); err != nil {
			return zero, zero, err
		}
	default:
		return zero, zero, invalid("unsupported dispatch usage state %q", d.UsageState)
	}
	if err := validateCost(d.Cost); err != nil {
		return zero, zero, err
	}
	if !isHexDigest(d.EvidenceDigest) {
		return zero, zero, invalid("dispatch evidence digest must be 32-byte lowercase hex")
	}
	if d.StartedAt.IsZero() || d.CompletedAt.IsZero() || d.CompletedAt.Before(d.StartedAt) {
		return zero, zero, invalid("dispatch timestamps are required and ordered")
	}
	if len(d.Attempts) == 0 || len(d.Attempts) > 6 {
		return zero, zero, invalid("dispatch must contain between 1 and 6 attempts")
	}
	seen := make(map[string]struct{}, len(d.Attempts))
	for ordinal, attempt := range d.Attempts {
		if err := validateAttempt(ordinal, attempt, d.StartedAt, d.CompletedAt); err != nil {
			return zero, zero, err
		}
		if _, exists := seen[attempt.AttemptID]; exists {
			return zero, zero, invalid("duplicate attempt ID %q", attempt.AttemptID)
		}
		seen[attempt.AttemptID] = struct{}{}
		if ordinal < len(d.Attempts)-1 && attempt.State != UsageKnownZero {
			return zero, zero, invalid("only proven known-zero attempts may precede a retry")
		}
	}
	terminal := d.Attempts[len(d.Attempts)-1]
	if d.UsageState == UsageKnownZero && terminal.State != UsageKnownZero ||
		d.UsageState == UsageKnownActual && terminal.State != UsageKnownActual ||
		d.UsageState == UsageUnknown && terminal.State != UsageUnknown {
		return zero, zero, invalid("terminal attempt state does not match dispatch usage state")
	}
	return input, output, nil
}

func validateAttempt(ordinal int, a Attempt, dispatchStart, dispatchEnd time.Time) error {
	if err := boundedIdentifier("attempt ID", a.AttemptID, 256); err != nil {
		return err
	}
	if a.Ordinal != ordinal {
		return invalid("attempt ordinals must be contiguous and ordered")
	}
	if err := requireUUID("attempt backend ID", a.BackendID, true); err != nil {
		return err
	}
	if err := boundedCode("attempt provider ID", a.ProviderID, false); err != nil {
		return err
	}
	if a.State != UsageKnownZero && a.State != UsageKnownActual && a.State != UsageUnknown {
		return invalid("unsupported attempt state %q", a.State)
	}
	if a.StatusCode != 0 && (a.StatusCode < 100 || a.StatusCode > 599) {
		return invalid("attempt status code is outside HTTP range")
	}
	if err := boundedCode("attempt error code", a.ErrorCode, false); err != nil {
		return err
	}
	if a.StartedAt.Before(dispatchStart) || a.CompletedAt.After(dispatchEnd) || a.CompletedAt.Before(a.StartedAt) {
		return invalid("attempt timestamps are outside dispatch lifetime")
	}
	return nil
}

func validateCost(cost DispatchCost) error {
	if !currencyPattern.MatchString(cost.Currency) {
		return invalid("dispatch cost requires an ISO-4217 currency")
	}
	amount, err := parseQuantity("dispatch cost numerator", cost.Numerator)
	if err != nil {
		return err
	}
	switch cost.State {
	case CostComplete:
		if cost.Reason != "" {
			return invalid("complete cost cannot carry an unknown reason")
		}
	case CostUnknown:
		if !amount.IsZero() {
			return invalid("unknown cost cannot claim a numerator")
		}
		if err := boundedCode("unknown cost reason", cost.Reason, true); err != nil {
			return err
		}
	default:
		return invalid("unsupported cost state %q", cost.State)
	}
	return nil
}

func validatePrincipal(p PrincipalSnapshot) error {
	for label, value := range map[string]string{
		"API key ID": p.APIKeyID, "credential ID": p.CredentialID, "user ID": p.UserID, "team ID": p.TeamID,
	} {
		if err := requireUUID(label, value, true); err != nil {
			return err
		}
	}
	if p.APIKeyID == "" {
		return invalid("API key ID is required")
	}
	for label, value := range map[string]string{
		"API key display snapshot": p.APIKeyName, "user display snapshot": p.UserName, "team display snapshot": p.TeamName,
	} {
		if err := boundedSafeText(label, value, 256, true); err != nil {
			return err
		}
	}
	return nil
}

func validateRouting(r RoutingSnapshot) error {
	for label, value := range map[string]string{
		"entrypoint ID": r.EntrypointID, "entrypoint rule ID": r.EntrypointRuleID, "recipe ID": r.RecipeID,
	} {
		if err := boundedCode(label, value, false); err != nil {
			return err
		}
	}
	if r.RoutingRevision < 0 || r.AccessRevision < 0 || r.RecipeRevision < 0 {
		return invalid("routing revisions cannot be negative")
	}
	for _, value := range []string{r.EntrypointName, r.EntrypointRuleName, r.RecipeName} {
		if err := boundedSafeText("routing display snapshot", value, 256, true); err != nil {
			return err
		}
	}
	return nil
}

func validateMetadata(metadata map[string]string) error {
	allowed := map[string]struct{}{
		"client_name": {}, "client_version": {}, "correlation_id": {}, "content_type": {},
		"request_digest": {}, "sdk": {}, "sdk_version": {}, "user_agent_family": {},
	}
	for key, value := range metadata {
		if !metaPattern.MatchString(key) {
			return invalid("metadata key %q is not canonical", key)
		}
		if _, ok := allowed[key]; !ok {
			return invalid("metadata key %q is not in the safe allowlist", key)
		}
		if len(value) > 256 || strings.ContainsRune(value, '\x00') || looksSensitive(value) {
			return invalid("metadata value for %q is unsafe", key)
		}
	}
	return nil
}

func validateFence(f UnknownFence) error {
	if err := requireUUID("fence ID", f.FenceID, false); err != nil {
		return err
	}
	if err := boundedCode("fence reason", f.Reason, true); err != nil {
		return err
	}
	if len(f.Bindings) == 0 || len(f.Bindings) > 256 {
		return invalid("fence must identify affected bindings")
	}
	seen := make(map[string]struct{}, len(f.Bindings))
	for _, binding := range f.Bindings {
		if err := requireUUID("fence binding ID", binding.BindingID, false); err != nil {
			return err
		}
		if err := requireUUID("fence rule ID", binding.RuleID, false); err != nil {
			return err
		}
		key := binding.BindingID + "/" + binding.RuleID
		if _, exists := seen[key]; exists {
			return invalid("duplicate fence binding %q", key)
		}
		seen[key] = struct{}{}
		for label, value := range map[string]string{"admission limit": binding.AdmissionLimit, "maximum debit": binding.MaximumDebit} {
			if value != "" {
				if _, err := parseQuantity(label, value); err != nil {
					return err
				}
			}
		}
	}
	return nil
}

func validateReceipts(receipts []QuotaReceipt) error {
	if len(receipts) > 512 {
		return invalid("too many quota receipts")
	}
	seen := make(map[string]struct{}, len(receipts))
	for _, receipt := range receipts {
		if err := requireUUID("quota binding ID", receipt.BindingID, false); err != nil {
			return err
		}
		if err := requireUUID("quota rule ID", receipt.RuleID, false); err != nil {
			return err
		}
		if err := boundedCode("quota metric", receipt.Metric, true); err != nil {
			return err
		}
		if _, err := parseQuantity("quota receipt amount", receipt.Amount); err != nil {
			return err
		}
		key := receipt.BindingID + "/" + receipt.RuleID
		if _, exists := seen[key]; exists {
			return invalid("duplicate quota receipt %q", key)
		}
		seen[key] = struct{}{}
	}
	return nil
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
