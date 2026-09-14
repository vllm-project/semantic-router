package sessiontools

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

const (
	// managerDefaultTTL is used only when the caller leaves the store-level
	// option unset. Decision-level sticky bounds remain per-request values in
	// SelectionInput because one store can serve multiple recipes.
	managerDefaultTTL              = time.Duration(config.ToolSessionStoreDefaultTTLSeconds) * time.Second
	managerDefaultMaxStateBytes    = config.ToolSessionStoreDefaultMaxStateBytes
	managerDefaultOperationTimeout = time.Duration(config.ToolSessionStoreDefaultTimeoutMs) * time.Millisecond
	// DefaultManagerCASRetries is intentionally small: a request must never
	// wait indefinitely for a busy shared session key.
	DefaultManagerCASRetries = 2
)

// SelectionReceiptReason is a bounded reason label for sticky selection
// accounting. Values are deliberately independent of request content or
// storage keys so callers can safely expose them as metrics labels.
type SelectionReceiptReason string

const (
	SelectionReasonNone              SelectionReceiptReason = ""
	SelectionReasonDisabled          SelectionReceiptReason = "disabled"
	SelectionReasonUntrusted         SelectionReceiptReason = "untrusted_identity"
	SelectionReasonInvalidInput      SelectionReceiptReason = "invalid_input"
	SelectionReasonStoreUnavailable  SelectionReceiptReason = "store_unavailable"
	SelectionReasonStateCorrupted    SelectionReceiptReason = "state_corrupted"
	SelectionReasonStateInvalid      SelectionReceiptReason = "state_invalid"
	SelectionReasonStateExpired      SelectionReceiptReason = "state_expired"
	SelectionReasonPolicyChanged     SelectionReceiptReason = "policy_changed"
	SelectionReasonCatalogChanged    SelectionReceiptReason = "catalog_changed"
	SelectionReasonCapabilityChanged SelectionReceiptReason = "capability_changed"
	SelectionReasonStateTooLarge     SelectionReceiptReason = "state_too_large"
	SelectionReasonCASConflict       SelectionReceiptReason = "cas_conflict"
)

// ManagerOptions configures store-level behavior shared by all decision
// policies using a Manager. Zero values use the bounded repository defaults;
// MaxCASRetries is the one exception: zero deliberately disables retries.
type ManagerOptions struct {
	TTL              time.Duration
	MaxStateBytes    int
	OperationTimeout time.Duration
	MaxCASRetries    int
	Clock            func() time.Time
}

// DefaultManagerOptions returns the repository's bounded manager defaults.
// Callers may override MaxCASRetries with zero when an immediate, single-shot
// update is preferred.
func DefaultManagerOptions() ManagerOptions {
	return ManagerOptions{
		TTL:              managerDefaultTTL,
		MaxStateBytes:    managerDefaultMaxStateBytes,
		OperationTimeout: managerDefaultOperationTimeout,
		MaxCASRetries:    DefaultManagerCASRetries,
		Clock:            time.Now,
	}
}

// SelectionInput contains the current request's already-authorized catalog
// and ordinary request-time selection. Historical state is advisory only: the
// manager rechecks every retained identity against Authorized before merging.
// MaxTools, MaxNewToolsPerTurn, and PinCalledTools are decision-level values
// and therefore belong here rather than in ManagerOptions.
type SelectionInput struct {
	Enabled bool
	Trusted bool
	Key     string
	Quota   QuotaKey

	PolicyFingerprint     string
	CatalogFingerprint    string
	CapabilityFingerprint string

	Authorized      []ToolCandidate
	Selected        []ToolCandidate
	CalledToolNames []string
	Turn            int

	MaxTools           int
	MaxNewToolsPerTurn int
	PinCalledTools     bool
	StrategyID         string
}

// SelectionReceipt is content-minimized accounting for one manager call.
// Selected tool names and definitions are returned separately and are never
// embedded in the receipt.
type SelectionReceipt struct {
	Merge MergeReceipt

	Reused      bool
	Committed   bool
	Invalidated bool
	Fallback    bool
	Reason      SelectionReceiptReason
	CASRetries  int
}

// SelectionResult contains the final identity-only selection. When sticky is
// disabled, identity is untrusted, or storage is unavailable, Selected is an
// unchanged copy of the ordinary request-time selection and State is zero.
// When the manager commits successfully, State contains the corresponding
// bounded envelope and Selected is derived from State.Tools.
type SelectionResult struct {
	Selected []ToolCandidate
	State    State
	Receipt  SelectionReceipt
}

// Manager owns request-time state validation, deterministic merge, and the
// bounded CAS retry/fallback policy layered over Store. It does not authorize
// tools and never persists full tool definitions.
type Manager struct {
	store   Store
	options ManagerOptions
}

// NewManager constructs a sticky-selection manager over store. The store is
// owned by the caller; Router lifecycle code should close it through its
// resource scope rather than through Manager.
func NewManager(store Store, options ManagerOptions) (*Manager, error) {
	if store == nil {
		return nil, fmt.Errorf("sessiontools: manager store must not be nil")
	}
	normalized, err := normalizeManagerOptions(options)
	if err != nil {
		return nil, err
	}
	return &Manager{store: store, options: normalized}, nil
}

func normalizeManagerOptions(options ManagerOptions) (ManagerOptions, error) {
	if options.TTL == 0 {
		options.TTL = managerDefaultTTL
	}
	if options.MaxStateBytes == 0 {
		options.MaxStateBytes = managerDefaultMaxStateBytes
	}
	if options.OperationTimeout == 0 {
		options.OperationTimeout = managerDefaultOperationTimeout
	}
	if options.TTL < 0 {
		return ManagerOptions{}, fmt.Errorf("sessiontools: manager ttl must be greater than zero")
	}
	if options.MaxStateBytes < 0 {
		return ManagerOptions{}, fmt.Errorf("sessiontools: manager max_state_bytes must not be negative")
	}
	if options.OperationTimeout < 0 {
		return ManagerOptions{}, fmt.Errorf("sessiontools: manager operation timeout must not be negative")
	}
	if options.MaxCASRetries < 0 {
		return ManagerOptions{}, fmt.Errorf("sessiontools: manager max CAS retries must not be negative")
	}
	if options.Clock == nil {
		options.Clock = time.Now
	}
	return options, nil
}

// Select applies sticky state to one already-authorized request. Every
// storage, validation, and CAS failure fails open to the ordinary selection;
// the returned receipt identifies the bounded reason and no error escapes the
// request path. Configuration errors are rejected by NewManager instead.
func (m *Manager) Select(ctx context.Context, input SelectionInput) SelectionResult {
	result := SelectionResult{
		Selected: cloneCandidates(input.Selected),
	}
	if m == nil {
		result.Receipt = SelectionReceipt{Fallback: true, Reason: SelectionReasonStoreUnavailable}
		return result
	}
	if !input.Enabled {
		result.Receipt.Reason = SelectionReasonDisabled
		return result
	}
	if !input.Trusted {
		result.Receipt = SelectionReceipt{Fallback: true, Reason: SelectionReasonUntrusted}
		return result
	}
	if err := validateSelectionInput(input); err != nil {
		result.Receipt = SelectionReceipt{Fallback: true, Reason: SelectionReasonInvalidInput}
		return result
	}
	if ctx == nil {
		ctx = context.Background()
	}
	operationCtx, cancel := context.WithTimeout(ctx, m.options.OperationTimeout)
	defer cancel()
	return m.selectWithStore(operationCtx, input, result)
}

// Update is an explicit alias for Select for callers that describe the
// operation in terms of committing the next session state.
func (m *Manager) Update(ctx context.Context, input SelectionInput) SelectionResult {
	return m.Select(ctx, input)
}

func validateSelectionInput(input SelectionInput) error {
	if strings.TrimSpace(input.Key) == "" {
		return fmt.Errorf("sessiontools: sticky selection key must not be empty")
	}
	if strings.TrimSpace(input.Quota.Principal) == "" || strings.TrimSpace(input.Quota.Namespace) == "" {
		return fmt.Errorf("sessiontools: sticky selection quota identity must not be empty")
	}
	if strings.TrimSpace(input.PolicyFingerprint) == "" ||
		strings.TrimSpace(input.CatalogFingerprint) == "" ||
		strings.TrimSpace(input.CapabilityFingerprint) == "" {
		return fmt.Errorf("sessiontools: sticky selection fingerprints must all be set")
	}
	if input.Turn < 0 {
		return fmt.Errorf("sessiontools: sticky selection turn must not be negative")
	}
	if input.MaxTools <= 0 {
		return fmt.Errorf("sessiontools: sticky selection max_tools must be greater than zero")
	}
	if input.MaxNewToolsPerTurn < 0 || input.MaxNewToolsPerTurn > input.MaxTools {
		return fmt.Errorf("sessiontools: sticky selection max_new_tools_per_turn is out of bounds")
	}
	return validateCandidates(input.Authorized, "authorized")
}

type stateLoadResult struct {
	state       State
	expected    uint64
	found       bool
	invalidated bool
	retry       bool
	reason      SelectionReceiptReason
	fallback    SelectionReceiptReason
}

func (m *Manager) selectWithStore(
	ctx context.Context,
	input SelectionInput,
	result SelectionResult,
) SelectionResult {
	for attempt := 0; attempt <= m.options.MaxCASRetries; attempt++ {
		loaded := m.loadState(ctx, input)
		if loaded.retry {
			if attempt < m.options.MaxCASRetries {
				result.Receipt.CASRetries++
				continue
			}
			result.Receipt.Fallback = true
			result.Receipt.Reason = SelectionReasonCASConflict
			return result
		}
		if loaded.fallback != SelectionReasonNone {
			result.Receipt.Fallback = true
			result.Receipt.Reason = loaded.fallback
			return result
		}
		result.Receipt.Invalidated = result.Receipt.Invalidated || loaded.invalidated
		if loaded.reason != SelectionReasonNone {
			result.Receipt.Reason = loaded.reason
		}

		merged, err := MergeToolSet(MergeInput{
			Previous:           loaded.state,
			Authorized:         input.Authorized,
			Selected:           input.Selected,
			CalledToolNames:    input.CalledToolNames,
			Turn:               input.Turn,
			MaxTools:           input.MaxTools,
			MaxNewToolsPerTurn: input.MaxNewToolsPerTurn,
			PinCalledTools:     input.PinCalledTools,
		})
		if err != nil {
			result.Receipt.Fallback = true
			result.Receipt.Reason = SelectionReasonInvalidInput
			return result
		}

		next, err := m.buildNextState(loaded.state, merged.State, input, loaded.expected)
		if err != nil {
			result.Receipt.Fallback = true
			result.Receipt.Reason = stateBuildFailureReason(err)
			return result
		}
		if err := next.Validate(input.MaxTools, m.options.MaxStateBytes); err != nil {
			result.Receipt.Fallback = true
			result.Receipt.Reason = SelectionReasonStateTooLarge
			return result
		}

		applied, casErr := m.store.CompareAndSwap(ctx, input.Key, loaded.expected, next, m.options.TTL, input.Quota)
		if applied && casErr == nil {
			result.State = next.Clone()
			result.Selected = candidatesFromState(next)
			result.Receipt.Merge = merged.Receipt
			result.Receipt.Reused = loaded.found
			result.Receipt.Committed = true
			return result
		}
		if errors.Is(casErr, ErrRevisionMismatch) || (!applied && casErr == nil) {
			if attempt < m.options.MaxCASRetries {
				result.Receipt.CASRetries++
				continue
			}
			result.Receipt.Fallback = true
			result.Receipt.Reason = SelectionReasonCASConflict
			return result
		}
		result.Receipt.Fallback = true
		result.Receipt.Reason = SelectionReasonStoreUnavailable
		return result
	}
	result.Receipt.Fallback = true
	result.Receipt.Reason = SelectionReasonCASConflict
	return result
}

func (m *Manager) loadState(ctx context.Context, input SelectionInput) stateLoadResult {
	loaded, metadata, err := loadWithMetadata(m.store, ctx, input.Key)
	if err != nil {
		if !errors.Is(err, ErrStateCorrupted) {
			return stateLoadResult{fallback: SelectionReasonStoreUnavailable}
		}
		// Store implementations own best-effort removal of unreadable values
		// (see ErrStateCorrupted's contract). Do not issue an unconditional
		// delete here: a concurrent writer may have replaced the value after
		// the failed read.
		return stateLoadResult{invalidated: true, reason: SelectionReasonStateCorrupted}
	}
	if metadata.Expired {
		// LoadMetadataStore owns expiry cleanup and makes the value logically
		// absent before returning this result. The token remains receipt/audit
		// metadata; deleting it again would race a concurrent readmission.
		return stateLoadResult{invalidated: true, reason: SelectionReasonStateExpired}
	}
	if !loaded.Found {
		return stateLoadResult{}
	}

	state := loaded.State
	token := StateToken{
		Revision:   state.Revision,
		Generation: metadata.ObservedGeneration,
	}
	if err := state.Validate(input.MaxTools, m.options.MaxStateBytes); err != nil {
		return m.invalidateState(ctx, input.Key, token, SelectionReasonStateInvalid)
	}
	if !state.ExpiresAt.After(m.options.Clock()) {
		return m.invalidateState(ctx, input.Key, token, SelectionReasonStateExpired)
	}
	if reason := fingerprintMismatchReason(state, input); reason != SelectionReasonNone {
		return m.invalidateState(ctx, input.Key, token, reason)
	}
	return stateLoadResult{
		state:    state,
		expected: state.Revision,
		found:    true,
	}
}

func loadWithMetadata(store Store, ctx context.Context, key string) (VersionedState, LoadMetadata, error) {
	if metadataStore, ok := store.(LoadMetadataStore); ok {
		return metadataStore.LoadWithMetadata(ctx, key)
	}
	loaded, err := store.Load(ctx, key)
	return loaded, LoadMetadata{}, err
}

func (m *Manager) invalidateState(
	ctx context.Context,
	key string,
	token StateToken,
	reason SelectionReceiptReason,
) stateLoadResult {
	conditional, ok := m.store.(ConditionalDeleteStore)
	if !ok || token.Revision == 0 {
		// Compatibility-only Store implementations cannot safely invalidate a
		// value after releasing Load: an unconditional Delete could remove a
		// newer concurrent state. Fail open to this request's ordinary selection.
		return stateLoadResult{fallback: reason}
	}
	deleted, err := conditional.DeleteIfToken(ctx, key, token)
	if err != nil {
		return stateLoadResult{fallback: SelectionReasonStoreUnavailable}
	}
	if !deleted {
		return stateLoadResult{retry: true}
	}
	return stateLoadResult{invalidated: true, reason: reason}
}

func fingerprintMismatchReason(state State, input SelectionInput) SelectionReceiptReason {
	if state.PolicyFingerprint != input.PolicyFingerprint {
		return SelectionReasonPolicyChanged
	}
	if state.CatalogFingerprint != input.CatalogFingerprint {
		return SelectionReasonCatalogChanged
	}
	if state.CapabilityFingerprint != input.CapabilityFingerprint {
		return SelectionReasonCapabilityChanged
	}
	return SelectionReasonNone
}

func (m *Manager) buildNextState(previous, merged State, input SelectionInput, expected uint64) (State, error) {
	if expected == ^uint64(0) {
		return State{}, fmt.Errorf("sessiontools: state revision overflow")
	}
	now := m.options.Clock()
	next := merged.Clone()
	next.SchemaVersion = SchemaVersion
	next.Revision = expected + 1
	next.PolicyFingerprint = input.PolicyFingerprint
	next.CatalogFingerprint = input.CatalogFingerprint
	next.CapabilityFingerprint = input.CapabilityFingerprint
	next.StrategyID = input.StrategyID
	next.CreatedAt = now
	if !previous.CreatedAt.IsZero() {
		next.CreatedAt = previous.CreatedAt
	}
	next.LastSeenAt = now
	next.ExpiresAt = now.Add(m.options.TTL)
	return next, nil
}

func stateBuildFailureReason(err error) SelectionReceiptReason {
	if strings.Contains(err.Error(), "revision overflow") {
		return SelectionReasonStateInvalid
	}
	return SelectionReasonInvalidInput
}

func cloneCandidates(candidates []ToolCandidate) []ToolCandidate {
	if candidates == nil {
		return nil
	}
	return append([]ToolCandidate(nil), candidates...)
}

func candidatesFromState(state State) []ToolCandidate {
	if len(state.Tools) == 0 {
		return []ToolCandidate{}
	}
	candidates := make([]ToolCandidate, len(state.Tools))
	for index, tool := range state.Tools {
		candidates[index] = ToolCandidate{
			Name:                  tool.Name,
			DefinitionFingerprint: tool.DefinitionFingerprint,
		}
	}
	return candidates
}
