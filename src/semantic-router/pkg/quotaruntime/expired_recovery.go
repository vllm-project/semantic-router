package quotaruntime

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/google/uuid"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/quota"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/usageledger"
)

const (
	maximumRecoveryDispatches    = 256
	maximumRecoveryAttemptFields = 8192
	expiredAdmissionReason       = "admission_lease_expired"
)

// ExpiredRecoveryResult describes one bounded partition recovery pass. A
// caller can safely run the pass concurrently on every Router replica:
// Finalize compare-and-sets the attempt revision and terminal identity.
type ExpiredRecoveryResult struct {
	Found         bool
	Recovered     bool
	Idempotent    bool
	Retry         bool
	AdmissionID   string
	EvidenceState string
	StreamID      string
	ServerTime    time.Time
}

type expiredAdmissionSnapshot struct {
	partition        string
	admissionID      string
	admissionDigest  string
	deadline         time.Time
	serverTime       time.Time
	record           admissionRecoveryRecord
	dispatches       []expiredDispatch
	evidenceRevision uint64
}

type expiredDispatch struct {
	id               string
	ordinal          uint32
	planDigest       string
	evidenceModelID  string
	evidenceModelRev int64
	evidencePresent  bool
}

type expiredAdmissionReadState string

const (
	expiredAdmissionMissing  expiredAdmissionReadState = "missing"
	expiredAdmissionPending  expiredAdmissionReadState = "expired"
	expiredAdmissionRenewed  expiredAdmissionReadState = "renewed"
	expiredAdmissionTerminal expiredAdmissionReadState = "terminal"
)

// RecoverOldestExpiredAdmission settles at most one expired lease in a quota
// partition. It never guesses usage: a dispatch without BeginAttempt evidence
// is proven not sent and settles known-zero; any started attempt without
// authoritative usage settles unknown and fences every affected binding.
func (e *RedisEngine) RecoverOldestExpiredAdmission(
	ctx context.Context,
	partition string,
) (ExpiredRecoveryResult, error) {
	if e == nil || e.client == nil {
		return ExpiredRecoveryResult{}, fmt.Errorf("%w: quota runtime is unavailable", ErrRuntimeUnavailable)
	}
	snapshot, found, err := e.readOldestExpiredAdmission(ctx, partition)
	if err != nil || !found {
		return ExpiredRecoveryResult{Found: found}, err
	}
	request, buildErr := e.buildExpiredFinalization(ctx, snapshot)
	if buildErr != nil {
		return ExpiredRecoveryResult{Found: true, AdmissionID: snapshot.admissionID}, buildErr
	}
	settled, finalizeErr := e.Finalize(ctx, request)
	if finalizeErr != nil {
		if errors.Is(finalizeErr, ErrConflict) || errors.Is(finalizeErr, ErrAdmissionNotFound) ||
			errors.Is(finalizeErr, ErrEvidenceChanged) {
			state, observeErr := e.observeExpiredAdmissionRace(ctx, snapshot)
			if observeErr != nil {
				return ExpiredRecoveryResult{Found: true, AdmissionID: snapshot.admissionID}, observeErr
			}
			switch state {
			case expiredAdmissionTerminal:
				return ExpiredRecoveryResult{
					Found: true, Recovered: true, Idempotent: true, Retry: true,
					AdmissionID: snapshot.admissionID,
				}, nil
			case expiredAdmissionRenewed:
				return ExpiredRecoveryResult{}, nil
			case expiredAdmissionPending:
				return ExpiredRecoveryResult{
					Found: true, Retry: true, AdmissionID: snapshot.admissionID,
				}, nil
			case expiredAdmissionMissing:
				return ExpiredRecoveryResult{Found: true, AdmissionID: snapshot.admissionID}, fmt.Errorf(
					"%w: expired admission disappeared without a terminal marker",
					ErrRuntimeCorrupt,
				)
			}
		}
		return ExpiredRecoveryResult{Found: true, AdmissionID: snapshot.admissionID}, finalizeErr
	}
	return ExpiredRecoveryResult{
		Found: true, Recovered: true, Idempotent: settled.Idempotent, Retry: true,
		AdmissionID: snapshot.admissionID, EvidenceState: settled.EvidenceState,
		StreamID: settled.StreamID, ServerTime: settled.ServerTime,
	}, nil
}

func (e *RedisEngine) readOldestExpiredAdmission(
	ctx context.Context,
	partition string,
) (expiredAdmissionSnapshot, bool, error) {
	keys, err := newPartitionKeysWithPrefix(e.keyPrefix, partition)
	if err != nil {
		return expiredAdmissionSnapshot{}, false, err
	}
	value, err := nextExpiredScript.Run(ctx, e.client, []string{keys.pendingIndex}).Result()
	if err != nil {
		return expiredAdmissionSnapshot{}, false, mapScriptError(err)
	}
	fields, err := scriptStrings(value, 4)
	if err != nil {
		return expiredAdmissionSnapshot{}, false, err
	}
	if fields[0] != "next_expired" {
		return expiredAdmissionSnapshot{}, false, fmt.Errorf("%w: unexpected expired admission state", ErrRuntimeCorrupt)
	}
	if fields[2] == "" {
		return expiredAdmissionSnapshot{}, false, nil
	}
	serverTime, err := parseMilliseconds(fields[1])
	if err != nil {
		return expiredAdmissionSnapshot{}, false, err
	}
	deadline, err := parseMilliseconds(fields[3])
	if err != nil {
		return expiredAdmissionSnapshot{}, false, err
	}
	snapshot, state, err := e.readExpiredAdmission(
		ctx, partition, fields[2], fields[3], serverTime, deadline,
	)
	return snapshot, state == expiredAdmissionPending, err
}

func (e *RedisEngine) readExpiredAdmission(
	ctx context.Context,
	partition, admissionID, deadlineText string,
	serverTime, deadline time.Time,
) (expiredAdmissionSnapshot, expiredAdmissionReadState, error) {
	if err := validateOpaque("admission ID", admissionID); err != nil {
		return expiredAdmissionSnapshot{}, expiredAdmissionMissing, fmt.Errorf("%w: invalid expired admission identity", ErrRuntimeCorrupt)
	}
	keys, _ := newPartitionKeysWithPrefix(e.keyPrefix, partition)
	scriptKeys := []string{
		keys.pendingIndex, keys.pending(admissionID), keys.dispatches(admissionID),
		keys.attempts(admissionID), keys.terminal(admissionID),
	}
	if err := validateRuntimeKeys(scriptKeys, keys.tag, e.keyPrefix); err != nil {
		return expiredAdmissionSnapshot{}, expiredAdmissionMissing, err
	}
	value, err := readExpiredScript.Run(ctx, e.client, scriptKeys,
		admissionID, deadlineText,
		strconv.Itoa(maximumRecoveryDispatches), strconv.Itoa(maximumRecoveryAttemptFields),
	).Result()
	if err != nil {
		return expiredAdmissionSnapshot{}, expiredAdmissionMissing, mapScriptError(err)
	}
	values, ok := value.([]any)
	if !ok || len(values) < 2 {
		return expiredAdmissionSnapshot{}, expiredAdmissionMissing, fmt.Errorf("%w: expired admission response shape", ErrRuntimeCorrupt)
	}
	header, err := scriptStrings(values[:2], 2)
	if err != nil {
		return expiredAdmissionSnapshot{}, expiredAdmissionMissing, err
	}
	switch header[0] {
	case "expired_admission_gone":
		return expiredAdmissionSnapshot{}, expiredAdmissionMissing, nil
	case "expired_admission_renewed":
		return expiredAdmissionSnapshot{}, expiredAdmissionRenewed, nil
	case "expired_admission_terminal":
		return expiredAdmissionSnapshot{}, expiredAdmissionTerminal, nil
	}
	if header[0] != "expired_admission" || len(values) != 6 {
		return expiredAdmissionSnapshot{}, expiredAdmissionMissing, fmt.Errorf("%w: unexpected expired admission response", ErrRuntimeCorrupt)
	}
	observedServerTime, err := parseMilliseconds(header[1])
	if err != nil {
		return expiredAdmissionSnapshot{}, expiredAdmissionMissing, err
	}
	if observedServerTime.Before(serverTime) {
		return expiredAdmissionSnapshot{}, expiredAdmissionMissing, fmt.Errorf("%w: Redis recovery time moved backwards", ErrRuntimeCorrupt)
	}
	observedDeadline, err := recoveryString(values[2])
	if err != nil || observedDeadline != deadlineText {
		return expiredAdmissionSnapshot{}, expiredAdmissionMissing, fmt.Errorf("%w: expired admission deadline differs", ErrRuntimeCorrupt)
	}
	pending, err := recoveryStringMap(values[3], "pending admission")
	if err != nil {
		return expiredAdmissionSnapshot{}, expiredAdmissionMissing, err
	}
	dispatchJournal, err := recoveryStringMap(values[4], "dispatch journal")
	if err != nil {
		return expiredAdmissionSnapshot{}, expiredAdmissionMissing, err
	}
	attemptFields, err := recoveryStringMap(values[5], "attempt evidence")
	if err != nil {
		return expiredAdmissionSnapshot{}, expiredAdmissionMissing, err
	}
	record, err := decodeAdmissionRecoveryRecord(pending["recovery_record"], pending["recovery_digest"])
	if err != nil {
		return expiredAdmissionSnapshot{}, expiredAdmissionMissing, err
	}
	if pending["digest"] == "" || pending["plan_digest"] == "" {
		return expiredAdmissionSnapshot{}, expiredAdmissionMissing, fmt.Errorf("%w: expired admission identity is incomplete", ErrRuntimeCorrupt)
	}
	if err := validateRecoveryRuleSnapshot(e.keyPrefix, partition, pending, record.Rules); err != nil {
		return expiredAdmissionSnapshot{}, expiredAdmissionMissing, err
	}
	dispatches, err := parseExpiredDispatches(dispatchJournal, attemptFields)
	if err != nil {
		return expiredAdmissionSnapshot{}, expiredAdmissionMissing, err
	}
	revision, err := parseRecoveryRevision(attemptFields["revision"])
	if err != nil {
		return expiredAdmissionSnapshot{}, expiredAdmissionMissing, err
	}
	return expiredAdmissionSnapshot{
		partition: partition, admissionID: admissionID, admissionDigest: pending["digest"],
		deadline: deadline, serverTime: observedServerTime, record: record,
		dispatches: dispatches, evidenceRevision: revision,
	}, expiredAdmissionPending, nil
}

func (e *RedisEngine) observeExpiredAdmissionRace(
	ctx context.Context,
	snapshot expiredAdmissionSnapshot,
) (expiredAdmissionReadState, error) {
	_, state, err := e.readExpiredAdmission(
		ctx, snapshot.partition, snapshot.admissionID,
		strconv.FormatInt(snapshot.deadline.UnixMilli(), 10), snapshot.serverTime, snapshot.deadline,
	)
	return state, err
}

func validateRecoveryRuleSnapshot(prefix, partition string, pending map[string]string, rules []RuleBinding) error {
	compiled, err := compileRulesWithPrefix(prefix, partition, rules)
	if err != nil {
		return fmt.Errorf("%w: compile expired admission rules: %v", ErrRuntimeCorrupt, err)
	}
	actualCount, concurrencyCount := 0, 0
	for _, rule := range compiled {
		if rule.binding.isResponseActual() {
			actualCount++
			if pending["actual:"+rule.keys.meta] != rule.fingerprint {
				return fmt.Errorf("%w: expired actual rule fingerprint differs", ErrRuntimeCorrupt)
			}
		}
		if rule.binding.isConcurrency() {
			concurrencyCount++
			if pending["concurrency:"+rule.keys.events] != rule.fingerprint {
				return fmt.Errorf("%w: expired concurrency rule fingerprint differs", ErrRuntimeCorrupt)
			}
		}
	}
	if pending["actual_rule_count"] != strconv.Itoa(actualCount) ||
		pending["concurrency_count"] != strconv.Itoa(concurrencyCount) {
		return fmt.Errorf("%w: expired recovery rule counts differ", ErrRuntimeCorrupt)
	}
	return nil
}

func parseExpiredDispatches(journal, attempts map[string]string) ([]expiredDispatch, error) {
	dispatches := make([]expiredDispatch, 0, len(journal))
	seenOrdinals := make(map[uint32]struct{}, len(journal))
	for dispatchID, encoded := range journal {
		separator := strings.IndexByte(encoded, '|')
		if separator < 1 || separator == len(encoded)-1 {
			return nil, fmt.Errorf("%w: expired dispatch journal value is invalid", ErrRuntimeCorrupt)
		}
		ordinal, err := parseUint32(encoded[:separator], "dispatch ordinal")
		if err != nil {
			return nil, err
		}
		if _, duplicate := seenOrdinals[ordinal]; duplicate {
			return nil, fmt.Errorf("%w: expired dispatch ordinals are duplicated", ErrRuntimeCorrupt)
		}
		seenOrdinals[ordinal] = struct{}{}
		item := expiredDispatch{id: dispatchID, ordinal: ordinal, planDigest: encoded[separator+1:]}
		prefix := "dispatch:" + keyComponent(dispatchID) + ":"
		storedPlan, present := attempts[prefix+"plan_digest"]
		if present {
			modelRevision, parseErr := strconv.ParseInt(attempts[prefix+"model_revision"], 10, 64)
			storedOrdinal, ordinalErr := parseUint32(attempts[prefix+"ordinal"], "attempt dispatch ordinal")
			if parseErr != nil || modelRevision <= 0 || ordinalErr != nil || storedOrdinal != ordinal ||
				storedPlan != item.planDigest || attempts[prefix+"dispatch_id"] != dispatchID ||
				attempts[prefix+"model_id"] == "" {
				return nil, fmt.Errorf("%w: expired dispatch attempt identity differs", ErrRuntimeCorrupt)
			}
			item.evidencePresent = true
			item.evidenceModelID = attempts[prefix+"model_id"]
			item.evidenceModelRev = modelRevision
		} else {
			for field := range attempts {
				if strings.HasPrefix(field, prefix) {
					return nil, fmt.Errorf("%w: partial expired dispatch attempt evidence", ErrRuntimeCorrupt)
				}
			}
		}
		dispatches = append(dispatches, item)
	}
	for field, dispatchID := range attempts {
		if !strings.HasSuffix(field, ":dispatch_id") {
			continue
		}
		if _, exists := journal[dispatchID]; !exists {
			return nil, fmt.Errorf("%w: attempt evidence has no dispatch journal", ErrRuntimeCorrupt)
		}
	}
	sort.Slice(dispatches, func(left, right int) bool {
		return dispatches[left].ordinal < dispatches[right].ordinal
	})
	return dispatches, nil
}

func parseRecoveryRevision(value string) (uint64, error) {
	if value == "" {
		return 0, nil
	}
	revision, err := strconv.ParseUint(value, 10, 64)
	if err != nil || revision > maximumEvidenceRevision {
		return 0, fmt.Errorf("%w: invalid expired attempt evidence revision", ErrRuntimeCorrupt)
	}
	return revision, nil
}

func recoveryStringMap(value any, label string) (map[string]string, error) {
	values, ok := value.([]any)
	if !ok || len(values)%2 != 0 {
		return nil, fmt.Errorf("%w: %s has an invalid response shape", ErrRuntimeCorrupt, label)
	}
	result := make(map[string]string, len(values)/2)
	for index := 0; index < len(values); index += 2 {
		pair, err := scriptStrings(values[index:index+2], 2)
		if err != nil {
			return nil, err
		}
		if _, duplicate := result[pair[0]]; duplicate {
			return nil, fmt.Errorf("%w: %s contains duplicate fields", ErrRuntimeCorrupt, label)
		}
		result[pair[0]] = pair[1]
	}
	return result, nil
}

func recoveryString(value any) (string, error) {
	switch typed := value.(type) {
	case string:
		return typed, nil
	case []byte:
		return string(typed), nil
	case int64:
		return strconv.FormatInt(typed, 10), nil
	default:
		return "", fmt.Errorf("%w: recovery script field has type %T", ErrRuntimeCorrupt, value)
	}
}

func (e *RedisEngine) buildExpiredFinalization(
	ctx context.Context,
	snapshot expiredAdmissionSnapshot,
) (FinalizationRequest, error) {
	dispatches := make([]usageledger.Dispatch, 0, len(snapshot.dispatches))
	known, unknown := 0, 0
	observedRevision := snapshot.evidenceRevision
	for index, journal := range snapshot.dispatches {
		item, revision, err := e.expiredUsageDispatch(ctx, snapshot, journal, index)
		if err != nil {
			return FinalizationRequest{}, err
		}
		if revision != observedRevision {
			return FinalizationRequest{}, ErrEvidenceChanged
		}
		dispatches = append(dispatches, item)
		if item.UsageState == usageledger.UsageUnknown {
			unknown++
		} else {
			known++
		}
	}
	if len(dispatches) == 0 {
		dispatches = append(dispatches, undispatchedRecoveryUsage(snapshot))
		known++
	}
	evidenceState := usageledger.EvidenceKnown
	if unknown > 0 && known > 0 {
		evidenceState = usageledger.EvidenceMixed
	} else if unknown > 0 {
		evidenceState = usageledger.EvidenceUnknown
	}
	actualEvidence, fenceBindings, err := expiredRecoveryEvidence(snapshot.record.Rules, unknown)
	if err != nil {
		return FinalizationRequest{}, err
	}
	completedAt := snapshot.deadline.UTC()
	for _, dispatch := range dispatches {
		if dispatch.CompletedAt.After(completedAt) {
			completedAt = dispatch.CompletedAt
		}
	}
	occurredAt := snapshot.record.Context.OccurredAt.UTC()
	if completedAt.Before(occurredAt) {
		occurredAt = completedAt
	}
	event := expiredRecoveryEvent(snapshot, evidenceState, unknown, occurredAt, completedAt, dispatches)
	fenceID := ""
	if unknown > 0 && len(fenceBindings) > 0 {
		fenceID = snapshot.record.Context.FenceID
		event.Fence = &usageledger.UnknownFence{
			FenceID: fenceID, Reason: expiredAdmissionReason, Bindings: fenceBindings,
		}
	}
	encoded, digest, err := finalizeRecoveryEvent(event)
	if err != nil {
		return FinalizationRequest{}, err
	}
	return FinalizationRequest{
		Partition: snapshot.partition, AdmissionID: snapshot.admissionID,
		AdmissionDigest: snapshot.admissionDigest, FinalizationDigest: digest,
		DispatchCount: uint32(len(snapshot.dispatches)), EvidenceRevision: observedRevision,
		ExpectedAdmissionDeadline: snapshot.deadline,
		Event:                     encoded, EventEvidenceState: evidenceState,
		FenceID: fenceID, Rules: snapshot.record.Rules, Evidence: actualEvidence,
	}, nil
}

func expiredRecoveryEvent(
	snapshot expiredAdmissionSnapshot,
	evidenceState usageledger.EvidenceState,
	unknown int,
	occurredAt time.Time,
	completedAt time.Time,
	dispatches []usageledger.Dispatch,
) usageledger.TerminalEvent {
	return usageledger.TerminalEvent{
		Schema:              usageledger.TerminalEventSchema,
		EventID:             snapshot.record.Context.EventID,
		NamespaceID:         snapshot.record.Context.NamespaceID,
		AdmissionID:         snapshot.admissionID,
		FinalizationDigest:  strings.Repeat("0", sha256.Size*2),
		EvidenceState:       evidenceState,
		ExternalRequestID:   snapshot.record.Context.ExternalRequestID,
		ReplayID:            snapshot.record.Context.ReplayID,
		Protocol:            snapshot.record.Context.Protocol,
		Path:                snapshot.record.Context.Path,
		StatusCode:          500,
		ErrorCode:           expiredAdmissionReason,
		OccurredAt:          occurredAt,
		CompletedAt:         completedAt,
		LatencyMilliseconds: completedAt.Sub(occurredAt).Milliseconds(),
		Stream:              snapshot.record.Context.Stream,
		Principal: usageledger.PrincipalSnapshot{
			APIKeyID: snapshot.record.Context.Principal.APIKeyID,
			UserID:   snapshot.record.Context.Principal.UserID,
			TeamID:   snapshot.record.Context.Principal.TeamID,
		},
		Routing: usageledger.RoutingSnapshot{
			EntrypointID:       snapshot.record.Context.Routing.EntrypointID,
			EntrypointName:     snapshot.record.Context.Routing.EntrypointName,
			EntrypointRuleID:   snapshot.record.Context.Routing.EntrypointRuleID,
			EntrypointRuleName: snapshot.record.Context.Routing.EntrypointRuleName,
			RecipeID:           snapshot.record.Context.Routing.RecipeID,
			RecipeName:         snapshot.record.Context.Routing.RecipeName,
			RecipeRevision:     snapshot.record.Context.Routing.RecipeRevision,
			RoutingRevision:    snapshot.record.Context.Routing.RoutingRevision,
			AccessRevision:     snapshot.record.Context.Routing.AccessRevision,
		},
		Served: usageledger.ServedUsage{
			InputTokens: "0", InputKnown: unknown == 0,
			OutputTokens: "0", OutputKnown: unknown == 0,
		},
		Dispatches: dispatches,
	}
}

func expiredRecoveryEvidence(
	rules []RuleBinding,
	unknown int,
) (map[quota.CounterIdentity]ActualEvidence, []usageledger.FenceBinding, error) {
	actual := make(map[quota.CounterIdentity]ActualEvidence)
	fences := make([]usageledger.FenceBinding, 0)
	for _, binding := range rules {
		if !binding.isResponseActual() {
			continue
		}
		identity, err := binding.Counter()
		if err != nil {
			return nil, nil, err
		}
		if unknown == 0 {
			actual[identity] = ActualEvidence{State: ActualEvidenceKnown, Amount: quota.QuotaInteger{}}
			continue
		}
		actual[identity] = ActualEvidence{State: ActualEvidenceUnknown, Reason: expiredAdmissionReason}
		fences = append(fences, usageledger.FenceBinding{
			BindingID: identity.BindingID, RuleID: identity.RuleID,
			AdmissionLimit: binding.limit().String(),
		})
	}
	return actual, fences, nil
}

func (e *RedisEngine) expiredUsageDispatch(
	ctx context.Context,
	snapshot expiredAdmissionSnapshot,
	dispatch expiredDispatch,
	ordinal int,
) (usageledger.Dispatch, uint64, error) {
	if !dispatch.evidencePresent {
		item := baseRecoveryDispatch(snapshot, dispatch.id, ordinal)
		item.DispatchType = "planned"
		item.Attempts = []usageledger.Attempt{knownZeroRecoveryAttempt(dispatch.id, item.StartedAt)}
		item.EvidenceDigest = recoveryDispatchDigest(item)
		return item, snapshot.evidenceRevision, nil
	}
	evidence, err := e.ReadAttemptEvidence(ctx, ReadAttemptEvidenceRequest{
		AttemptEvidenceReference: AttemptEvidenceReference{
			Partition: snapshot.partition, AdmissionID: snapshot.admissionID,
			AdmissionDigest: snapshot.admissionDigest, DispatchID: dispatch.id,
			Ordinal: dispatch.ordinal, DispatchPlanDigest: dispatch.planDigest,
			ModelID: dispatch.evidenceModelID, ModelRevision: dispatch.evidenceModelRev,
		},
	})
	if err != nil {
		return usageledger.Dispatch{}, 0, err
	}
	if !evidence.Present {
		return usageledger.Dispatch{}, 0, fmt.Errorf("%w: expired dispatch evidence disappeared", ErrRuntimeCorrupt)
	}
	item := baseRecoveryDispatch(snapshot, dispatch.id, ordinal)
	item.DispatchType = evidence.Evidence.DispatchType
	item.ModelID = evidence.Evidence.ModelID
	item.ModelRevision = evidence.Evidence.ModelRevision
	item.PricingRevision = evidence.Evidence.ModelRevision
	item.StartedAt = evidence.Evidence.StartedAt.UTC()
	item.CompletedAt = snapshot.deadline.UTC()
	item.Attempts = make([]usageledger.Attempt, 0, len(evidence.Evidence.Attempts))
	dispatchUnknown := false
	for attemptOrdinal, attempt := range evidence.Evidence.Attempts {
		state := usageledger.UsageUnknown
		errorCode := attempt.ErrorCode
		completedAt := attempt.CompletedAt.UTC()
		if attempt.State == AttemptEvidenceKnownZero && attempt.Finished {
			state = usageledger.UsageKnownZero
		} else if errorCode == "" {
			errorCode = expiredAdmissionReason
		}
		if state == usageledger.UsageUnknown {
			dispatchUnknown = true
		}
		if completedAt.IsZero() {
			completedAt = snapshot.deadline.UTC()
		}
		if completedAt.After(item.CompletedAt) {
			item.CompletedAt = completedAt
		}
		item.Attempts = append(item.Attempts, usageledger.Attempt{
			AttemptID: attempt.AttemptID, Ordinal: attemptOrdinal,
			BackendID:  canonicalRecoveryUUIDOrEmpty(attempt.BackendID),
			ProviderID: canonicalRecoveryCodeOrEmpty(attempt.ProviderID),
			State:      state, StatusCode: attempt.StatusCode, ErrorCode: canonicalRecoveryCodeOrEmpty(errorCode),
			StartedAt: attempt.StartedAt.UTC(), CompletedAt: completedAt,
		})
	}
	if len(item.Attempts) == 0 {
		item.Attempts = []usageledger.Attempt{knownZeroRecoveryAttempt(dispatch.id, item.StartedAt)}
	} else if dispatchUnknown {
		item.UsageState = usageledger.UsageUnknown
		item.UnknownReason = expiredAdmissionReason
		item.Cost.State = usageledger.CostUnknown
		item.Cost.Reason = expiredAdmissionReason
	}
	item.EvidenceDigest = recoveryDispatchDigest(item)
	return item, evidence.Revision, nil
}

func undispatchedRecoveryUsage(snapshot expiredAdmissionSnapshot) usageledger.Dispatch {
	digest := sha256.Sum256([]byte(snapshot.admissionID))
	id := "recovery-" + hex.EncodeToString(digest[:16])
	item := baseRecoveryDispatch(snapshot, id, 0)
	item.DispatchType = "not_dispatched"
	item.Attempts = []usageledger.Attempt{knownZeroRecoveryAttempt(id, item.StartedAt)}
	item.EvidenceDigest = recoveryDispatchDigest(item)
	return item
}

func baseRecoveryDispatch(snapshot expiredAdmissionSnapshot, id string, ordinal int) usageledger.Dispatch {
	fallback := snapshot.record.Context.FallbackDispatch
	return usageledger.Dispatch{
		DispatchID: id, Ordinal: ordinal, DispatchType: "recovery",
		ModelID: fallback.ModelID, ModelName: fallback.ModelName,
		ModelRevision: fallback.ModelRevision, PricingRevision: fallback.ModelRevision,
		InputTokens: "0", CacheReadTokens: "0", CacheWriteTokens: "0", OutputTokens: "0",
		UsageState: usageledger.UsageKnownZero,
		Cost: usageledger.DispatchCost{
			Currency: fallback.Currency, State: usageledger.CostComplete, Numerator: "0",
		},
		StartedAt: snapshot.deadline.UTC(), CompletedAt: snapshot.deadline.UTC(),
	}
}

func knownZeroRecoveryAttempt(dispatchID string, at time.Time) usageledger.Attempt {
	return usageledger.Attempt{
		AttemptID: dispatchID + "/not-started", Ordinal: 0,
		State: usageledger.UsageKnownZero, StartedAt: at, CompletedAt: at,
	}
}

func recoveryDispatchDigest(dispatch usageledger.Dispatch) string {
	dispatch.EvidenceDigest = ""
	payload, _ := json.Marshal(dispatch)
	digest := sha256.Sum256(payload)
	return hex.EncodeToString(digest[:])
}

func finalizeRecoveryEvent(event usageledger.TerminalEvent) (string, string, error) {
	if _, err := event.Validate(); err != nil {
		return "", "", fmt.Errorf("validate expired admission usage event: %w", err)
	}
	payload, err := json.Marshal(event)
	if err != nil {
		return "", "", fmt.Errorf("encode expired admission usage event: %w", err)
	}
	digest := sha256.Sum256(payload)
	digestText := hex.EncodeToString(digest[:])
	event.FinalizationDigest = digestText
	if _, err := event.Validate(); err != nil {
		return "", "", fmt.Errorf("validate finalized expired admission usage event: %w", err)
	}
	payload, err = json.Marshal(event)
	if err != nil {
		return "", "", fmt.Errorf("encode finalized expired admission usage event: %w", err)
	}
	return string(payload), digestText, nil
}

func canonicalRecoveryUUIDOrEmpty(value string) string {
	parsed, err := uuid.Parse(value)
	if err != nil || parsed.String() != value {
		return ""
	}
	return value
}

func canonicalRecoveryCodeOrEmpty(value string) string {
	if value == "" || !recoveryCodePattern.MatchString(value) {
		return ""
	}
	return value
}
