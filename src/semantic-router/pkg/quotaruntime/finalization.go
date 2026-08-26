package quotaruntime

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"strconv"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/quota"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/usageledger"
)

const maxUsageEventBytes = 1 << 20

func (e *RedisEngine) Finalize(
	ctx context.Context,
	request FinalizationRequest,
) (FinalizationResult, error) {
	actual, concurrency, err := e.validateFinalizationRequest(request)
	if err != nil {
		return FinalizationResult{}, err
	}
	keys, args, err := e.buildFinalizationScriptInput(request, actual, concurrency)
	if err != nil {
		return FinalizationResult{}, err
	}
	value, finalizeErr := finalizeScript.Run(ctx, e.client, keys, args...).Result()
	if finalizeErr != nil {
		return FinalizationResult{}, mapScriptError(finalizeErr)
	}
	fields, finalizeErr := scriptStrings(value, 5)
	if finalizeErr != nil {
		return FinalizationResult{}, finalizeErr
	}
	if fields[0] != "finalized" {
		return FinalizationResult{}, fmt.Errorf(
			"%w: unexpected finalization state %q",
			ErrRuntimeCorrupt,
			fields[0],
		)
	}
	serverTime, finalizeErr := parseMilliseconds(fields[2])
	if finalizeErr != nil {
		return FinalizationResult{}, finalizeErr
	}
	return FinalizationResult{
		MutationResult: MutationResult{Idempotent: fields[1] == "1", ServerTime: serverTime},
		EvidenceState:  fields[3],
		StreamID:       fields[4],
	}, nil
}

func (e *RedisEngine) validateFinalizationRequest(
	request FinalizationRequest,
) ([]compiledRule, []compiledRule, error) {
	if err := validateEnvelope(request.Partition, request.AdmissionID, request.AdmissionDigest); err != nil {
		return nil, nil, err
	}
	if err := validateDigest("finalization digest", request.FinalizationDigest); err != nil {
		return nil, nil, err
	}
	if request.EvidenceRevision > maximumEvidenceRevision {
		return nil, nil, fmt.Errorf("%w: attempt evidence revision is outside the supported range", ErrInvalidRequest)
	}
	if !request.ExpectedAdmissionDeadline.IsZero() &&
		(request.ExpectedAdmissionDeadline.UnixMilli() <= 0 ||
			request.ExpectedAdmissionDeadline.Nanosecond()%int(time.Millisecond) != 0) {
		return nil, nil, fmt.Errorf(
			"%w: expected admission deadline must be a positive millisecond-aligned instant",
			ErrInvalidRequest,
		)
	}
	if request.Event == "" || len(request.Event) > maxUsageEventBytes || strings.ContainsRune(request.Event, '\x00') {
		return nil, nil, fmt.Errorf(
			"%w: usage event must be non-empty, NUL-free, and at most %d bytes",
			ErrInvalidRequest,
			maxUsageEventBytes,
		)
	}
	switch request.EventEvidenceState {
	case usageledger.EvidenceKnown, usageledger.EvidenceMixed, usageledger.EvidenceUnknown:
	default:
		return nil, nil, fmt.Errorf(
			"%w: invalid terminal event evidence state %q",
			ErrInvalidRequest,
			request.EventEvidenceState,
		)
	}
	rules, finalizeErr := compileRulesWithPrefix(e.keyPrefix, request.Partition, request.Rules)
	if finalizeErr != nil {
		return nil, nil, finalizeErr
	}
	actual := filterRules(rules, func(rule compiledRule) bool { return rule.binding.isResponseActual() })
	concurrency := filterRules(rules, func(rule compiledRule) bool { return rule.binding.isConcurrency() })
	if len(request.Evidence) != len(actual) {
		return nil, nil, fmt.Errorf(
			"%w: evidence must classify every response_actual rule",
			ErrInvalidRequest,
		)
	}

	unknownCount := 0
	for _, rule := range actual {
		evidence, exists := request.Evidence[rule.identity]
		if !exists {
			return nil, nil, fmt.Errorf(
				"%w: missing evidence for %s",
				ErrInvalidRequest,
				rule.identity.String(),
			)
		}
		switch evidence.State {
		case ActualEvidenceKnown:
			if evidence.Reason != "" {
				return nil, nil, fmt.Errorf(
					"%w: known evidence cannot carry an unknown reason",
					ErrInvalidRequest,
				)
			}
		case ActualEvidenceUnknown:
			unknownCount++
			if !evidence.Amount.IsZero() {
				return nil, nil, fmt.Errorf(
					"%w: unknown evidence cannot carry an amount",
					ErrInvalidRequest,
				)
			}
			if err := validateOpaque("unknown evidence reason", evidence.Reason); err != nil {
				return nil, nil, err
			}
			if len(evidence.Reason) > 128 {
				return nil, nil, fmt.Errorf(
					"%w: unknown evidence reason is too long",
					ErrInvalidRequest,
				)
			}
		default:
			return nil, nil, fmt.Errorf(
				"%w: invalid actual evidence state %q",
				ErrInvalidRequest,
				evidence.State,
			)
		}
	}
	for identity := range request.Evidence {
		if err := identity.Validate(); err != nil {
			return nil, nil, fmt.Errorf("%w: evidence counter: %w", ErrInvalidRequest, err)
		}
	}
	if unknownCount > 0 {
		if err := validateOpaque("fence ID", request.FenceID); err != nil {
			return nil, nil, err
		}
	} else if request.FenceID != "" {
		return nil, nil, fmt.Errorf(
			"%w: fence ID is valid only when some usage is unknown",
			ErrInvalidRequest,
		)
	}
	return actual, concurrency, nil
}

func (e *RedisEngine) buildFinalizationScriptInput(
	request FinalizationRequest,
	actual, concurrency []compiledRule,
) ([]string, []any, error) {
	partition, _ := newPartitionKeysWithPrefix(e.keyPrefix, request.Partition)
	fenceMarkerKey := partition.terminal(request.AdmissionID)
	if request.FenceID != "" {
		fenceMarkerKey = partition.fence(request.FenceID)
	}
	keys := []string{
		partition.pendingIndex,
		partition.pending(request.AdmissionID),
		partition.dispatches(request.AdmissionID),
		partition.terminal(request.AdmissionID),
		partition.usageStream,
		fenceMarkerKey,
	}

	fenceKeys := unknownEnforceFenceKeys(actual, request.Evidence)
	fenceOrdinals := make(map[string]int, len(fenceKeys))
	for index, key := range fenceKeys {
		fenceOrdinals[key] = index + 1
	}
	planDigest := finalizationPlanFingerprint(request, actual)
	args := []any{
		request.AdmissionID,
		request.AdmissionDigest,
		request.FinalizationDigest,
		planDigest,
		request.Event,
		strconv.FormatInt(e.finalizationMarkerTTL.Milliseconds(), 10),
		request.FenceID,
		strconv.FormatUint(uint64(request.DispatchCount), 10),
		strconv.FormatUint(request.EvidenceRevision, 10),
		string(request.EventEvidenceState),
		finalizationExpectedDeadline(request.ExpectedAdmissionDeadline),
		strconv.Itoa(len(actual)),
	}
	for _, rule := range actual {
		evidence := request.Evidence[rule.identity]
		fenceOrdinal := 0
		if evidence.State == ActualEvidenceUnknown &&
			rule.binding.Rule.Enforcement == quota.EnforcementEnforce {
			fenceOrdinal = fenceOrdinals[rule.keys.fences]
		}
		keys = append(keys, rule.keys.meta, rule.keys.events, rule.keys.values)
		args = append(args,
			rule.fingerprint,
			string(evidence.State),
			evidence.Amount.String(),
			evidence.Reason,
			string(rule.binding.Rule.Algorithm),
			strconv.FormatInt(rule.windowMS, 10),
			rule.calendarSchedule,
			strconv.Itoa(fenceOrdinal),
		)
	}
	args = append(args, strconv.Itoa(len(fenceKeys)))
	keys = append(keys, fenceKeys...)
	args = append(args, strconv.Itoa(len(concurrency)))
	for _, rule := range concurrency {
		keys = append(keys, rule.keys.events)
		args = append(args, rule.fingerprint)
	}
	// Attempt evidence is request-scoped recovery state. Keep it last so the
	// variable rule/fence offsets above remain stable, then let Finalize delete
	// it in the same atomic operation as counter settlement and XADD.
	keys = append(keys, partition.attempts(request.AdmissionID))
	if err := validateRuntimeKeys(keys, partition.tag, e.keyPrefix); err != nil {
		return nil, nil, err
	}
	return keys, args, nil
}

func unknownEnforceFenceKeys(
	rules []compiledRule,
	evidence map[quota.CounterIdentity]ActualEvidence,
) []string {
	seen := make(map[string]struct{})
	keys := make([]string, 0, len(rules))
	for _, rule := range rules {
		if evidence[rule.identity].State != ActualEvidenceUnknown ||
			rule.binding.Rule.Enforcement != quota.EnforcementEnforce {
			continue
		}
		if _, exists := seen[rule.keys.fences]; exists {
			continue
		}
		seen[rule.keys.fences] = struct{}{}
		keys = append(keys, rule.keys.fences)
	}
	return keys
}

func finalizationPlanFingerprint(request FinalizationRequest, rules []compiledRule) string {
	fields := []string{
		request.FenceID,
		request.Event,
		string(request.EventEvidenceState),
		strconv.FormatUint(uint64(request.DispatchCount), 10),
		strconv.FormatUint(request.EvidenceRevision, 10),
		finalizationExpectedDeadline(request.ExpectedAdmissionDeadline),
	}
	for _, rule := range rules {
		evidence := request.Evidence[rule.identity]
		fields = append(fields,
			rule.fingerprint,
			string(evidence.State),
			evidence.Amount.String(),
			evidence.Reason,
		)
	}
	digest := sha256.Sum256([]byte(strings.Join(fields, "\x00")))
	return hex.EncodeToString(digest[:])
}

func finalizationExpectedDeadline(value time.Time) string {
	if value.IsZero() {
		return ""
	}
	return strconv.FormatInt(value.UnixMilli(), 10)
}
