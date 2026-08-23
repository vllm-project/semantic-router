package quotaruntime

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"strconv"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/quota"
)

const maxUsageEventBytes = 1 << 20

func (e *RedisEngine) Finalize(
	ctx context.Context,
	request FinalizationRequest,
) (FinalizationResult, error) {
	if err := validateEnvelope(request.Partition, request.AdmissionID, request.AdmissionDigest); err != nil {
		return FinalizationResult{}, err
	}
	if err := validateDigest("finalization digest", request.FinalizationDigest); err != nil {
		return FinalizationResult{}, err
	}
	if request.DispatchCount == 0 {
		return FinalizationResult{}, fmt.Errorf("%w: finalization requires at least one dispatch", ErrInvalidRequest)
	}
	if request.EvidenceRevision > maximumEvidenceRevision {
		return FinalizationResult{}, fmt.Errorf("%w: attempt evidence revision is outside the supported range", ErrInvalidRequest)
	}
	if request.Event == "" || len(request.Event) > maxUsageEventBytes || strings.ContainsRune(request.Event, '\x00') {
		return FinalizationResult{}, fmt.Errorf(
			"%w: usage event must be non-empty, NUL-free, and at most %d bytes",
			ErrInvalidRequest,
			maxUsageEventBytes,
		)
	}
	rules, finalizeErr := compileRulesWithPrefix(e.keyPrefix, request.Partition, request.Rules)
	if finalizeErr != nil {
		return FinalizationResult{}, finalizeErr
	}
	actual := filterRules(rules, func(rule compiledRule) bool { return rule.binding.isResponseActual() })
	concurrency := filterRules(rules, func(rule compiledRule) bool { return rule.binding.isConcurrency() })
	if len(request.Evidence) != len(actual) {
		return FinalizationResult{}, fmt.Errorf(
			"%w: evidence must classify every response_actual rule",
			ErrInvalidRequest,
		)
	}

	unknownCount := 0
	for _, rule := range actual {
		evidence, exists := request.Evidence[rule.identity]
		if !exists {
			return FinalizationResult{}, fmt.Errorf(
				"%w: missing evidence for %s",
				ErrInvalidRequest,
				rule.identity.String(),
			)
		}
		switch evidence.State {
		case ActualEvidenceKnown:
			if evidence.Reason != "" {
				return FinalizationResult{}, fmt.Errorf(
					"%w: known evidence cannot carry an unknown reason",
					ErrInvalidRequest,
				)
			}
		case ActualEvidenceUnknown:
			unknownCount++
			if !evidence.Amount.IsZero() {
				return FinalizationResult{}, fmt.Errorf(
					"%w: unknown evidence cannot carry an amount",
					ErrInvalidRequest,
				)
			}
			if err := validateOpaque("unknown evidence reason", evidence.Reason); err != nil {
				return FinalizationResult{}, err
			}
			if len(evidence.Reason) > 128 {
				return FinalizationResult{}, fmt.Errorf(
					"%w: unknown evidence reason is too long",
					ErrInvalidRequest,
				)
			}
		default:
			return FinalizationResult{}, fmt.Errorf(
				"%w: invalid actual evidence state %q",
				ErrInvalidRequest,
				evidence.State,
			)
		}
	}
	for identity := range request.Evidence {
		if err := identity.Validate(); err != nil {
			return FinalizationResult{}, fmt.Errorf("%w: evidence counter: %w", ErrInvalidRequest, err)
		}
	}
	if unknownCount > 0 {
		if err := validateOpaque("fence ID", request.FenceID); err != nil {
			return FinalizationResult{}, err
		}
	} else if request.FenceID != "" {
		return FinalizationResult{}, fmt.Errorf(
			"%w: fence ID is valid only when some usage is unknown",
			ErrInvalidRequest,
		)
	}

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
		strconv.FormatUint(uint64(request.DispatchCount), 10),
		strconv.FormatUint(request.EvidenceRevision, 10),
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
