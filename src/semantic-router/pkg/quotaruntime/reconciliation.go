package quotaruntime

import (
	"context"
	"fmt"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/quota"
)

// CounterCorrection is one immutable counter delta derived from the original
// admission and its authoritative reconciliation evidence.
type CounterCorrection struct {
	BindingID              string            `json:"bindingId"`
	RuleID                 string            `json:"ruleId"`
	Metric                 quota.Metric      `json:"metric"`
	Algorithm              quota.Algorithm   `json:"algorithm"`
	Enforcement            quota.Enforcement `json:"enforcement"`
	Amount                 string            `json:"amount"`
	CounterIncompleteCount string            `json:"counterIncompleteCount"`
	ChargeAt               time.Time         `json:"chargeAt"`
	Window                 time.Duration     `json:"window,omitempty"`
	CalendarStart          time.Time         `json:"calendarStart,omitempty"`
	CalendarEnd            time.Time         `json:"calendarEnd,omitempty"`
	Charge                 bool              `json:"charge"`
	Known                  bool              `json:"known"`
}

type ReconciliationRequest struct {
	Partition        string
	FenceID          string
	AdmissionID      string
	ReconciliationID string
	PlanDigest       string
	Event            string
	Corrections      []CounterCorrection
}

type ReconciliationResult struct {
	MutationResult
	StreamID string
}

// FenceCounter identifies one enforced response-actual counter protected by
// an unknown-usage fence. Fence release validates the counter and its binding
// fence set in the same Redis transaction, so the last fence can never be
// removed while incomplete usage remains.
type FenceCounter struct {
	BindingID string
	RuleID    string
	Metric    quota.Metric
}

type FenceRemovalRequest struct {
	Partition        string
	FenceID          string
	ReconciliationID string
	PlanDigest       string
	Counters         []FenceCounter
}

func (e *RedisEngine) ApplyReconciliation(
	ctx context.Context,
	request ReconciliationRequest,
) (ReconciliationResult, error) {
	if err := validatePartition(request.Partition); err != nil {
		return ReconciliationResult{}, err
	}
	for label, value := range map[string]string{
		"fence ID": request.FenceID, "admission ID": request.AdmissionID,
		"reconciliation ID": request.ReconciliationID,
	} {
		if err := validateOpaque(label, value); err != nil {
			return ReconciliationResult{}, err
		}
	}
	if err := validateDigest("reconciliation plan digest", request.PlanDigest); err != nil {
		return ReconciliationResult{}, err
	}
	if request.Event == "" || len(request.Event) > maxUsageEventBytes || strings.ContainsRune(request.Event, '\x00') ||
		len(request.Corrections) == 0 || len(request.Corrections) > 4096 {
		return ReconciliationResult{}, fmt.Errorf("%w: invalid reconciliation event or correction set", ErrInvalidRequest)
	}
	corrections := append([]CounterCorrection(nil), request.Corrections...)
	sort.Slice(corrections, func(i, j int) bool {
		if corrections[i].BindingID != corrections[j].BindingID {
			return corrections[i].BindingID < corrections[j].BindingID
		}
		return corrections[i].RuleID < corrections[j].RuleID
	})
	partition, _ := newPartitionKeysWithPrefix(e.keyPrefix, request.Partition)
	keys := []string{partition.fence(request.FenceID), partition.reconciliationStream}
	args := []any{
		request.FenceID, request.ReconciliationID, request.PlanDigest,
		request.AdmissionID, request.Event, strconv.Itoa(len(corrections)),
	}
	seen := make(map[quota.CounterIdentity]struct{}, len(corrections))
	for index, correction := range corrections {
		identity, kind, err := validateCounterCorrection(correction)
		if err != nil {
			return ReconciliationResult{}, fmt.Errorf("correction %d: %w", index, err)
		}
		if _, duplicate := seen[identity]; duplicate {
			return ReconciliationResult{}, fmt.Errorf("%w: duplicate reconciliation counter", ErrInvalidRequest)
		}
		seen[identity] = struct{}{}
		counter := partition.counter(identity, kind)
		keys = append(keys, counter.meta, counter.events, counter.values, counter.fences)
		calendarStart, calendarEnd := "", ""
		if correction.Algorithm == quota.AlgorithmCalendarWindow {
			calendarStart = strconv.FormatInt(correction.CalendarStart.UnixMilli(), 10)
			calendarEnd = strconv.FormatInt(correction.CalendarEnd.UnixMilli(), 10)
		}
		charge := "0"
		if correction.Charge {
			charge = "1"
		}
		known := "0"
		if correction.Known {
			known = "1"
		}
		args = append(args, correction.Amount, correction.CounterIncompleteCount,
			string(correction.Algorithm), strconv.FormatInt(correction.Window.Milliseconds(), 10),
			strconv.FormatInt(correction.ChargeAt.UnixMilli(), 10), calendarStart, calendarEnd,
			"reconciliation:"+request.ReconciliationID+":"+strconv.Itoa(index), charge, known,
			string(correction.Enforcement))
	}
	if err := validateRuntimeKeys(keys, partition.tag, e.keyPrefix); err != nil {
		return ReconciliationResult{}, err
	}
	value, err := reconcileUnknownScript.Run(ctx, e.client, keys, args...).Result()
	if err != nil {
		return ReconciliationResult{}, mapScriptError(err)
	}
	fields, err := scriptStrings(value, 4)
	if err != nil || fields[0] != "corrected" || fields[2] == "" {
		return ReconciliationResult{}, fmt.Errorf("%w: invalid reconciliation result", ErrRuntimeCorrupt)
	}
	serverTime, err := parseMilliseconds(fields[3])
	if err != nil {
		return ReconciliationResult{}, err
	}
	return ReconciliationResult{MutationResult: MutationResult{
		Idempotent: fields[1] == "1", ServerTime: serverTime,
	}, StreamID: fields[2]}, nil
}

func validateCounterCorrection(value CounterCorrection) (quota.CounterIdentity, quota.CounterKind, error) {
	identity, err := quota.NewCounterIdentity(value.BindingID, value.RuleID)
	if err != nil {
		return quota.CounterIdentity{}, "", fmt.Errorf("%w: invalid counter identity", ErrInvalidRequest)
	}
	kind, err := value.Metric.CounterKind()
	if err != nil || value.Metric == quota.MetricRequests || value.Metric == quota.MetricConcurrentRequests {
		return quota.CounterIdentity{}, "", fmt.Errorf("%w: unsupported reconciliation metric", ErrInvalidRequest)
	}
	if value.Algorithm != quota.AlgorithmSlidingLog && value.Algorithm != quota.AlgorithmCalendarWindow {
		return quota.CounterIdentity{}, "", fmt.Errorf("%w: unsupported reconciliation algorithm", ErrInvalidRequest)
	}
	if value.Enforcement != quota.EnforcementEnforce && value.Enforcement != quota.EnforcementShadow {
		return quota.CounterIdentity{}, "", fmt.Errorf("%w: invalid reconciliation enforcement", ErrInvalidRequest)
	}
	amount, amountErr := quota.ParseQuotaInteger(value.Amount)
	incomplete, incompleteErr := quota.ParseQuotaInteger(value.CounterIncompleteCount)
	if amountErr != nil || incompleteErr != nil || incomplete.IsZero() || value.ChargeAt.IsZero() {
		return quota.CounterIdentity{}, "", fmt.Errorf("%w: invalid reconciliation quantity", ErrInvalidRequest)
	}
	if !value.Charge && !amount.IsZero() {
		return quota.CounterIdentity{}, "", fmt.Errorf("%w: waived reconciliation cannot carry an amount", ErrInvalidRequest)
	}
	if value.Known && !value.Charge {
		return quota.CounterIdentity{}, "", fmt.Errorf("%w: known reconciliation must carry a charge", ErrInvalidRequest)
	}
	if value.Algorithm == quota.AlgorithmSlidingLog {
		if value.Window <= 0 || value.Window%time.Second != 0 || !value.CalendarStart.IsZero() || !value.CalendarEnd.IsZero() {
			return quota.CounterIdentity{}, "", fmt.Errorf("%w: invalid sliding reconciliation window", ErrInvalidRequest)
		}
	} else if value.Window != 0 || value.CalendarStart.IsZero() || value.CalendarEnd.IsZero() ||
		!value.CalendarStart.Before(value.CalendarEnd) || value.ChargeAt.Before(value.CalendarStart) || !value.ChargeAt.Before(value.CalendarEnd) {
		return quota.CounterIdentity{}, "", fmt.Errorf("%w: invalid calendar reconciliation interval", ErrInvalidRequest)
	}
	return identity, kind, nil
}

func (e *RedisEngine) RemoveReconciledFence(
	ctx context.Context,
	request FenceRemovalRequest,
) (MutationResult, error) {
	if err := validatePartition(request.Partition); err != nil {
		return MutationResult{}, err
	}
	for label, value := range map[string]string{
		"fence ID": request.FenceID, "reconciliation ID": request.ReconciliationID,
	} {
		if err := validateOpaque(label, value); err != nil {
			return MutationResult{}, err
		}
	}
	if err := validateDigest("reconciliation plan digest", request.PlanDigest); err != nil {
		return MutationResult{}, err
	}
	counters := append([]FenceCounter(nil), request.Counters...)
	if len(counters) > 4096 {
		return MutationResult{}, fmt.Errorf("%w: too many fenced counters", ErrInvalidRequest)
	}
	sort.Slice(counters, func(left, right int) bool {
		if counters[left].BindingID != counters[right].BindingID {
			return counters[left].BindingID < counters[right].BindingID
		}
		return counters[left].RuleID < counters[right].RuleID
	})
	partition, _ := newPartitionKeysWithPrefix(e.keyPrefix, request.Partition)
	keys := []string{partition.fence(request.FenceID)}
	seen := make(map[quota.CounterIdentity]struct{}, len(counters))
	for index, counter := range counters {
		identity, err := quota.NewCounterIdentity(counter.BindingID, counter.RuleID)
		if err != nil {
			return MutationResult{}, fmt.Errorf(
				"%w: invalid fenced counter %d",
				ErrInvalidRequest,
				index,
			)
		}
		if _, duplicate := seen[identity]; duplicate {
			return MutationResult{}, fmt.Errorf("%w: duplicate fenced counter", ErrInvalidRequest)
		}
		seen[identity] = struct{}{}
		kind, err := counter.Metric.CounterKind()
		if err != nil || counter.Metric == quota.MetricRequests ||
			counter.Metric == quota.MetricConcurrentRequests {
			return MutationResult{}, fmt.Errorf(
				"%w: fenced counter %d is not response-actual",
				ErrInvalidRequest,
				index,
			)
		}
		counterKeys := partition.counter(identity, kind)
		keys = append(keys, counterKeys.fences, counterKeys.meta)
	}
	if err := validateRuntimeKeys(keys, partition.tag, e.keyPrefix); err != nil {
		return MutationResult{}, err
	}
	value, err := removeReconciledFenceScript.Run(ctx, e.client, keys,
		request.FenceID, request.ReconciliationID, request.PlanDigest,
		strconv.FormatInt(e.finalizationMarkerTTL.Milliseconds(), 10),
		strconv.Itoa(len(counters))).Result()
	if err != nil {
		return MutationResult{}, mapScriptError(err)
	}
	return parseMutationResult(value, "released")
}
