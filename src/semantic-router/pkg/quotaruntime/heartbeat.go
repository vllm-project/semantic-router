package quotaruntime

import (
	"context"
	"fmt"
	"strconv"
	"time"
)

// Heartbeat atomically renews one pending admission and every concurrency
// lease it owns. The immutable admission digest and plan digest prevent a
// different process from retargeting or widening the admitted work.
func (e *RedisEngine) Heartbeat(
	ctx context.Context,
	request AdmissionHeartbeatRequest,
) (AdmissionHeartbeatResult, error) {
	if err := validateEnvelope(request.Partition, request.AdmissionID, request.AdmissionDigest); err != nil {
		return AdmissionHeartbeatResult{}, err
	}
	if err := validateDigest("admission plan digest", request.PlanDigest); err != nil {
		return AdmissionHeartbeatResult{}, err
	}
	if request.LeaseDuration <= 0 || request.LeaseDuration%time.Millisecond != 0 {
		return AdmissionHeartbeatResult{}, fmt.Errorf(
			"%w: heartbeat lease duration must be a positive whole number of milliseconds",
			ErrInvalidRequest,
		)
	}
	rules, err := compileRulesWithPrefix(e.keyPrefix, request.Partition, request.Rules)
	if err != nil {
		return AdmissionHeartbeatResult{}, err
	}
	concurrency := filterRules(rules, func(rule compiledRule) bool { return rule.binding.isConcurrency() })
	partition, _ := newPartitionKeysWithPrefix(e.keyPrefix, request.Partition)
	keys := []string{
		partition.pendingIndex,
		partition.pending(request.AdmissionID),
		partition.terminal(request.AdmissionID),
	}
	args := []any{
		request.AdmissionID,
		request.AdmissionDigest,
		request.PlanDigest,
		strconv.FormatInt(request.LeaseDuration.Milliseconds(), 10),
		strconv.Itoa(len(concurrency)),
	}
	for _, rule := range concurrency {
		keys = append(keys, rule.keys.events)
		args = append(args, rule.fingerprint)
	}
	if keyErr := validateRuntimeKeys(keys, partition.tag, e.keyPrefix); keyErr != nil {
		return AdmissionHeartbeatResult{}, keyErr
	}

	value, err := heartbeatScript.Run(ctx, e.client, keys, args...).Result()
	if err != nil {
		return AdmissionHeartbeatResult{}, mapScriptError(err)
	}
	fields, err := scriptStrings(value, 3)
	if err != nil {
		return AdmissionHeartbeatResult{}, err
	}
	result := AdmissionHeartbeatResult{Stopped: fields[0] == "stopped"}
	if fields[0] != "renewed" && fields[0] != "stopped" {
		return AdmissionHeartbeatResult{}, fmt.Errorf(
			"%w: unexpected admission heartbeat state %q",
			ErrRuntimeCorrupt,
			fields[0],
		)
	}
	result.ServerTime, err = parseMilliseconds(fields[1])
	if err != nil {
		return AdmissionHeartbeatResult{}, err
	}
	if fields[2] != "" {
		result.Deadline, err = parseMilliseconds(fields[2])
		if err != nil {
			return AdmissionHeartbeatResult{}, err
		}
	}
	if !result.Stopped && !result.Deadline.After(result.ServerTime) {
		return AdmissionHeartbeatResult{}, fmt.Errorf(
			"%w: renewed admission deadline is not in the future",
			ErrRuntimeCorrupt,
		)
	}
	return result, nil
}
