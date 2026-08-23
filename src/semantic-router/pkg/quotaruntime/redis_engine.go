package quotaruntime

import (
	"context"
	"fmt"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/redis/go-redis/v9"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/quota"
)

const (
	defaultFinalizationMarkerTTL = 24 * time.Hour
	defaultMaxUsageBacklog       = int64(1_000_000)
	maximumMaxUsageBacklog       = int64(1_000_000_000)
)

type RedisEngineOptions struct {
	FinalizationMarkerTTL time.Duration
	// MaxUsageBacklog is the largest already-settled usage stream backlog at
	// which a new admission may start. The check runs in the same partition
	// script as access and quota admission, so every replica observes one
	// global boundary. Terminal finalization is deliberately never rejected:
	// requests admitted below the boundary must remain accountably settleable.
	MaxUsageBacklog int64
	KeyPrefix       string
}

// RedisEngine executes one server-side script per operation. It intentionally
// keeps no authorization or quota-result cache in process memory.
type RedisEngine struct {
	client                redis.Scripter
	finalizationMarkerTTL time.Duration
	maxUsageBacklog       int64
	keyPrefix             string
}

var _ Engine = (*RedisEngine)(nil)

func NewRedisEngine(client redis.Scripter, options RedisEngineOptions) (*RedisEngine, error) {
	if client == nil {
		return nil, fmt.Errorf("%w: Redis client is required", ErrInvalidRequest)
	}
	if err := validateKeyPrefix(options.KeyPrefix); err != nil {
		return nil, err
	}
	markerTTL := options.FinalizationMarkerTTL
	if markerTTL == 0 {
		markerTTL = defaultFinalizationMarkerTTL
	}
	if markerTTL <= 0 || markerTTL%time.Millisecond != 0 {
		return nil, fmt.Errorf(
			"%w: finalization marker TTL must be a positive whole number of milliseconds",
			ErrInvalidRequest,
		)
	}
	maxUsageBacklog := options.MaxUsageBacklog
	if maxUsageBacklog == 0 {
		maxUsageBacklog = defaultMaxUsageBacklog
	}
	if maxUsageBacklog < 1 || maxUsageBacklog > maximumMaxUsageBacklog {
		return nil, fmt.Errorf(
			"%w: maximum usage backlog must be between 1 and %d",
			ErrInvalidRequest,
			maximumMaxUsageBacklog,
		)
	}
	return &RedisEngine{
		client:                client,
		finalizationMarkerTTL: markerTTL,
		maxUsageBacklog:       maxUsageBacklog,
		keyPrefix:             options.KeyPrefix,
	}, nil
}

func (e *RedisEngine) CheckAccess(
	ctx context.Context,
	request AccessCheckRequest,
) (AccessCheckResult, error) {
	if err := validatePartition(request.Partition); err != nil {
		return AccessCheckResult{}, err
	}
	if len(request.Preconditions) == 0 {
		return AccessCheckResult{}, fmt.Errorf(
			"%w: access check requires atomic preconditions",
			ErrInvalidRequest,
		)
	}
	preconditions := append([]AdmissionPrecondition(nil), request.Preconditions...)
	for index, precondition := range preconditions {
		if err := precondition.Validate(); err != nil {
			return AccessCheckResult{}, fmt.Errorf("precondition %d: %w", index, err)
		}
	}
	sortAdmissionPreconditions(preconditions)
	partition, _ := newPartitionKeysWithPrefix(e.keyPrefix, request.Partition)
	keys := make([]string, 0, len(preconditions))
	args := []any{strconv.Itoa(len(preconditions))}
	for _, precondition := range preconditions {
		keys = append(keys, precondition.Key)
		args = append(args,
			string(precondition.Kind),
			precondition.Field,
			precondition.Expected,
			string(precondition.Failure),
			precondition.Reason,
		)
	}
	if err := validateRuntimeKeys(keys, partition.tag, e.keyPrefix); err != nil {
		return AccessCheckResult{}, err
	}
	value, err := checkAccessScript.Run(ctx, e.client, keys, args...).Result()
	if err != nil {
		return AccessCheckResult{
			Disposition: AdmissionUnavailable,
			Reason:      "access_runtime_error",
		}, mapScriptError(err)
	}
	fields, err := scriptStrings(value, 3)
	if err != nil {
		return AccessCheckResult{}, err
	}
	serverTime, err := parseMilliseconds(fields[1])
	if err != nil {
		return AccessCheckResult{}, err
	}
	result := AccessCheckResult{
		Disposition: AdmissionDisposition(fields[0]),
		ServerTime:  serverTime,
		Reason:      fields[2],
	}
	switch result.Disposition {
	case AdmissionAllowed,
		AdmissionUnauthenticated,
		AdmissionForbidden,
		AdmissionUnavailable:
		return result, nil
	default:
		return AccessCheckResult{}, fmt.Errorf(
			"%w: invalid access disposition %q",
			ErrRuntimeCorrupt,
			result.Disposition,
		)
	}
}

func (e *RedisEngine) Admit(ctx context.Context, request AdmissionRequest) (AdmissionResult, error) {
	rules, admitErr := validateAdmissionRequestWithPrefix(e.keyPrefix, request)
	if admitErr != nil {
		return AdmissionResult{}, admitErr
	}
	partition, _ := newPartitionKeysWithPrefix(e.keyPrefix, request.Partition)
	preconditions := append([]AdmissionPrecondition(nil), request.Preconditions...)
	sortAdmissionPreconditions(preconditions)
	planFingerprint := admissionPlanFingerprint(
		request.LeaseDuration.Milliseconds(),
		preconditions,
		rules,
	)
	keys := []string{
		partition.pendingIndex,
		partition.pending(request.AdmissionID),
		partition.dispatches(request.AdmissionID),
		partition.terminal(request.AdmissionID),
	}
	args := []any{
		request.AdmissionID,
		request.Digest,
		strconv.FormatInt(request.LeaseDuration.Milliseconds(), 10),
		planFingerprint,
		strconv.Itoa(len(preconditions)),
		strconv.Itoa(len(rules)),
	}
	for _, precondition := range preconditions {
		keys = append(keys, precondition.Key)
		args = append(args,
			string(precondition.Kind),
			precondition.Field,
			precondition.Expected,
			string(precondition.Failure),
			precondition.Reason,
		)
	}
	for _, rule := range rules {
		keys = append(keys, rule.keys.meta, rule.keys.events, rule.keys.values, rule.keys.fences)
		args = append(args,
			rule.identity.BindingID,
			rule.identity.RuleID,
			string(rule.binding.Rule.Metric),
			string(rule.binding.Rule.Algorithm),
			string(rule.binding.Rule.Accounting),
			string(rule.binding.Rule.Enforcement),
			rule.limit.String(),
			strconv.FormatInt(rule.windowMS, 10),
			rule.calendarSchedule,
			rule.bucketCapacity,
			rule.refillAmount,
			strconv.FormatInt(rule.refillPeriodMS, 10),
			strconv.FormatInt(rule.gcraEmissionUS, 10),
			rule.gcraBurst,
			rule.fingerprint,
		)
	}
	// Keep the stream key last so variable access precondition and rule key
	// offsets stay stable. The Lua admission contract treats this as a pure
	// backpressure observation before it mutates any counter or pending record.
	keys = append(keys, partition.usageStream)
	args = append(args, strconv.FormatInt(e.maxUsageBacklog, 10))
	if err := validateRuntimeKeys(keys, partition.tag, e.keyPrefix); err != nil {
		return AdmissionResult{}, err
	}

	value, admitErr := admitScript.Run(ctx, e.client, keys, args...).Result()
	if admitErr != nil {
		return AdmissionResult{
			Disposition:    AdmissionUnavailable,
			BlockingReason: "quota_runtime_error",
		}, mapScriptError(admitErr)
	}
	fields, admitErr := scriptStrings(value, 7)
	if admitErr != nil {
		return AdmissionResult{}, admitErr
	}
	result := AdmissionResult{
		Disposition:    AdmissionDisposition(fields[0]),
		Idempotent:     fields[1] == "1",
		BlockingReason: fields[6],
	}
	result.ServerTime, admitErr = parseMilliseconds(fields[3])
	if admitErr != nil {
		return AdmissionResult{}, admitErr
	}
	if fields[4] != "" {
		result.Deadline, admitErr = parseMilliseconds(fields[4])
		if admitErr != nil {
			return AdmissionResult{}, admitErr
		}
	}
	if fields[5] != "" {
		retryAt, parseErr := parseMilliseconds(fields[5])
		if parseErr != nil {
			return AdmissionResult{}, parseErr
		}
		result.RetryAt = &retryAt
		result.ResetAt = &retryAt
	}
	if fields[2] != "0" {
		index, parseErr := strconv.Atoi(fields[2])
		if parseErr != nil || index < 1 || index > len(rules) {
			return AdmissionResult{}, fmt.Errorf("%w: invalid limiting rule index %q", ErrRuntimeCorrupt, fields[2])
		}
		identity := rules[index-1].identity
		result.Limiting = &identity
	}
	switch result.Disposition {
	case AdmissionAllowed,
		AdmissionUnauthenticated,
		AdmissionForbidden,
		AdmissionRateLimited,
		AdmissionUnavailable:
		return result, nil
	default:
		return AdmissionResult{}, fmt.Errorf("%w: invalid admission disposition %q", ErrRuntimeCorrupt, result.Disposition)
	}
}

func sortAdmissionPreconditions(preconditions []AdmissionPrecondition) {
	sort.Slice(preconditions, func(left, right int) bool {
		leftValue := preconditionSortValue(preconditions[left])
		rightValue := preconditionSortValue(preconditions[right])
		return leftValue < rightValue
	})
}

func preconditionSortValue(precondition AdmissionPrecondition) string {
	return strings.Join([]string{
		precondition.Key,
		string(precondition.Kind),
		precondition.Field,
		precondition.Expected,
		string(precondition.Failure),
		precondition.Reason,
	}, "\x00")
}

func (e *RedisEngine) JournalDispatch(
	ctx context.Context,
	request DispatchJournalRequest,
) (MutationResult, error) {
	if err := validateEnvelope(request.Partition, request.AdmissionID, request.AdmissionDigest); err != nil {
		return MutationResult{}, err
	}
	if err := validateOpaque("dispatch ID", request.DispatchID); err != nil {
		return MutationResult{}, err
	}
	if err := validateDigest("dispatch digest", request.Digest); err != nil {
		return MutationResult{}, err
	}
	partition, _ := newPartitionKeysWithPrefix(e.keyPrefix, request.Partition)
	keys := []string{
		partition.pending(request.AdmissionID),
		partition.terminal(request.AdmissionID),
		partition.dispatches(request.AdmissionID),
	}
	if err := validateRuntimeKeys(keys, partition.tag, e.keyPrefix); err != nil {
		return MutationResult{}, err
	}
	value, err := journalDispatchScript.Run(ctx, e.client, keys,
		request.AdmissionID,
		request.AdmissionDigest,
		request.DispatchID,
		strconv.FormatUint(uint64(request.Ordinal), 10),
		request.Digest,
	).Result()
	if err != nil {
		return MutationResult{}, mapScriptError(err)
	}
	return parseMutationResult(value, "journaled")
}

func (e *RedisEngine) ReleaseConcurrency(
	ctx context.Context,
	request ConcurrencyReleaseRequest,
) (MutationResult, error) {
	if err := validateEnvelope(request.Partition, request.AdmissionID, request.AdmissionDigest); err != nil {
		return MutationResult{}, err
	}
	rules, releaseConcurrencyErr := compileRulesWithPrefix(e.keyPrefix, request.Partition, request.Rules)
	if releaseConcurrencyErr != nil {
		return MutationResult{}, releaseConcurrencyErr
	}
	concurrency := filterRules(rules, func(rule compiledRule) bool { return rule.binding.isConcurrency() })
	partition, _ := newPartitionKeysWithPrefix(e.keyPrefix, request.Partition)
	keys := []string{partition.pending(request.AdmissionID), partition.terminal(request.AdmissionID)}
	args := []any{request.AdmissionID, request.AdmissionDigest, strconv.Itoa(len(concurrency))}
	for _, rule := range concurrency {
		keys = append(keys, rule.keys.events)
		args = append(args, rule.fingerprint)
	}
	if err := validateRuntimeKeys(keys, partition.tag, e.keyPrefix); err != nil {
		return MutationResult{}, err
	}
	value, releaseConcurrencyErr := releaseConcurrencyScript.Run(ctx, e.client, keys, args...).Result()
	if releaseConcurrencyErr != nil {
		return MutationResult{}, mapScriptError(releaseConcurrencyErr)
	}
	return parseMutationResult(value, "released")
}

func (e *RedisEngine) ReadMeters(ctx context.Context, request MeterReadRequest) (MeterReadResult, error) {
	if err := validatePartition(request.Partition); err != nil {
		return MeterReadResult{}, err
	}
	rules, readMetersErr := compileRulesWithPrefix(e.keyPrefix, request.Partition, request.Rules)
	if readMetersErr != nil {
		return MeterReadResult{}, readMetersErr
	}
	partition, _ := newPartitionKeysWithPrefix(e.keyPrefix, request.Partition)
	keys := []string{partition.pendingIndex}
	args := []any{strconv.Itoa(len(rules))}
	for _, rule := range rules {
		keys = append(keys, rule.keys.meta, rule.keys.events, rule.keys.values, rule.keys.fences)
		args = append(args,
			string(rule.binding.Rule.Metric),
			string(rule.binding.Rule.Algorithm),
			string(rule.binding.Rule.Accounting),
			string(rule.binding.Rule.Enforcement),
			rule.limit.String(),
			strconv.FormatInt(rule.windowMS, 10),
			rule.calendarSchedule,
			rule.bucketCapacity,
			rule.refillAmount,
			strconv.FormatInt(rule.refillPeriodMS, 10),
			strconv.FormatInt(rule.gcraEmissionUS, 10),
			rule.gcraBurst,
		)
	}
	if err := validateRuntimeKeys(keys, partition.tag, e.keyPrefix); err != nil {
		return MeterReadResult{}, err
	}
	value, readMetersErr := readMetersScript.Run(ctx, e.client, keys, args...).Result()
	if readMetersErr != nil {
		return MeterReadResult{}, mapScriptError(readMetersErr)
	}
	fields, readMetersErr := scriptStrings(value, 2+len(rules)*6)
	if readMetersErr != nil {
		return MeterReadResult{}, readMetersErr
	}
	asOf, readMetersErr := parseMilliseconds(fields[0])
	if readMetersErr != nil {
		return MeterReadResult{}, readMetersErr
	}
	count, readMetersErr := strconv.Atoi(fields[1])
	if readMetersErr != nil || count != len(rules) {
		return MeterReadResult{}, fmt.Errorf("%w: meter count %q differs", ErrRuntimeCorrupt, fields[1])
	}
	result := MeterReadResult{Meters: make([]Meter, 0, len(rules)), AsOf: asOf}
	for index, rule := range rules {
		offset := 2 + index*6
		used, parseErr := quota.ParseQuotaInteger(fields[offset])
		if parseErr != nil {
			return MeterReadResult{}, fmt.Errorf("%w: invalid used quantity: %w", ErrRuntimeCorrupt, parseErr)
		}
		known, parseErr := quota.ParseQuotaInteger(fields[offset+1])
		if parseErr != nil {
			return MeterReadResult{}, fmt.Errorf("%w: invalid known dispatch quantity: %w", ErrRuntimeCorrupt, parseErr)
		}
		incomplete, parseErr := quota.ParseQuotaInteger(fields[offset+2])
		if parseErr != nil {
			return MeterReadResult{}, fmt.Errorf("%w: invalid incomplete dispatch quantity: %w", ErrRuntimeCorrupt, parseErr)
		}
		public, parseErr := quota.NewPublicMeter(quota.MeterSnapshot{
			Counter:              rule.identity,
			Metric:               rule.binding.Rule.Metric,
			Enforcement:          rule.binding.Rule.Enforcement,
			Limit:                rule.limit,
			Used:                 used,
			Currency:             rule.binding.Currency,
			KnownDispatches:      known,
			IncompleteDispatches: incomplete,
			FenceOpen:            fields[offset+4] == "1",
		})
		if parseErr != nil {
			return MeterReadResult{}, fmt.Errorf("%w: invalid public meter: %w", ErrRuntimeCorrupt, parseErr)
		}
		meter := Meter{
			PublicMeter: public,
			Algorithm:   rule.binding.Rule.Algorithm,
			Accounting:  rule.binding.Rule.Accounting,
		}
		if fields[offset+5] != "" {
			meter.ActiveFenceIDs = strings.Split(fields[offset+5], "\x00")
			for _, fenceID := range meter.ActiveFenceIDs {
				if err := validateOpaque("active fence ID", fenceID); err != nil {
					return MeterReadResult{}, fmt.Errorf("%w: invalid active fence identity", ErrRuntimeCorrupt)
				}
			}
		}
		if fields[offset+3] != "" {
			resetAt, resetErr := parseMilliseconds(fields[offset+3])
			if resetErr != nil {
				return MeterReadResult{}, resetErr
			}
			meter.ResetAt = &resetAt
		}
		result.Meters = append(result.Meters, meter)
	}
	return result, nil
}

func filterRules(rules []compiledRule, predicate func(compiledRule) bool) []compiledRule {
	filtered := make([]compiledRule, 0, len(rules))
	for _, rule := range rules {
		if predicate(rule) {
			filtered = append(filtered, rule)
		}
	}
	return filtered
}

func validateSingleHashTag(keys []string, expected string) error {
	for _, key := range keys {
		start := strings.IndexByte(key, '{')
		finish := strings.IndexByte(key, '}')
		if start < 0 || finish <= start || key[start:finish+1] != expected {
			return fmt.Errorf("%w: key %q is outside partition %s", ErrInvalidRequest, key, expected)
		}
	}
	return nil
}

func validateRuntimeKeys(keys []string, expectedTag, prefix string) error {
	if err := validateSingleHashTag(keys, expectedTag); err != nil {
		return err
	}
	if prefix == "" {
		return nil
	}
	wirePrefix := prefix + ":"
	for _, key := range keys {
		if !strings.HasPrefix(key, wirePrefix) {
			return fmt.Errorf(
				"%w: key %q is outside configured key prefix",
				ErrInvalidRequest,
				key,
			)
		}
	}
	return nil
}

func parseMutationResult(value any, expected string) (MutationResult, error) {
	fields, err := scriptStrings(value, 3)
	if err != nil {
		return MutationResult{}, err
	}
	if fields[0] != expected {
		return MutationResult{}, fmt.Errorf("%w: unexpected mutation state %q", ErrRuntimeCorrupt, fields[0])
	}
	serverTime, err := parseMilliseconds(fields[2])
	if err != nil {
		return MutationResult{}, err
	}
	return MutationResult{Idempotent: fields[1] == "1", ServerTime: serverTime}, nil
}

func scriptStrings(value any, expected int) ([]string, error) {
	values, ok := value.([]any)
	if !ok || len(values) != expected {
		return nil, fmt.Errorf("%w: script returned %T with unexpected length", ErrRuntimeCorrupt, value)
	}
	result := make([]string, len(values))
	for index, item := range values {
		switch typed := item.(type) {
		case string:
			result[index] = typed
		case []byte:
			result[index] = string(typed)
		case int64:
			result[index] = strconv.FormatInt(typed, 10)
		default:
			return nil, fmt.Errorf("%w: script field %d has type %T", ErrRuntimeCorrupt, index, item)
		}
	}
	return result, nil
}

func parseMilliseconds(value string) (time.Time, error) {
	milliseconds, err := strconv.ParseInt(value, 10, 64)
	if err != nil || milliseconds < 0 {
		return time.Time{}, fmt.Errorf("%w: invalid Redis millisecond time %q", ErrRuntimeCorrupt, value)
	}
	return time.UnixMilli(milliseconds).UTC(), nil
}

func mapScriptError(err error) error {
	message := err.Error()
	switch {
	case strings.Contains(message, "QUOTA_EVIDENCE_CHANGED"):
		return fmt.Errorf("%w: %s", ErrEvidenceChanged, message)
	case strings.Contains(message, "QUOTA_CONFLICT"):
		return fmt.Errorf("%w: %s", ErrConflict, message)
	case strings.Contains(message, "QUOTA_NOT_FOUND"):
		return fmt.Errorf("%w: %s", ErrAdmissionNotFound, message)
	case strings.Contains(message, "QUOTA_INVALID"):
		return fmt.Errorf("%w: %s", ErrInvalidRequest, message)
	case strings.Contains(message, "QUOTA_CORRUPT"):
		return fmt.Errorf("%w: %s", ErrRuntimeCorrupt, message)
	default:
		return fmt.Errorf("quota runtime store operation: %w", err)
	}
}
