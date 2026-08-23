package quotaruntime

import (
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"fmt"
	"regexp"
	"sort"
	"strconv"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/quota"
)

var (
	partitionPattern = regexp.MustCompile(`^[A-Za-z0-9._-]+$`)
	keyPrefixPattern = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._-]*(:[A-Za-z0-9][A-Za-z0-9._-]*)*$`)
)

type partitionKeys struct {
	tag                  string
	prefix               string
	pendingIndex         string
	usageStream          string
	reconciliationStream string
}

func newPartitionKeys(partition string) (partitionKeys, error) {
	return newPartitionKeysWithPrefix("", partition)
}

func newPartitionKeysWithPrefix(prefix, partition string) (partitionKeys, error) {
	if err := validateKeyPrefix(prefix); err != nil {
		return partitionKeys{}, err
	}
	if err := validatePartition(partition); err != nil {
		return partitionKeys{}, err
	}
	tag := "{" + partition + "}"
	return partitionKeys{
		tag:                  tag,
		prefix:               prefix,
		pendingIndex:         prefixedKey(prefix, "pending:"+tag),
		usageStream:          prefixedKey(prefix, "usage-stream:"+tag),
		reconciliationStream: prefixedKey(prefix, "quota-reconciliation-stream:"+tag),
	}, nil
}

func validateKeyPrefix(prefix string) error {
	if prefix == "" {
		return nil
	}
	if len(prefix) > 128 || !keyPrefixPattern.MatchString(prefix) {
		return fmt.Errorf(
			"%w: key prefix must be canonical colon-separated identifier segments",
			ErrInvalidRequest,
		)
	}
	return nil
}

func prefixedKey(prefix, key string) string {
	if prefix == "" {
		return key
	}
	return prefix + ":" + key
}

func validatePartition(partition string) error {
	if !partitionPattern.MatchString(partition) {
		return fmt.Errorf(
			"%w: partition must contain only letters, digits, dot, underscore, or hyphen",
			ErrInvalidRequest,
		)
	}
	return nil
}

func (p partitionKeys) pending(admissionID string) string {
	return prefixedKey(p.prefix, "pending:"+p.tag+":"+keyComponent(admissionID))
}

func (p partitionKeys) dispatches(admissionID string) string {
	return p.pending(admissionID) + ":dispatches"
}

func (p partitionKeys) attempts(admissionID string) string {
	return p.pending(admissionID) + ":attempts"
}

func (p partitionKeys) terminal(admissionID string) string {
	return prefixedKey(p.prefix, "settled:"+p.tag+":"+keyComponent(admissionID))
}

func (p partitionKeys) fence(fenceID string) string {
	return prefixedKey(p.prefix, "quota:"+p.tag+":unknown-fence:"+keyComponent(fenceID))
}

func (p partitionKeys) bindingFences(bindingID string) string {
	return prefixedKey(p.prefix, "quota:"+p.tag+":unknown-by-binding:"+keyComponent(bindingID))
}

type counterKeys struct {
	meta   string
	events string
	values string
	fences string
}

func (p partitionKeys) counter(identity quota.CounterIdentity, kind quota.CounterKind) counterKeys {
	base := prefixedKey(p.prefix, "quota:"+p.tag+":"+keyComponent(identity.BindingID)+":"+
		keyComponent(identity.RuleID)+":"+string(kind),
	)
	return counterKeys{
		meta:   base,
		events: base + ":events",
		values: base + ":values",
		fences: p.bindingFences(identity.BindingID),
	}
}

func keyComponent(value string) string {
	return base64.RawURLEncoding.EncodeToString([]byte(value))
}

type compiledRule struct {
	binding          RuleBinding
	identity         quota.CounterIdentity
	kind             quota.CounterKind
	keys             counterKeys
	limit            quota.QuotaInteger
	windowMS         int64
	calendarSchedule string
	bucketCapacity   string
	refillAmount     string
	refillPeriodMS   int64
	gcraEmissionUS   int64
	gcraBurst        string
	fingerprint      string
}

func compileRules(partition string, bindings []RuleBinding) ([]compiledRule, error) {
	return compileRulesWithPrefix("", partition, bindings)
}

func compileRulesWithPrefix(prefix, partition string, bindings []RuleBinding) ([]compiledRule, error) {
	partitionKeySet, err := newPartitionKeysWithPrefix(prefix, partition)
	if err != nil {
		return nil, err
	}
	rules := make([]compiledRule, 0, len(bindings))
	seen := make(map[quota.CounterIdentity]struct{}, len(bindings))
	for index, binding := range bindings {
		if err := binding.Validate(); err != nil {
			return nil, fmt.Errorf("rule %d: %w", index, err)
		}
		identity, err := binding.Counter()
		if err != nil {
			return nil, fmt.Errorf("rule %d counter: %w", index, err)
		}
		if _, exists := seen[identity]; exists {
			return nil, fmt.Errorf("%w: duplicate counter %s", ErrInvalidRequest, identity.String())
		}
		seen[identity] = struct{}{}
		kind, err := binding.Rule.Metric.CounterKind()
		if err != nil {
			return nil, fmt.Errorf("rule %d counter kind: %w", index, err)
		}
		compiled := compiledRule{
			binding:  binding,
			identity: identity,
			kind:     kind,
			keys:     partitionKeySet.counter(identity, kind),
			limit:    binding.limit(),
		}
		if binding.Rule.Algorithm == quota.AlgorithmSlidingLog {
			compiled.windowMS = binding.Rule.Window.Milliseconds()
		}
		if binding.Rule.Algorithm == quota.AlgorithmCalendarWindow {
			compiled.calendarSchedule = encodeCalendarSchedule(binding.CalendarSchedule)
		}
		if binding.Rule.Algorithm == quota.AlgorithmTokenBucket {
			compiled.bucketCapacity = binding.Rule.BucketCapacity.String()
			compiled.refillAmount = binding.Rule.RefillAmount.String()
			compiled.refillPeriodMS = binding.Rule.RefillPeriod.Milliseconds()
		}
		if binding.Rule.Algorithm == quota.AlgorithmGCRA {
			compiled.gcraEmissionUS = binding.Rule.GCRAEmissionInterval.Microseconds()
			compiled.gcraBurst = binding.Rule.GCRABurstTolerance.String()
		}
		compiled.fingerprint = ruleFingerprint(compiled)
		rules = append(rules, compiled)
	}
	sortCompiledRules(rules)
	return rules, nil
}

func encodeCalendarSchedule(schedule []CalendarInterval) string {
	intervals := make([]string, 0, len(schedule))
	for _, interval := range schedule {
		intervals = append(intervals, fmt.Sprintf(
			"%d:%d",
			interval.Start.UnixMilli(),
			interval.End.UnixMilli(),
		))
	}
	return strings.Join(intervals, ",")
}

func sortCompiledRules(rules []compiledRule) {
	sort.Slice(rules, func(left, right int) bool {
		if rules[left].binding.Rule.Ordinal != rules[right].binding.Rule.Ordinal {
			return rules[left].binding.Rule.Ordinal < rules[right].binding.Rule.Ordinal
		}
		return rules[left].identity.String() < rules[right].identity.String()
	})
}

func ruleFingerprint(rule compiledRule) string {
	fields := []string{
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
		string(rule.binding.Rule.CalendarPeriod),
		rule.binding.Rule.CalendarTimezone,
		rule.binding.Currency,
		rule.keys.meta,
		rule.keys.events,
		rule.keys.values,
	}
	digest := sha256.Sum256([]byte(strings.Join(fields, "\x00")))
	return hex.EncodeToString(digest[:])
}

func admissionPlanFingerprint(
	leaseMilliseconds int64,
	preconditions []AdmissionPrecondition,
	rules []compiledRule,
) string {
	fields := []string{strconv.FormatInt(leaseMilliseconds, 10)}
	for _, precondition := range preconditions {
		fields = append(fields,
			precondition.Key,
			string(precondition.Kind),
			precondition.Field,
			precondition.Expected,
			string(precondition.Failure),
			precondition.Reason,
		)
	}
	for _, rule := range rules {
		fields = append(fields, rule.fingerprint)
	}
	digest := sha256.Sum256([]byte(strings.Join(fields, "\x00")))
	return hex.EncodeToString(digest[:])
}
