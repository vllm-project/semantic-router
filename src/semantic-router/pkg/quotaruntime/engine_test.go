package quotaruntime

import (
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/quota"
)

func TestRuleBindingValidation(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		binding RuleBinding
		wantErr error
	}{
		{name: "request sliding", binding: requestRule(t, "binding", "rpm", "2", time.Minute, 0)},
		{name: "actual tokens", binding: tokenRule(t, "binding", "tpm", "100", time.Minute, 0)},
		{name: "cost", binding: costRule(t, "binding", "cost", "5.25", time.Minute, 0)},
		{name: "concurrency", binding: concurrencyRule(t, "binding", "concurrency", "2", 0)},
		{
			name: "calendar actual",
			binding: func() RuleBinding {
				binding := tokenRule(t, "binding", "daily", "100", time.Minute, 0)
				binding.Rule.Algorithm = quota.AlgorithmCalendarWindow
				binding.Rule.Window = 0
				binding.Rule.CalendarPeriod = quota.CalendarPeriodDay
				binding.Rule.CalendarTimezone = "UTC"
				binding.CalendarSchedule = calendarScheduleAroundNow()
				return binding
			}(),
		},
		{name: "token bucket", binding: tokenBucketRule(t, "binding", "burst", "10", "2", time.Second, 0)},
		{name: "GCRA", binding: gcraRule(t, "binding", "gcra", 100*time.Millisecond, "0", 0)},
		{
			name: "cost currency required",
			binding: func() RuleBinding {
				binding := costRule(t, "binding", "cost", "5.25", time.Minute, 0)
				binding.Currency = ""
				return binding
			}(),
			wantErr: ErrInvalidRequest,
		},
		{
			name: "currency rejected on tokens",
			binding: func() RuleBinding {
				binding := tokenRule(t, "binding", "tpm", "100", time.Minute, 0)
				binding.Currency = "USD"
				return binding
			}(),
			wantErr: ErrInvalidRequest,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			err := test.binding.Validate()
			if !errors.Is(err, test.wantErr) {
				t.Fatalf("Validate() error = %v, want %v", err, test.wantErr)
			}
		})
	}
}

func TestCalendarScheduleValidation(t *testing.T) {
	t.Parallel()

	binding := calendarRequestRule(t, "binding", "daily", "10", calendarScheduleAroundNow(), 0)
	gap := binding
	gap.CalendarSchedule = append([]CalendarInterval(nil), binding.CalendarSchedule...)
	gap.CalendarSchedule[1].Start = gap.CalendarSchedule[1].Start.Add(time.Millisecond)
	if err := gap.Validate(); !errors.Is(err, ErrInvalidRequest) {
		t.Fatalf("gapped calendar Validate() error = %v, want %v", err, ErrInvalidRequest)
	}
	nonCalendar := requestRule(t, "binding", "rpm", "10", time.Minute, 0)
	nonCalendar.CalendarSchedule = calendarScheduleAroundNow()
	if err := nonCalendar.Validate(); !errors.Is(err, ErrInvalidRequest) {
		t.Fatalf("non-calendar schedule Validate() error = %v, want %v", err, ErrInvalidRequest)
	}
}

func TestCompileRulesRejectsDuplicateCounterAndUsesOrdinalOrder(t *testing.T) {
	t.Parallel()

	first := requestRule(t, "binding-b", "rpm-b", "2", time.Minute, 2)
	second := requestRule(t, "binding-a", "rpm-a", "2", time.Minute, 1)
	rules, err := compileRules("partition", []RuleBinding{first, second})
	if err != nil {
		t.Fatalf("compileRules() error = %v", err)
	}
	if rules[0].binding.Rule.ID != "rpm-a" || rules[1].binding.Rule.ID != "rpm-b" {
		t.Fatalf("compiled order = [%s, %s], want ordinal order", rules[0].binding.Rule.ID, rules[1].binding.Rule.ID)
	}
	if _, err := compileRules("partition", []RuleBinding{first, first}); !errors.Is(err, ErrInvalidRequest) {
		t.Fatalf("duplicate error = %v, want %v", err, ErrInvalidRequest)
	}
}

func TestValidateAdmissionRequest(t *testing.T) {
	t.Parallel()

	valid := AdmissionRequest{
		Partition: "namespace-1", AdmissionID: "admission-1", Digest: "digest-1",
		LeaseDuration: time.Minute, Preconditions: testAdmissionPreconditions("namespace-1"),
	}
	tests := []struct {
		name   string
		mutate func(*AdmissionRequest)
	}{
		{name: "invalid partition", mutate: func(request *AdmissionRequest) { request.Partition = "bad{slot}" }},
		{name: "missing admission", mutate: func(request *AdmissionRequest) { request.AdmissionID = "" }},
		{name: "missing digest", mutate: func(request *AdmissionRequest) { request.Digest = "" }},
		{name: "zero lease", mutate: func(request *AdmissionRequest) { request.LeaseDuration = 0 }},
		{name: "submillisecond lease", mutate: func(request *AdmissionRequest) { request.LeaseDuration = time.Microsecond }},
		{name: "missing preconditions", mutate: func(request *AdmissionRequest) { request.Preconditions = nil }},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			request := valid
			test.mutate(&request)
			if _, err := validateAdmissionRequest(request); !errors.Is(err, ErrInvalidRequest) {
				t.Fatalf("validateAdmissionRequest() error = %v, want %v", err, ErrInvalidRequest)
			}
		})
	}
}

func TestAttemptEvidenceRequestValidation(t *testing.T) {
	t.Parallel()

	reference := DispatchReference{
		Partition: "partition-1", AdmissionID: "admission-1",
		AdmissionDigest: strings.Repeat("b", 64), DispatchID: "dispatch-1",
		DispatchPlanDigest: strings.Repeat("a", 64),
		ModelID:            "model-1", ModelRevision: 2, RequestDigest: strings.Repeat("A", 43),
	}
	valid := BeginDispatchRequest{
		DispatchReference: reference, DispatchType: "primary", Ordinal: 1,
		Deadline: time.Now().UTC().Add(time.Minute).Truncate(time.Millisecond), MaxAttempts: 3,
	}
	if err := validateBeginDispatchRequest(valid); err != nil {
		t.Fatalf("validateBeginDispatchRequest() error = %v", err)
	}
	invalid := valid
	invalid.DispatchPlanDigest = "not-a-digest"
	if err := validateBeginDispatchRequest(invalid); !errors.Is(err, ErrInvalidRequest) {
		t.Fatalf("invalid plan digest error = %v, want %v", err, ErrInvalidRequest)
	}
	invalid = valid
	invalid.MaxAttempts = maximumDispatchAttempts + 1
	if err := validateBeginDispatchRequest(invalid); !errors.Is(err, ErrInvalidRequest) {
		t.Fatalf("invalid max attempts error = %v, want %v", err, ErrInvalidRequest)
	}

	finish := FinishAttemptRequest{
		DispatchReference: reference, AttemptID: "attempt-1", AttemptNumber: 1,
		BackendID: "backend-1", ProviderID: "provider-1",
		State: AttemptEvidenceResponseStarted, StatusCode: 200,
	}
	if err := validateFinishAttemptRequest(finish); err != nil {
		t.Fatalf("validateFinishAttemptRequest() error = %v", err)
	}
	finish.State = AttemptEvidenceKnownZero
	finish.StatusCode = 0
	finish.ErrorCode = "transport_error"
	if err := validateFinishAttemptRequest(finish); err != nil {
		t.Fatalf("known-zero validateFinishAttemptRequest() error = %v", err)
	}
	finish.ErrorCode = ""
	if err := validateFinishAttemptRequest(finish); !errors.Is(err, ErrInvalidRequest) {
		t.Fatalf("known-zero without error error = %v, want %v", err, ErrInvalidRequest)
	}
}

func TestAdmissionPreconditionValidation(t *testing.T) {
	t.Parallel()

	keyspace, err := NewAccessProjectionKeyspace("namespace-1")
	if err != nil {
		t.Fatalf("NewAccessProjectionKeyspace() error = %v", err)
	}
	valid := AdmissionPrecondition{
		Key: keyspace.Active("key-1"), Kind: AdmissionCheckHashEqual,
		Field: "revision", Expected: "7", Failure: AdmissionUnavailable,
		Reason: "active_policy_changed",
	}
	tests := []struct {
		name   string
		mutate func(*AdmissionPrecondition)
	}{
		{name: "invalid failure", mutate: func(value *AdmissionPrecondition) { value.Failure = AdmissionAllowed }},
		{name: "missing reason", mutate: func(value *AdmissionPrecondition) { value.Reason = "" }},
		{name: "missing hash field", mutate: func(value *AdmissionPrecondition) { value.Field = "" }},
		{name: "missing expected", mutate: func(value *AdmissionPrecondition) { value.Expected = "" }},
		{name: "unknown kind", mutate: func(value *AdmissionPrecondition) { value.Kind = "unknown" }},
		{name: "absent accepts no field", mutate: func(value *AdmissionPrecondition) {
			value.Kind = AdmissionCheckKeyAbsent
			value.Expected = ""
		}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			value := valid
			test.mutate(&value)
			if err := value.Validate(); !errors.Is(err, ErrInvalidRequest) {
				t.Fatalf("Validate() error = %v, want %v", err, ErrInvalidRequest)
			}
		})
	}
}

func requestRule(
	t *testing.T,
	bindingID string,
	ruleID string,
	limit string,
	window time.Duration,
	ordinal int,
) RuleBinding {
	t.Helper()
	whole := quotaInteger(t, limit)
	return RuleBinding{
		BindingID: bindingID,
		Rule: quota.RateLimitRule{
			ID: ruleID, Metric: quota.MetricRequests, Algorithm: quota.AlgorithmSlidingLog,
			Accounting: quota.AccountingRequest, Enforcement: quota.EnforcementEnforce,
			WholeLimit: &whole, Window: window, Ordinal: ordinal,
		},
	}
}

func tokenRule(
	t *testing.T,
	bindingID string,
	ruleID string,
	limit string,
	window time.Duration,
	ordinal int,
) RuleBinding {
	t.Helper()
	whole := quotaInteger(t, limit)
	return RuleBinding{
		BindingID: bindingID,
		Rule: quota.RateLimitRule{
			ID: ruleID, Metric: quota.MetricTotalTokens, Algorithm: quota.AlgorithmSlidingLog,
			Accounting: quota.AccountingResponseActual, Enforcement: quota.EnforcementEnforce,
			WholeLimit: &whole, Window: window, Ordinal: ordinal,
		},
	}
}

func costRule(
	t *testing.T,
	bindingID string,
	ruleID string,
	limit string,
	window time.Duration,
	ordinal int,
) RuleBinding {
	t.Helper()
	cost := currencyDecimal(t, limit)
	return RuleBinding{
		BindingID: bindingID,
		Currency:  "USD",
		Rule: quota.RateLimitRule{
			ID: ruleID, Metric: quota.MetricCost, Algorithm: quota.AlgorithmSlidingLog,
			Accounting: quota.AccountingResponseActual, Enforcement: quota.EnforcementEnforce,
			CostLimit: &cost, Window: window, Ordinal: ordinal,
		},
	}
}

func concurrencyRule(t *testing.T, bindingID, ruleID, limit string, ordinal int) RuleBinding {
	t.Helper()
	whole := quotaInteger(t, limit)
	return RuleBinding{
		BindingID: bindingID,
		Rule: quota.RateLimitRule{
			ID: ruleID, Metric: quota.MetricConcurrentRequests, Algorithm: quota.AlgorithmConcurrency,
			Accounting: quota.AccountingRequest, Enforcement: quota.EnforcementEnforce,
			WholeLimit: &whole, Ordinal: ordinal,
		},
	}
}

func calendarRequestRule(
	t *testing.T,
	bindingID string,
	ruleID string,
	limit string,
	schedule []CalendarInterval,
	ordinal int,
) RuleBinding {
	t.Helper()
	whole := quotaInteger(t, limit)
	return RuleBinding{
		BindingID: bindingID,
		Rule: quota.RateLimitRule{
			ID: ruleID, Metric: quota.MetricRequests, Algorithm: quota.AlgorithmCalendarWindow,
			Accounting: quota.AccountingRequest, Enforcement: quota.EnforcementEnforce,
			WholeLimit: &whole, CalendarPeriod: quota.CalendarPeriodDay,
			CalendarTimezone: "UTC", Ordinal: ordinal,
		},
		CalendarSchedule: schedule,
	}
}

func calendarTokenRule(
	t *testing.T,
	bindingID string,
	ruleID string,
	limit string,
	schedule []CalendarInterval,
	ordinal int,
) RuleBinding {
	t.Helper()
	whole := quotaInteger(t, limit)
	return RuleBinding{
		BindingID: bindingID,
		Rule: quota.RateLimitRule{
			ID: ruleID, Metric: quota.MetricTotalTokens, Algorithm: quota.AlgorithmCalendarWindow,
			Accounting: quota.AccountingResponseActual, Enforcement: quota.EnforcementEnforce,
			WholeLimit: &whole, CalendarPeriod: quota.CalendarPeriodDay,
			CalendarTimezone: "UTC", Ordinal: ordinal,
		},
		CalendarSchedule: schedule,
	}
}

func tokenBucketRule(
	t *testing.T,
	bindingID string,
	ruleID string,
	capacity string,
	refill string,
	period time.Duration,
	ordinal int,
) RuleBinding {
	t.Helper()
	bucketCapacity := quotaInteger(t, capacity)
	refillAmount := quotaInteger(t, refill)
	return RuleBinding{
		BindingID: bindingID,
		Rule: quota.RateLimitRule{
			ID: ruleID, Metric: quota.MetricRequests, Algorithm: quota.AlgorithmTokenBucket,
			Accounting: quota.AccountingRequest, Enforcement: quota.EnforcementEnforce,
			BucketCapacity: &bucketCapacity, RefillAmount: &refillAmount,
			RefillPeriod: period, Ordinal: ordinal,
		},
	}
}

func gcraRule(
	t *testing.T,
	bindingID string,
	ruleID string,
	emission time.Duration,
	burstMicroseconds string,
	ordinal int,
) RuleBinding {
	t.Helper()
	burst := quotaInteger(t, burstMicroseconds)
	return RuleBinding{
		BindingID: bindingID,
		Rule: quota.RateLimitRule{
			ID: ruleID, Metric: quota.MetricRequests, Algorithm: quota.AlgorithmGCRA,
			Accounting: quota.AccountingRequest, Enforcement: quota.EnforcementEnforce,
			GCRAEmissionInterval: emission, GCRABurstTolerance: &burst, Ordinal: ordinal,
		},
	}
}

func calendarScheduleAroundNow() []CalendarInterval {
	now := time.Now().UTC().Truncate(time.Millisecond)
	start := now.Add(-time.Hour)
	middle := now.Add(time.Hour)
	return []CalendarInterval{
		{Start: start, End: middle},
		{Start: middle, End: middle.Add(2 * time.Hour)},
	}
}

func quotaInteger(t *testing.T, value string) quota.QuotaInteger {
	t.Helper()
	parsed, err := quota.ParseQuotaInteger(value)
	if err != nil {
		t.Fatalf("ParseQuotaInteger(%q) error = %v", value, err)
	}
	return parsed
}

func currencyDecimal(t *testing.T, value string) quota.CurrencyDecimal {
	t.Helper()
	parsed, err := quota.ParseCurrencyDecimal(value)
	if err != nil {
		t.Fatalf("ParseCurrencyDecimal(%q) error = %v", value, err)
	}
	return parsed
}

func testAdmissionPreconditions(partition string) []AdmissionPrecondition {
	keyspace, err := NewAccessProjectionKeyspace(partition)
	if err != nil {
		panic(err)
	}
	return []AdmissionPrecondition{{
		Key: keyspace.Active("key-1"), Kind: AdmissionCheckHashEqual,
		Field: "revision", Expected: "1", Failure: AdmissionUnavailable,
		Reason: "active_policy_changed",
	}}
}
