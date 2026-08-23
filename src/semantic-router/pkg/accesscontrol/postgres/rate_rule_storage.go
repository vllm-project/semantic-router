package postgres

import (
	"database/sql"
	"fmt"
	"math"
	"math/big"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
)

const (
	costScaleDigits = 15
	maxQuotaDigits  = 42
)

var costScale = new(big.Int).Exp(big.NewInt(10), big.NewInt(costScaleDigits), nil)

type storedRateLimitRule struct {
	limitValue                 any
	windowSeconds              any
	calendarPeriod             any
	timezone                   any
	bucketCapacity             any
	refillAmount               any
	refillPeriodMilliseconds   any
	gcraEmissionIntervalMicros any
	gcraBurstTolerance         any
}

func encodeRateLimitRule(rule accesscontrol.RateLimitRule) (storedRateLimitRule, error) {
	if rule.Ordinal > math.MaxInt32 {
		return storedRateLimitRule{}, fmt.Errorf("rate-limit rule ordinal exceeds PostgreSQL INTEGER")
	}
	stored := storedRateLimitRule{
		calendarPeriod: nullableString(string(rule.CalendarPeriod)),
		timezone:       nullableString(rule.Timezone),
	}
	if rule.Limit != "" {
		value := string(rule.Limit)
		if rule.Metric == accesscontrol.RateMetricCost {
			var err error
			value, err = scaleCostDecimal(value)
			if err != nil {
				return storedRateLimitRule{}, fmt.Errorf("encode cost limit: %w", err)
			}
		}
		stored.limitValue = value
	}
	if rule.Window != 0 {
		stored.windowSeconds = int64(rule.Window / time.Second)
	}
	if rule.BucketCapacity != "" {
		stored.bucketCapacity = string(rule.BucketCapacity)
	}
	if rule.RefillAmount != "" {
		stored.refillAmount = string(rule.RefillAmount)
	}
	if rule.RefillPeriod != 0 {
		stored.refillPeriodMilliseconds = int64(rule.RefillPeriod / time.Millisecond)
	}
	if rule.GCRAEmissionInterval != 0 {
		stored.gcraEmissionIntervalMicros = int64(rule.GCRAEmissionInterval / time.Microsecond)
	}
	if rule.GCRABurstTolerance != nil {
		stored.gcraBurstTolerance = *rule.GCRABurstTolerance
	}
	return stored, nil
}

func scanRateLimitRule(scanner rowScanner) (accesscontrol.RateLimitRule, error) {
	var rule accesscontrol.RateLimitRule
	var stored scannedRateLimitRule
	if err := scanner.Scan(
		&rule.ID, &rule.PolicyID, &rule.Metric, &rule.Algorithm,
		&stored.limitValue, &stored.windowSeconds, &stored.calendarPeriod, &stored.timezone,
		&stored.bucketCapacity, &stored.refillAmount, &stored.refillPeriodMilliseconds,
		&stored.gcraEmissionIntervalMicros, &stored.gcraBurstTolerance,
		&rule.Accounting, &rule.Enforcement, &stored.ordinal,
	); err != nil {
		return accesscontrol.RateLimitRule{}, err
	}
	if err := stored.apply(&rule); err != nil {
		return accesscontrol.RateLimitRule{}, err
	}
	if err := rule.Validate(); err != nil {
		return accesscontrol.RateLimitRule{}, fmt.Errorf("validate stored rule: %w", err)
	}
	return rule, nil
}

type scannedRateLimitRule struct {
	limitValue                 sql.NullString
	windowSeconds              sql.NullInt64
	calendarPeriod             sql.NullString
	timezone                   sql.NullString
	bucketCapacity             sql.NullString
	refillAmount               sql.NullString
	refillPeriodMilliseconds   sql.NullInt64
	gcraEmissionIntervalMicros sql.NullInt64
	gcraBurstTolerance         sql.NullInt64
	ordinal                    int64
}

func (stored scannedRateLimitRule) apply(rule *accesscontrol.RateLimitRule) error {
	if stored.ordinal < 0 || stored.ordinal > math.MaxUint32 {
		return fmt.Errorf("database returned invalid rule ordinal %d", stored.ordinal)
	}
	rule.Ordinal = uint32(stored.ordinal)
	if err := stored.applyQuotaValues(rule); err != nil {
		return err
	}
	if err := stored.applyDurations(rule); err != nil {
		return err
	}
	stored.applyCalendar(rule)
	return stored.applyGCRA(rule)
}

func (stored scannedRateLimitRule) applyQuotaValues(rule *accesscontrol.RateLimitRule) error {
	if stored.limitValue.Valid {
		value := stored.limitValue.String
		if rule.Metric == accesscontrol.RateMetricCost {
			var err error
			value, err = unscaleCostInteger(value)
			if err != nil {
				return fmt.Errorf("decode cost limit: %w", err)
			}
		}
		rule.Limit = accesscontrol.QuotaValue(value)
	}
	if stored.bucketCapacity.Valid {
		rule.BucketCapacity = accesscontrol.QuotaValue(stored.bucketCapacity.String)
	}
	if stored.refillAmount.Valid {
		rule.RefillAmount = accesscontrol.QuotaValue(stored.refillAmount.String)
	}
	return nil
}

func (stored scannedRateLimitRule) applyDurations(rule *accesscontrol.RateLimitRule) error {
	var err error
	if rule.Window, err = scannedDuration(stored.windowSeconds, time.Second, "window_seconds"); err != nil {
		return err
	}
	if rule.RefillPeriod, err = scannedDuration(
		stored.refillPeriodMilliseconds, time.Millisecond, "refill_period_milliseconds",
	); err != nil {
		return err
	}
	rule.GCRAEmissionInterval, err = scannedDuration(
		stored.gcraEmissionIntervalMicros, time.Microsecond, "gcra_emission_interval_microseconds",
	)
	return err
}

func (stored scannedRateLimitRule) applyCalendar(rule *accesscontrol.RateLimitRule) {
	if stored.calendarPeriod.Valid {
		rule.CalendarPeriod = accesscontrol.CalendarPeriod(stored.calendarPeriod.String)
	}
	if stored.timezone.Valid {
		rule.Timezone = stored.timezone.String
	}
}

func (stored scannedRateLimitRule) applyGCRA(rule *accesscontrol.RateLimitRule) error {
	if !stored.gcraBurstTolerance.Valid {
		return nil
	}
	value := stored.gcraBurstTolerance.Int64
	rule.GCRABurstTolerance = &value
	return nil
}

func scaleCostDecimal(value string) (string, error) {
	parts := strings.Split(value, ".")
	if len(parts) > 2 || len(parts) == 0 || len(parts[0]) == 0 {
		return "", fmt.Errorf("invalid canonical cost decimal")
	}
	fraction := ""
	if len(parts) == 2 {
		fraction = parts[1]
	}
	if len(fraction) > costScaleDigits {
		return "", fmt.Errorf("cost decimal exceeds %d fractional digits", costScaleDigits)
	}
	digits := strings.TrimLeft(parts[0]+fraction+strings.Repeat("0", costScaleDigits-len(fraction)), "0")
	if digits == "" {
		digits = "0"
	}
	if len(digits) > maxQuotaDigits {
		return "", fmt.Errorf("scaled cost exceeds NUMERIC(%d,0)", maxQuotaDigits)
	}
	if _, ok := new(big.Int).SetString(digits, 10); !ok {
		return "", fmt.Errorf("invalid canonical cost decimal")
	}
	return digits, nil
}

func unscaleCostInteger(value string) (string, error) {
	numerator, ok := new(big.Int).SetString(value, 10)
	if !ok || numerator.Sign() < 0 {
		return "", fmt.Errorf("invalid non-negative cost numerator")
	}
	whole, remainder := new(big.Int), new(big.Int)
	whole.QuoRem(numerator, costScale, remainder)
	if remainder.Sign() == 0 {
		return whole.String(), nil
	}
	fraction := remainder.String()
	fraction = strings.Repeat("0", costScaleDigits-len(fraction)) + fraction
	fraction = strings.TrimRight(fraction, "0")
	return whole.String() + "." + fraction, nil
}

func scannedDuration(value sql.NullInt64, unit time.Duration, field string) (time.Duration, error) {
	if !value.Valid {
		return 0, nil
	}
	if value.Int64 <= 0 || value.Int64 > math.MaxInt64/int64(unit) {
		return 0, fmt.Errorf("database returned invalid %s", field)
	}
	return time.Duration(value.Int64) * unit, nil
}
