// Package usageaccounting normalizes authoritative provider usage and computes
// exact cost debits pinned to a Model pricing revision.
package usageaccounting

import (
	"errors"
	"fmt"
	"regexp"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/quota"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingsnapshot"
)

const priceFractionDigits = 9

var (
	pricePattern      = regexp.MustCompile(`^(0|[1-9][0-9]*)(\.[0-9]+)?$`)
	ErrInvalidPricing = errors.New("invalid model pricing")
	ErrInvalidUsage   = errors.New("invalid authoritative usage")
)

// OptionalPrice distinguishes an unpriced bucket from an explicitly free
// bucket. NanoPerMillion is exact nano-currency units per million tokens.
type OptionalPrice struct {
	Present        bool
	Canonical      string
	NanoPerMillion quota.QuotaInteger
}

// Pricing is the effective four-bucket value pinned in an immutable Model
// revision. Cache prices have already inherited Input where omitted or blank.
type Pricing struct {
	Currency   string
	Input      OptionalPrice
	Output     OptionalPrice
	CacheRead  OptionalPrice
	CacheWrite OptionalPrice
}

// PricingInput mirrors authoring semantics. Input/Output nil means unpriced;
// CacheRead/CacheWrite nil or blank means inherit Input. A pointer to "0" is
// explicitly free and never inherits.
type PricingInput struct {
	Currency   string
	Input      *string
	Output     *string
	CacheRead  *string
	CacheWrite *string
}

func CompilePricing(input PricingInput) (Pricing, error) {
	if !regexp.MustCompile(`^[A-Z]{3}$`).MatchString(input.Currency) {
		return Pricing{}, fmt.Errorf("%w: currency must be an ISO-4217 code", ErrInvalidPricing)
	}
	inputPrice, err := parseOptionalPrice(input.Input, false)
	if err != nil {
		return Pricing{}, fmt.Errorf("%w: input: %w", ErrInvalidPricing, err)
	}
	outputPrice, err := parseOptionalPrice(input.Output, false)
	if err != nil {
		return Pricing{}, fmt.Errorf("%w: output: %w", ErrInvalidPricing, err)
	}
	cacheRead, err := parseCachePrice(input.CacheRead, inputPrice)
	if err != nil {
		return Pricing{}, fmt.Errorf("%w: cache read: %w", ErrInvalidPricing, err)
	}
	cacheWrite, err := parseCachePrice(input.CacheWrite, inputPrice)
	if err != nil {
		return Pricing{}, fmt.Errorf("%w: cache write: %w", ErrInvalidPricing, err)
	}
	return Pricing{
		Currency:   input.Currency,
		Input:      inputPrice,
		Output:     outputPrice,
		CacheRead:  cacheRead,
		CacheWrite: cacheWrite,
	}, nil
}

// CompileSnapshotPricing is the only bridge from an immutable routing Model
// revision into accounting. It preserves nil versus explicit zero and relies
// on the same cache-inheritance rules used by standalone and managed snapshots.
func CompileSnapshotPricing(currency string, model routingsnapshot.Model) (Pricing, error) {
	return CompilePricing(PricingInput{
		Currency:   currency,
		Input:      cloneString(model.Pricing.InputCostPerMillionTokens),
		Output:     cloneString(model.Pricing.OutputCostPerMillionTokens),
		CacheRead:  cloneString(model.Pricing.CacheReadCostPerMillionTokens),
		CacheWrite: cloneString(model.Pricing.CacheWriteCostPerMillionTokens),
	})
}

func cloneString(value *string) *string {
	if value == nil {
		return nil
	}
	copy := *value
	return &copy
}

func parseCachePrice(value *string, inherited OptionalPrice) (OptionalPrice, error) {
	if value == nil || *value == "" {
		return inherited, nil
	}
	return parseOptionalPrice(value, false)
}

func parseOptionalPrice(value *string, blankAllowed bool) (OptionalPrice, error) {
	if value == nil {
		return OptionalPrice{}, nil
	}
	if *value == "" && blankAllowed {
		return OptionalPrice{}, nil
	}
	canonical, nano, err := parsePrice(*value)
	if err != nil {
		return OptionalPrice{}, err
	}
	return OptionalPrice{Present: true, Canonical: canonical, NanoPerMillion: nano}, nil
}

func parsePrice(value string) (string, quota.QuotaInteger, error) {
	if !pricePattern.MatchString(value) {
		return "", quota.QuotaInteger{}, errors.New("must be a plain non-negative decimal")
	}
	whole, fraction, hasPoint := strings.Cut(value, ".")
	if hasPoint {
		if len(fraction) > priceFractionDigits {
			return "", quota.QuotaInteger{}, fmt.Errorf("supports at most %d fractional digits", priceFractionDigits)
		}
		fraction = strings.TrimRight(fraction, "0")
	}
	canonical := whole
	if fraction != "" {
		canonical += "." + fraction
	}
	if len(whole) > 7 ||
		(len(whole) == 7 && (whole > "1000000" || (whole == "1000000" && strings.Trim(fraction, "0") != ""))) {
		return "", quota.QuotaInteger{}, errors.New("must not exceed 1000000")
	}
	nanoText := strings.TrimLeft(whole+fraction+strings.Repeat("0", priceFractionDigits-len(fraction)), "0")
	if nanoText == "" {
		nanoText = "0"
	}
	nano, err := quota.ParseQuotaInteger(nanoText)
	if err != nil {
		return "", quota.QuotaInteger{}, err
	}
	return canonical, nano, nil
}

// ActualUsage carries backend-authoritative billing buckets. InputTotal and
// OutputKnown report whether those totals were authoritatively supplied. Cache
// flags allow providers without cache detail to remain exactly billable when
// cache rates equal the normal input rate.
type ActualUsage struct {
	InputTotal      quota.QuotaInteger
	InputKnown      bool
	Output          quota.QuotaInteger
	OutputKnown     bool
	CacheRead       quota.QuotaInteger
	CacheReadKnown  bool
	CacheWrite      quota.QuotaInteger
	CacheWriteKnown bool
}

type CostCompleteness string

const (
	CostComplete CostCompleteness = "complete"
	CostUnknown  CostCompleteness = "unknown"
)

// Cost is exact at the public 10^-15 currency scale. Unknown cost never
// masquerades as zero; callers must install the configured usage fence.
type Cost struct {
	Currency     string
	Numerator    quota.QuotaInteger
	Completeness CostCompleteness
	Reason       string
}

func CalculateCost(pricing Pricing, usage ActualUsage) (Cost, error) {
	result := Cost{Currency: pricing.Currency, Completeness: CostComplete}
	if !usage.InputKnown {
		return unknownCost(pricing.Currency, "input_tokens_missing"), nil
	}
	if !usage.OutputKnown {
		return unknownCost(pricing.Currency, "output_tokens_missing"), nil
	}
	if (!usage.CacheReadKnown && !usage.CacheRead.IsZero()) ||
		(!usage.CacheWriteKnown && !usage.CacheWrite.IsZero()) {
		return Cost{}, fmt.Errorf("%w: unknown cache buckets must not carry token values", ErrInvalidUsage)
	}

	knownCache, err := usage.CacheRead.Add(usage.CacheWrite)
	if err != nil || knownCache.Compare(usage.InputTotal) > 0 {
		return Cost{}, fmt.Errorf("%w: cache tokens exceed input total", ErrInvalidUsage)
	}
	remainingInput, _ := usage.InputTotal.Sub(knownCache)

	inputUnknown := (!usage.CacheReadKnown && !samePrice(pricing.CacheRead, pricing.Input)) ||
		(!usage.CacheWriteKnown && !samePrice(pricing.CacheWrite, pricing.Input))
	if inputUnknown && !usage.InputTotal.IsZero() {
		return unknownCost(pricing.Currency, "cache_breakdown_missing"), nil
	}

	charges := []struct {
		tokens quota.QuotaInteger
		price  OptionalPrice
	}{
		{tokens: remainingInput, price: pricing.Input},
		{tokens: usage.Output, price: pricing.Output},
	}
	if usage.CacheReadKnown {
		charges = append(charges, struct {
			tokens quota.QuotaInteger
			price  OptionalPrice
		}{tokens: usage.CacheRead, price: pricing.CacheRead})
	}
	if usage.CacheWriteKnown {
		charges = append(charges, struct {
			tokens quota.QuotaInteger
			price  OptionalPrice
		}{tokens: usage.CacheWrite, price: pricing.CacheWrite})
	}

	for _, charge := range charges {
		if charge.tokens.IsZero() {
			continue
		}
		if !charge.price.Present {
			return unknownCost(pricing.Currency, "required_rate_missing"), nil
		}
		debit, err := charge.tokens.Mul(charge.price.NanoPerMillion)
		if err != nil {
			return Cost{}, fmt.Errorf("cost numerator overflow: %w", err)
		}
		result.Numerator, err = result.Numerator.Add(debit)
		if err != nil {
			return Cost{}, fmt.Errorf("cost numerator overflow: %w", err)
		}
	}
	return result, nil
}

func samePrice(left, right OptionalPrice) bool {
	return left.Present == right.Present && (!left.Present || left.NanoPerMillion == right.NanoPerMillion)
}

func unknownCost(currency, reason string) Cost {
	return Cost{Currency: currency, Completeness: CostUnknown, Reason: reason}
}
