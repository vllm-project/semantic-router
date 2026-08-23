package usageaccounting

import (
	"errors"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/quota"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingsnapshot"
)

func stringPointer(value string) *string { return &value }

func mustQuantity(t *testing.T, value string) quota.QuotaInteger {
	t.Helper()
	parsed, err := quota.ParseQuotaInteger(value)
	if err != nil {
		t.Fatalf("ParseQuotaInteger(%q) error = %v", value, err)
	}
	return parsed
}

func TestCompilePricingInheritanceAndExplicitZero(t *testing.T) {
	pricing, err := CompilePricing(PricingInput{
		Currency:   "USD",
		Input:      stringPointer("0.500000000"),
		Output:     stringPointer("1.5"),
		CacheWrite: stringPointer("0"),
	})
	if err != nil {
		t.Fatalf("CompilePricing() error = %v", err)
	}
	if pricing.Input.Canonical != "0.5" || pricing.CacheRead.Canonical != "0.5" {
		t.Fatalf("cache read did not inherit normalized input: %+v", pricing)
	}
	if !pricing.CacheWrite.Present || pricing.CacheWrite.Canonical != "0" {
		t.Fatalf("explicit zero was not preserved: %+v", pricing.CacheWrite)
	}
	if pricing.Input.NanoPerMillion.String() != "500000000" {
		t.Fatalf("input nano rate = %s", pricing.Input.NanoPerMillion.String())
	}
}

func TestCompileSnapshotPricingPreservesExplicitZeroAndInheritance(t *testing.T) {
	zero := "0"
	input := "0.5"
	output := "1.25"
	pricing, err := CompileSnapshotPricing("USD", routingsnapshot.Model{Pricing: routingsnapshot.ModelPricing{
		InputCostPerMillionTokens: &input, OutputCostPerMillionTokens: &output,
		CacheWriteCostPerMillionTokens: &zero,
	}})
	if err != nil {
		t.Fatal(err)
	}
	if pricing.CacheRead.Canonical != "0.5" || !pricing.CacheWrite.Present || pricing.CacheWrite.Canonical != "0" {
		t.Fatalf("compiled snapshot pricing = %+v", pricing)
	}
}

func TestCompilePricingRejectsInvalidValues(t *testing.T) {
	for _, input := range []PricingInput{
		{Currency: "usd", Input: stringPointer("1")},
		{Currency: "USD", Input: stringPointer("")},
		{Currency: "USD", Input: stringPointer("1e3")},
		{Currency: "USD", Input: stringPointer("0.0000000001")},
		{Currency: "USD", Input: stringPointer("1000000.1")},
	} {
		if _, err := CompilePricing(input); !errors.Is(err, ErrInvalidPricing) {
			t.Fatalf("CompilePricing(%+v) error = %v", input, err)
		}
	}
}

func TestCalculateCostUsesActualBillingBuckets(t *testing.T) {
	pricing, err := CompilePricing(PricingInput{
		Currency:   "USD",
		Input:      stringPointer("0.5"),
		Output:     stringPointer("1.5"),
		CacheRead:  stringPointer("0.05"),
		CacheWrite: stringPointer("0.625"),
	})
	if err != nil {
		t.Fatal(err)
	}
	cost, err := CalculateCost(pricing, ActualUsage{
		InputTotal:      mustQuantity(t, "1000"),
		InputKnown:      true,
		Output:          mustQuantity(t, "200"),
		OutputKnown:     true,
		CacheRead:       mustQuantity(t, "300"),
		CacheReadKnown:  true,
		CacheWrite:      mustQuantity(t, "100"),
		CacheWriteKnown: true,
	})
	if err != nil {
		t.Fatalf("CalculateCost() error = %v", err)
	}
	// 600*0.5 + 300*0.05 + 100*0.625 + 200*1.5 = 677.5
	// token-price products are the exact 10^-15 currency numerator.
	if cost.Completeness != CostComplete || cost.Numerator.String() != "677500000000" {
		t.Fatalf("CalculateCost() = %+v", cost)
	}
	if got := quota.NewCurrencyDecimalFromScaled(cost.Numerator).String(); got != "0.0006775" {
		t.Fatalf("public cost = %s, want 0.0006775", got)
	}
}

func TestCalculateCostAllowsMissingCacheBreakdownOnlyAtEqualRates(t *testing.T) {
	inherited, err := CompilePricing(PricingInput{
		Currency: "USD", Input: stringPointer("1"), Output: stringPointer("2"),
	})
	if err != nil {
		t.Fatal(err)
	}
	usage := ActualUsage{
		InputTotal: mustQuantity(t, "10"), InputKnown: true,
		Output: mustQuantity(t, "2"), OutputKnown: true,
	}
	cost, err := CalculateCost(inherited, usage)
	if err != nil || cost.Completeness != CostComplete {
		t.Fatalf("equal-rate missing breakdown = (%+v, %v)", cost, err)
	}

	differential, err := CompilePricing(PricingInput{
		Currency: "USD", Input: stringPointer("1"), Output: stringPointer("2"), CacheRead: stringPointer("0.1"),
	})
	if err != nil {
		t.Fatal(err)
	}
	cost, err = CalculateCost(differential, usage)
	if err != nil || cost.Completeness != CostUnknown || cost.Reason != "cache_breakdown_missing" {
		t.Fatalf("differential missing breakdown = (%+v, %v)", cost, err)
	}
}

func TestCalculateCostDistinguishesFreeAndUnpriced(t *testing.T) {
	free, _ := CompilePricing(PricingInput{Currency: "USD", Input: stringPointer("0"), Output: stringPointer("0")})
	usage := ActualUsage{InputTotal: mustQuantity(t, "10"), InputKnown: true, Output: mustQuantity(t, "1"), OutputKnown: true}
	cost, err := CalculateCost(free, usage)
	if err != nil || cost.Completeness != CostComplete || !cost.Numerator.IsZero() {
		t.Fatalf("free cost = (%+v, %v)", cost, err)
	}

	unpriced, _ := CompilePricing(PricingInput{Currency: "USD"})
	cost, err = CalculateCost(unpriced, usage)
	if err != nil || cost.Completeness != CostUnknown || cost.Reason != "required_rate_missing" {
		t.Fatalf("unpriced cost = (%+v, %v)", cost, err)
	}
}

func TestCalculateCostRejectsImpossibleCacheUsage(t *testing.T) {
	pricing, _ := CompilePricing(PricingInput{Currency: "USD", Input: stringPointer("1"), Output: stringPointer("1")})
	_, err := CalculateCost(pricing, ActualUsage{
		InputTotal: mustQuantity(t, "5"), InputKnown: true,
		OutputKnown: true,
		CacheRead:   mustQuantity(t, "6"), CacheReadKnown: true,
	})
	if !errors.Is(err, ErrInvalidUsage) {
		t.Fatalf("CalculateCost() error = %v, want %v", err, ErrInvalidUsage)
	}
}
