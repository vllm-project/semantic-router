package usageaccounting

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"sync"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/quota"
)

type EvidenceState string

const (
	EvidenceKnownZero   EvidenceState = "known_zero"
	EvidenceKnownActual EvidenceState = "known_actual"
	EvidenceUnknown     EvidenceState = "unknown"
)

type DispatchUsage struct {
	DispatchID    string
	ModelID       string
	ModelRevision int64
	State         EvidenceState
	Usage         ActualUsage
	Pricing       Pricing
	Reason        string
}

type ServedUsage struct {
	Input       quota.QuotaInteger
	InputKnown  bool
	Output      quota.QuotaInteger
	OutputKnown bool
}

type MetricValue struct {
	Value    quota.QuotaInteger
	Complete bool
	Reason   string
}

// Aggregate is an immutable request-level accounting result. Backend totals
// include every Recipe dispatch exactly once; served totals describe only the
// canonical external request and response.
type Aggregate struct {
	Input                MetricValue
	Output               MetricValue
	Total                MetricValue
	ServedInput          MetricValue
	ServedOutput         MetricValue
	ServedTotal          MetricValue
	Cost                 MetricValue
	Currency             string
	KnownDispatches      quota.QuotaInteger
	IncompleteDispatches quota.QuotaInteger
	Digest               string
}

type Aggregator struct {
	mu         sync.Mutex
	dispatches map[string]DispatchUsage
	served     ServedUsage
	servedSet  bool
	finalized  bool
}

func NewAggregator() *Aggregator {
	return &Aggregator{dispatches: make(map[string]DispatchUsage)}
}

func (a *Aggregator) RecordDispatch(value DispatchUsage) error {
	if a == nil {
		return fmt.Errorf("usage aggregator is nil")
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.finalized {
		return fmt.Errorf("usage aggregator is already finalized")
	}
	if strings.TrimSpace(value.DispatchID) == "" || strings.TrimSpace(value.ModelID) == "" || value.ModelRevision <= 0 {
		return fmt.Errorf("dispatch ID, model ID, and positive model revision are required")
	}
	if strings.TrimSpace(value.Pricing.Currency) == "" {
		return fmt.Errorf("dispatch requires its pinned pricing currency")
	}
	if _, exists := a.dispatches[value.DispatchID]; exists {
		return fmt.Errorf("dispatch %q is already recorded", value.DispatchID)
	}
	switch value.State {
	case EvidenceKnownZero:
		if !usageIsZero(value.Usage) {
			return fmt.Errorf("known-zero dispatch must not carry usage")
		}
	case EvidenceKnownActual:
	case EvidenceUnknown:
		if strings.TrimSpace(value.Reason) == "" {
			return fmt.Errorf("unknown dispatch requires a reason")
		}
		if !usageIsZero(value.Usage) {
			return fmt.Errorf("unknown dispatch must not claim usage")
		}
	default:
		return fmt.Errorf("unsupported dispatch evidence state %q", value.State)
	}
	a.dispatches[value.DispatchID] = value
	return nil
}

func (a *Aggregator) SetServedUsage(value ServedUsage) error {
	if a == nil {
		return fmt.Errorf("usage aggregator is nil")
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.finalized {
		return fmt.Errorf("usage aggregator is already finalized")
	}
	if a.servedSet {
		return fmt.Errorf("served usage is already recorded")
	}
	a.served = value
	a.servedSet = true
	return nil
}

func (a *Aggregator) Finalize() (Aggregate, error) {
	if a == nil {
		return Aggregate{}, fmt.Errorf("usage aggregator is nil")
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.finalized {
		return Aggregate{}, fmt.Errorf("usage aggregator is already finalized")
	}
	if len(a.dispatches) == 0 {
		return Aggregate{}, fmt.Errorf("at least one dispatch evidence record is required")
	}
	a.finalized = true
	ids := make([]string, 0, len(a.dispatches))
	for id := range a.dispatches {
		ids = append(ids, id)
	}
	sort.Strings(ids)

	result := Aggregate{
		Input: MetricValue{Complete: true}, Output: MetricValue{Complete: true}, Total: MetricValue{Complete: true},
		ServedInput:  MetricValue{Value: a.served.Input, Complete: a.servedSet && a.served.InputKnown},
		ServedOutput: MetricValue{Value: a.served.Output, Complete: a.servedSet && a.served.OutputKnown},
		Cost:         MetricValue{Complete: true},
	}
	result.ServedInput.Reason = completenessReason(result.ServedInput.Complete, "served_input_tokens_missing")
	result.ServedOutput.Reason = completenessReason(result.ServedOutput.Complete, "served_output_tokens_missing")
	var finalizeErr error
	result.ServedTotal, finalizeErr = addMetrics(result.ServedInput, result.ServedOutput)
	if finalizeErr != nil {
		return Aggregate{}, fmt.Errorf("served token total overflow: %w", finalizeErr)
	}

	canonical := make([]canonicalDispatchUsage, 0, len(ids))
	for _, id := range ids {
		dispatch := a.dispatches[id]
		canonical = append(canonical, canonicalizeDispatch(dispatch))
		if result.Currency == "" {
			result.Currency = dispatch.Pricing.Currency
		} else if result.Currency != dispatch.Pricing.Currency {
			return Aggregate{}, fmt.Errorf("dispatch currencies do not match")
		}
		switch dispatch.State {
		case EvidenceKnownZero:
			result.KnownDispatches, _ = addSmall(result.KnownDispatches, 1)
		case EvidenceUnknown:
			result.IncompleteDispatches, _ = addSmall(result.IncompleteDispatches, 1)
			markIncomplete(&result.Input, dispatch.Reason)
			markIncomplete(&result.Output, dispatch.Reason)
			markIncomplete(&result.Cost, dispatch.Reason)
		case EvidenceKnownActual:
			result.KnownDispatches, _ = addSmall(result.KnownDispatches, 1)
			if dispatch.Usage.InputKnown {
				var err error
				result.Input.Value, err = result.Input.Value.Add(dispatch.Usage.InputTotal)
				if err != nil {
					return Aggregate{}, fmt.Errorf("input token total overflow: %w", err)
				}
			} else {
				markIncomplete(&result.Input, "input_tokens_missing")
			}
			if dispatch.Usage.OutputKnown {
				var err error
				result.Output.Value, err = result.Output.Value.Add(dispatch.Usage.Output)
				if err != nil {
					return Aggregate{}, fmt.Errorf("output token total overflow: %w", err)
				}
			} else {
				markIncomplete(&result.Output, "output_tokens_missing")
			}
			cost, err := CalculateCost(dispatch.Pricing, dispatch.Usage)
			if err != nil {
				return Aggregate{}, fmt.Errorf("dispatch %s cost: %w", dispatch.DispatchID, err)
			}
			if result.Currency != cost.Currency {
				return Aggregate{}, fmt.Errorf("dispatch currencies do not match")
			}
			if cost.Completeness == CostUnknown {
				markIncomplete(&result.Cost, cost.Reason)
			} else {
				result.Cost.Value, err = result.Cost.Value.Add(cost.Numerator)
				if err != nil {
					return Aggregate{}, fmt.Errorf("cost total overflow: %w", err)
				}
			}
		}
	}
	result.Total, finalizeErr = addMetrics(result.Input, result.Output)
	if finalizeErr != nil {
		return Aggregate{}, fmt.Errorf("total token overflow: %w", finalizeErr)
	}
	payload, finalizeErr := json.Marshal(struct {
		Dispatches []canonicalDispatchUsage `json:"dispatches"`
		Served     canonicalServedUsage     `json:"served"`
	}{Dispatches: canonical, Served: canonicalizeServed(a.served, a.servedSet)})
	if finalizeErr != nil {
		return Aggregate{}, fmt.Errorf("encode canonical usage: %w", finalizeErr)
	}
	digest := sha256.Sum256(payload)
	result.Digest = hex.EncodeToString(digest[:])
	return result, nil
}

func (a Aggregate) Metric(metric quota.Metric) MetricValue {
	switch metric {
	case quota.MetricInputTokens:
		return a.Input
	case quota.MetricOutputTokens:
		return a.Output
	case quota.MetricTotalTokens:
		return a.Total
	case quota.MetricServedInputTokens:
		return a.ServedInput
	case quota.MetricServedOutputTokens:
		return a.ServedOutput
	case quota.MetricServedTotalTokens:
		return a.ServedTotal
	case quota.MetricCost:
		return a.Cost
	default:
		return MetricValue{Complete: false, Reason: "metric_is_not_response_actual"}
	}
}

func addMetrics(left, right MetricValue) (MetricValue, error) {
	result := MetricValue{Complete: left.Complete && right.Complete}
	if !left.Complete {
		result.Reason = left.Reason
	} else if !right.Complete {
		result.Reason = right.Reason
	}
	value, err := left.Value.Add(right.Value)
	if err != nil {
		return MetricValue{}, err
	}
	result.Value = value
	return result, nil
}

func markIncomplete(value *MetricValue, reason string) {
	if value.Complete {
		value.Complete = false
		value.Reason = reason
	}
}

func completenessReason(complete bool, reason string) string {
	if complete {
		return ""
	}
	return reason
}

func addSmall(value quota.QuotaInteger, amount uint64) (quota.QuotaInteger, error) {
	parsed, _ := quota.ParseQuotaInteger(fmt.Sprintf("%d", amount))
	return value.Add(parsed)
}

func usageIsZero(value ActualUsage) bool {
	return value.InputTotal.IsZero() && value.Output.IsZero() && value.CacheRead.IsZero() && value.CacheWrite.IsZero()
}

type canonicalDispatchUsage struct {
	DispatchID      string        `json:"dispatchId"`
	ModelID         string        `json:"modelId"`
	ModelRevision   int64         `json:"modelRevision"`
	State           EvidenceState `json:"state"`
	Input           string        `json:"input"`
	InputKnown      bool          `json:"inputKnown"`
	Output          string        `json:"output"`
	OutputKnown     bool          `json:"outputKnown"`
	CacheRead       string        `json:"cacheRead"`
	CacheReadKnown  bool          `json:"cacheReadKnown"`
	CacheWrite      string        `json:"cacheWrite"`
	CacheWriteKnown bool          `json:"cacheWriteKnown"`
	Currency        string        `json:"currency"`
	Reason          string        `json:"reason"`
}

func canonicalizeDispatch(value DispatchUsage) canonicalDispatchUsage {
	return canonicalDispatchUsage{
		DispatchID: value.DispatchID, ModelID: value.ModelID, ModelRevision: value.ModelRevision, State: value.State,
		Input: value.Usage.InputTotal.String(), InputKnown: value.Usage.InputKnown,
		Output: value.Usage.Output.String(), OutputKnown: value.Usage.OutputKnown,
		CacheRead: value.Usage.CacheRead.String(), CacheReadKnown: value.Usage.CacheReadKnown,
		CacheWrite: value.Usage.CacheWrite.String(), CacheWriteKnown: value.Usage.CacheWriteKnown,
		Currency: value.Pricing.Currency, Reason: value.Reason,
	}
}

type canonicalServedUsage struct {
	Set         bool   `json:"set"`
	Input       string `json:"input"`
	InputKnown  bool   `json:"inputKnown"`
	Output      string `json:"output"`
	OutputKnown bool   `json:"outputKnown"`
}

func canonicalizeServed(value ServedUsage, set bool) canonicalServedUsage {
	return canonicalServedUsage{Set: set, Input: value.Input.String(), InputKnown: value.InputKnown, Output: value.Output.String(), OutputKnown: value.OutputKnown}
}
