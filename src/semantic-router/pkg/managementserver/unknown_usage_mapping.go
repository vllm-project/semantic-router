package managementserver

import (
	"strconv"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementapi"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/quotareconciliation"
)

func unknownUsageResponse(value quotareconciliation.Fence, internal, evidence, actor bool) managementapi.UnknownUsageFence {
	meters := make([]managementapi.UnknownUsageFenceMeter, len(value.Bindings))
	for index, binding := range value.Bindings {
		window := ""
		if binding.Window > 0 {
			window = binding.Window.String()
		}
		meters[index] = managementapi.UnknownUsageFenceMeter{
			BindingID: binding.BindingID, RuleID: binding.RuleID, PolicyID: binding.PolicyID,
			SubjectKind: string(binding.Subject.Kind), SubjectID: binding.Subject.ID,
			Metric: string(binding.Metric), Algorithm: string(binding.Algorithm), Enforcement: string(binding.Enforcement),
			AdmissionLimit: binding.AdmissionLimit, MaximumDebit: binding.MaximumDebit,
			Window: window, CalendarPeriod: string(binding.CalendarPeriod), Timezone: binding.Timezone, Currency: binding.Currency,
		}
	}
	result := managementapi.UnknownUsageFence{
		FenceID: value.ID, AdmissionID: value.AdmissionID, State: string(value.State), Revision: value.Revision,
		Reason: value.Reason, Meters: meters, KnownCharge: unknownUsageCharge(value.KnownCharge),
		CreatedAt: value.CreatedAt, UpdatedAt: value.UpdatedAt, ResolvedAt: cloneResponseTime(value.ResolvedAt),
	}
	if internal {
		result.Dispatches = make([]managementapi.UnknownUsageDispatch, len(value.Unknown))
		for index, dispatch := range value.Unknown {
			result.Dispatches[index] = managementapi.UnknownUsageDispatch{
				DispatchID: dispatch.DispatchID, ModelID: dispatch.ModelID, BackendID: dispatch.BackendID,
				ProviderID: dispatch.ProviderID, ProviderModelID: dispatch.ProviderModelID,
				PricingRevision: dispatch.PricingRevision,
			}
		}
	}
	if evidence {
		result.Evidence = make([]managementapi.UnknownUsageEvidence, len(value.Unknown))
		for index, dispatch := range value.Unknown {
			result.Evidence[index] = managementapi.UnknownUsageEvidence{
				DispatchID: dispatch.DispatchID, EvidenceDigest: dispatch.EvidenceDigest, Reason: dispatch.Reason,
			}
		}
	}
	if value.Reconciliation != nil {
		result.Reconciliation = &managementapi.UnknownUsageReconciliation{
			ReconciliationID: value.Reconciliation.ID, Strategy: string(value.Reconciliation.Strategy),
			CreatedAt: value.Reconciliation.CreatedAt, AppliedAt: cloneResponseTime(value.Reconciliation.AppliedAt),
		}
		if actor {
			result.Reconciliation.ActorPrincipalID = value.Reconciliation.ActorID
			result.Reconciliation.Reason = value.Reconciliation.Reason
		}
	}
	return result
}

func unknownUsageCharge(value quotareconciliation.Charge) managementapi.UnknownUsageCharge {
	costs := make([]managementapi.UnknownUsageCost, len(value.Costs))
	for index, cost := range value.Costs {
		costs[index] = managementapi.UnknownUsageCost{Currency: cost.Currency, Numerator: cost.Numerator}
	}
	return managementapi.UnknownUsageCharge{InputTokens: value.InputTokens, OutputTokens: value.OutputTokens, TotalTokens: value.TotalTokens, Costs: costs}
}

func unknownUsageActual(value *managementapi.UnknownUsageActual) *quotareconciliation.ActualUsage {
	if value == nil {
		return nil
	}
	dispatches := make([]quotareconciliation.ActualDispatchUsage, len(value.Dispatches))
	for index, dispatch := range value.Dispatches {
		dispatches[index] = quotareconciliation.ActualDispatchUsage{
			DispatchID: dispatch.DispatchID, EvidenceDigest: dispatch.EvidenceDigest,
			InputTokens: string(dispatch.InputTokens), CacheReadTokens: string(dispatch.CacheReadTokens),
			CacheWriteTokens: string(dispatch.CacheWriteTokens), OutputTokens: string(dispatch.OutputTokens),
			Cost: quotareconciliation.Cost{Currency: dispatch.Cost.Currency, Numerator: dispatch.Cost.Numerator},
		}
	}
	return &quotareconciliation.ActualUsage{
		Dispatches:        dispatches,
		ServedInputTokens: string(value.ServedInputTokens), ServedOutputTokens: string(value.ServedOutputTokens),
	}
}

func unknownUsageOperation(value quotareconciliation.Operation) managementapi.Operation {
	return managementapi.Operation{
		OperationID: value.ID, Kind: value.Kind, State: managementapi.OperationState(value.State),
		Progress: managementapi.OperationProgress{
			Total:     managementapi.WholeQuantity(strconv.FormatUint(value.Total, 10)),
			Completed: managementapi.WholeQuantity(strconv.FormatUint(value.Completed, 10)), Failed: "0",
		},
		TargetIDs: []string{value.FenceID}, Revisions: managementapi.RevisionState{},
		CreatedAt: value.CreatedAt, UpdatedAt: value.UpdatedAt, CompletedAt: cloneResponseTime(value.CompletedAt),
	}
}
