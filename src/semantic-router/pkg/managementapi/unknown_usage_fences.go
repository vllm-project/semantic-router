package managementapi

import "time"

type UnknownUsageFenceMeter struct {
	BindingID      string `json:"bindingId"`
	RuleID         string `json:"ruleId"`
	PolicyID       string `json:"policyId"`
	SubjectKind    string `json:"subjectKind"`
	SubjectID      string `json:"subjectId"`
	Metric         string `json:"metric"`
	Algorithm      string `json:"algorithm"`
	Enforcement    string `json:"enforcement"`
	AdmissionLimit string `json:"admissionLimit,omitempty"`
	MaximumDebit   string `json:"maximumDebit,omitempty"`
	Window         string `json:"window,omitempty"`
	CalendarPeriod string `json:"calendarPeriod,omitempty"`
	Timezone       string `json:"timezone,omitempty"`
	Currency       string `json:"currency,omitempty"`
}

type UnknownUsageCost struct {
	Currency  string `json:"currency"`
	Numerator string `json:"numerator"`
}

type UnknownUsageCharge struct {
	InputTokens  string             `json:"inputTokens"`
	OutputTokens string             `json:"outputTokens"`
	TotalTokens  string             `json:"totalTokens"`
	Costs        []UnknownUsageCost `json:"costs"`
}

type UnknownUsageDispatch struct {
	DispatchID      string `json:"dispatchId"`
	ModelID         string `json:"modelId,omitempty"`
	BackendID       string `json:"backendId,omitempty"`
	ProviderID      string `json:"providerId,omitempty"`
	ProviderModelID string `json:"providerModelId,omitempty"`
	PricingRevision int64  `json:"pricingRevision,omitempty"`
}

type UnknownUsageEvidence struct {
	DispatchID     string `json:"dispatchId"`
	EvidenceDigest string `json:"evidenceDigest"`
	Reason         string `json:"reason"`
}

type UnknownUsageReconciliation struct {
	ReconciliationID string     `json:"reconciliationId"`
	Strategy         string     `json:"strategy"`
	ActorPrincipalID string     `json:"actorPrincipalId,omitempty"`
	Reason           string     `json:"reason,omitempty"`
	CreatedAt        time.Time  `json:"createdAt"`
	AppliedAt        *time.Time `json:"appliedAt,omitempty"`
}

type UnknownUsageFence struct {
	FenceID        string                      `json:"fenceId"`
	AdmissionID    string                      `json:"admissionId"`
	State          string                      `json:"state"`
	Revision       uint64                      `json:"revision"`
	Reason         string                      `json:"reason"`
	Meters         []UnknownUsageFenceMeter    `json:"meters"`
	KnownCharge    UnknownUsageCharge          `json:"knownCharge"`
	Dispatches     []UnknownUsageDispatch      `json:"dispatches,omitempty"`
	Evidence       []UnknownUsageEvidence      `json:"evidence,omitempty"`
	Reconciliation *UnknownUsageReconciliation `json:"reconciliation,omitempty"`
	CreatedAt      time.Time                   `json:"createdAt"`
	UpdatedAt      time.Time                   `json:"updatedAt"`
	ResolvedAt     *time.Time                  `json:"resolvedAt,omitempty"`
}

type UnknownUsageFencePage = Page[UnknownUsageFence]

type UnknownUsageFenceDetail struct {
	Data UnknownUsageFence `json:"data"`
}

type UnknownUsageActualDispatch struct {
	DispatchID       string           `json:"dispatchId"`
	EvidenceDigest   string           `json:"evidenceDigest"`
	InputTokens      WholeQuantity    `json:"inputTokens"`
	CacheReadTokens  WholeQuantity    `json:"cacheReadTokens"`
	CacheWriteTokens WholeQuantity    `json:"cacheWriteTokens"`
	OutputTokens     WholeQuantity    `json:"outputTokens"`
	Cost             UnknownUsageCost `json:"cost"`
}

type UnknownUsageActual struct {
	Dispatches         []UnknownUsageActualDispatch `json:"dispatches"`
	ServedInputTokens  WholeQuantity                `json:"servedInputTokens"`
	ServedOutputTokens WholeQuantity                `json:"servedOutputTokens"`
}

type UnknownUsageReconcileRequest struct {
	Strategy           string              `json:"strategy"`
	Actual             *UnknownUsageActual `json:"actual,omitempty"`
	EvidenceReferences []string            `json:"evidenceReferences"`
	Reason             string              `json:"reason"`
}
