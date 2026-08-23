package managementapi

import "time"

const (
	APIVersion                = "v1"
	ContractVersion           = "0.4"
	BasePath                  = "/management/v1"
	JSONMediaType             = "application/vnd.vllm-semantic-router.management.v1+json"
	EventStreamMediaType      = "text/event-stream"
	HeaderIdempotencyKey      = "Idempotency-Key"
	HeaderIfMatch             = "If-Match"
	HeaderETag                = "ETag"
	HeaderRequestID           = "X-Request-Id"
	HeaderIdempotencyReplayed = "Idempotency-Replayed"
	HeaderSecretResultClaim   = "Secret-Result-Claim"
	HeaderNamespaceID         = "VLLM-SR-Namespace"
)

// ErrorResponse is the only public Management error envelope. Details are
// deliberately typed so handlers cannot accidentally serialize internal state.
type ErrorResponse struct {
	Error APIError `json:"error"`
}

type APIError struct {
	Code      string           `json:"code"`
	Message   string           `json:"message"`
	RequestID string           `json:"requestId,omitempty"`
	Details   []ErrorDetail    `json:"details,omitempty"`
	StepUp    *StepUpChallenge `json:"stepUp,omitempty"`
}

type ErrorDetail struct {
	Field  string `json:"field,omitempty"`
	Reason string `json:"reason"`
}

type StepUpChallenge struct {
	ChallengeID string    `json:"challengeId"`
	ExpiresAt   time.Time `json:"expiresAt"`
	Methods     []string  `json:"methods"`
}

// Page is the canonical keyset-paginated collection envelope. Offset fields
// intentionally do not exist.
type Page[T any] struct {
	Data []T      `json:"data"`
	Page PageInfo `json:"page"`
}

type PageInfo struct {
	NextCursor string `json:"nextCursor,omitempty"`
	HasMore    bool   `json:"hasMore"`
	PageSize   int    `json:"pageSize"`
}

type IdempotencyMetadata struct {
	Replayed          bool   `json:"replayed"`
	OriginalRequestID string `json:"originalRequestId,omitempty"`
}

type RevisionState struct {
	DesiredRevision     int64 `json:"desiredRevision"`
	StagedRevision      int64 `json:"stagedRevision,omitempty"`
	PublicationRevision int64 `json:"publicationRevision,omitempty"`
	AppliedRevision     int64 `json:"appliedRevision"`
}

type OperationState string

const (
	OperationPending            OperationState = "pending"
	OperationRunning            OperationState = "running"
	OperationSucceeded          OperationState = "succeeded"
	OperationPartiallySucceeded OperationState = "partially_succeeded"
	OperationFailed             OperationState = "failed"
	OperationCancelled          OperationState = "cancelled"
)

type Operation struct {
	OperationID string                 `json:"operationId"`
	Kind        string                 `json:"kind"`
	State       OperationState         `json:"state"`
	Progress    OperationProgress      `json:"progress"`
	TargetIDs   []string               `json:"targetIds,omitempty"`
	Revisions   RevisionState          `json:"revisions"`
	ItemErrors  []OperationItemFailure `json:"itemErrors,omitempty"`
	CreatedAt   time.Time              `json:"createdAt"`
	UpdatedAt   time.Time              `json:"updatedAt"`
	CompletedAt *time.Time             `json:"completedAt,omitempty"`
}

// Counts use decimal strings so bulk cardinalities retain exact JSON semantics
// in every generated client.
type OperationProgress struct {
	Total     WholeQuantity `json:"total"`
	Completed WholeQuantity `json:"completed"`
	Failed    WholeQuantity `json:"failed"`
}

type OperationItemFailure struct {
	ItemID string `json:"itemId,omitempty"`
	Code   string `json:"code"`
	Reason string `json:"reason"`
}

type SecretKind string

const (
	SecretKindInferenceAPIKey     SecretKind = "inference_api_key"
	SecretKindInvitationToken     SecretKind = "invitation_token"
	SecretKindServiceCredential   SecretKind = "service_credential"
	SecretKindDelegatedCredential SecretKind = "delegated_inference_credential"
	SecretKindManagementToken     SecretKind = "management_access_token"
)

// SecretEnvelope is the only synchronous generic wire type that may contain a
// generated credential. It is always paired with no-store response metadata.
type SecretEnvelope struct {
	ResourceID string     `json:"resourceId"`
	Kind       SecretKind `json:"kind"`
	Secret     string     `json:"secret"`
	ExpiresAt  *time.Time `json:"expiresAt,omitempty"`
}

type ManagementTokenEnvelope struct {
	AccessToken         string `json:"accessToken"`
	TokenType           string `json:"tokenType"`
	ExpiresIn           int64  `json:"expiresIn"`
	ManagementSessionID string `json:"managementSessionId"`
}

// WholeQuantity is a non-negative base-10 integer encoded as a JSON string.
type WholeQuantity string

// DecimalQuantity is a non-negative base-10 decimal encoded as a JSON string.
// It never accepts exponent notation and is also used for cost quota meters.
type DecimalQuantity string

// CurrencyDecimal is a non-negative plain decimal with at most 15 fractional
// digits. Different currencies are never combined into one value.
type CurrencyDecimal string

type CostCompleteness string

const (
	CostComplete CostCompleteness = "complete"
	CostPartial  CostCompleteness = "partial"
	CostUnknown  CostCompleteness = "unknown"
)

type CostSummary struct {
	Currency             string           `json:"currency"`
	KnownAmount          CurrencyDecimal  `json:"knownAmount"`
	Completeness         CostCompleteness `json:"completeness"`
	KnownDispatches      WholeQuantity    `json:"knownDispatches"`
	IncompleteDispatches WholeQuantity    `json:"incompleteDispatches"`
}

type GrantSource struct {
	SubjectType string `json:"subjectType"`
	SubjectID   string `json:"subjectId"`
	BindingID   string `json:"bindingId"`
}

type EffectiveGrant struct {
	ResourceType string      `json:"resourceType"`
	ResourceID   string      `json:"resourceId"`
	Permissions  []string    `json:"permissions"`
	Effect       string      `json:"effect"`
	Source       GrantSource `json:"source"`
}

type EffectiveAccess struct {
	Grants []EffectiveGrant `json:"grants"`
}

type MeterFreshness struct {
	Source string    `json:"source"`
	AsOf   time.Time `json:"asOf"`
}

type QuotaMeter struct {
	PolicyID             string           `json:"policyId"`
	RuleID               string           `json:"ruleId"`
	BindingID            string           `json:"bindingId"`
	Source               GrantSource      `json:"source"`
	CounterOwner         string           `json:"counterOwner"`
	Metric               string           `json:"metric"`
	Algorithm            string           `json:"algorithm"`
	Accounting           string           `json:"accounting"`
	Enforcement          string           `json:"enforcement"`
	Window               string           `json:"window,omitempty"`
	Currency             string           `json:"currency,omitempty"`
	Limit                DecimalQuantity  `json:"limit"`
	Used                 DecimalQuantity  `json:"used"`
	Remaining            *DecimalQuantity `json:"remaining"`
	Overage              *DecimalQuantity `json:"overage,omitempty"`
	ResetAt              *time.Time       `json:"resetAt,omitempty"`
	Completeness         string           `json:"completeness"`
	KnownDispatches      WholeQuantity    `json:"knownDispatches"`
	IncompleteDispatches WholeQuantity    `json:"incompleteDispatches"`
	CapacityState        string           `json:"capacityState"`
	ActiveFenceIDs       []string         `json:"activeFenceIds"`
	Freshness            MeterFreshness   `json:"freshness"`
}

type EffectiveQuota struct {
	Meters             []QuotaMeter `json:"meters"`
	LimitingRuleID     string       `json:"limitingRuleId,omitempty"`
	UnknownUsageFences []string     `json:"unknownUsageFences"`
	AsOf               time.Time    `json:"asOf"`
}

type EffectivePolicy struct {
	Subject         PolicySubject   `json:"subject"`
	Revision        int64           `json:"revision"`
	AppliedRevision int64           `json:"appliedRevision"`
	Access          EffectiveAccess `json:"access"`
	Quota           EffectiveQuota  `json:"quota"`
}
