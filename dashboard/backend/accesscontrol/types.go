package accesscontrol

import "time"

const (
	StatusActive   = "active"
	StatusDisabled = "disabled"
	TeamRoleAdmin  = "admin"
	TeamRoleMember = "member"
)

type TeamMembership struct {
	TeamID string `json:"teamId"`
	UserID string `json:"userId"`
	Role   string `json:"role"`
}

type TeamMemberIdentity struct {
	ID     string `json:"id"`
	Email  string `json:"email"`
	Name   string `json:"name"`
	Status string `json:"status"`
}

type SelfTeamCatalog struct {
	Teams        []Team               `json:"items"`
	Members      []TeamMemberIdentity `json:"members"`
	AccessGroups []AccessGroup        `json:"accessGroups"`
	Budgets      []Budget             `json:"budgets"`
}

type User struct {
	ID             string           `json:"id"`
	Email          string           `json:"email"`
	Name           string           `json:"name"`
	Status         string           `json:"status"`
	AccessGroupIDs []string         `json:"accessGroupIds"`
	BudgetID       string           `json:"budgetId,omitempty"`
	Memberships    []TeamMembership `json:"memberships"`
	CreatedAt      time.Time        `json:"createdAt"`
	UpdatedAt      time.Time        `json:"updatedAt"`
}

type Team struct {
	ID             string           `json:"id"`
	Name           string           `json:"name"`
	Description    string           `json:"description"`
	Status         string           `json:"status"`
	Members        []TeamMembership `json:"members"`
	AccessGroupIDs []string         `json:"accessGroupIds"`
	BudgetID       string           `json:"budgetId"`
	CreatedAt      time.Time        `json:"createdAt"`
	UpdatedAt      time.Time        `json:"updatedAt"`
}

type APIKey struct {
	ID                 string               `json:"id"`
	Name               string               `json:"name"`
	Prefix             string               `json:"prefix"`
	UserID             string               `json:"-"`
	TeamID             string               `json:"-"`
	ContextTeamID      string               `json:"contextTeamId,omitempty"`
	OwnerType          string               `json:"ownerType"`
	OwnerID            string               `json:"ownerId"`
	BudgetID           string               `json:"budgetId,omitempty"`
	Status             string               `json:"status"`
	ExpiresAt          *time.Time           `json:"expiresAt,omitempty"`
	LastUsed           *time.Time           `json:"lastUsedAt,omitempty"`
	AccessGroupIDs     []string             `json:"accessGroupIds"`
	ModelPatterns      []string             `json:"modelPatterns,omitempty"`
	ModelPolicySource  string               `json:"modelPolicySource,omitempty"`
	EffectiveBudgetID  string               `json:"effectiveBudgetId,omitempty"`
	BudgetPolicySource string               `json:"budgetPolicySource,omitempty"`
	Quota              *APIKeyQuotaSnapshot `json:"quota,omitempty"`
	CreatedAt          time.Time            `json:"createdAt"`
	UpdatedAt          time.Time            `json:"updatedAt"`
}

type QuotaMeter struct {
	Limit     int64     `json:"limit"`
	Used      int64     `json:"used"`
	Remaining int64     `json:"remaining"`
	ResetsAt  time.Time `json:"resetsAt"`
}

type APIKeyQuotaSnapshot struct {
	BudgetID    string     `json:"budgetId"`
	BudgetName  string     `json:"budgetName"`
	Source      string     `json:"source"`
	RPM         QuotaMeter `json:"rpm"`
	TPM         QuotaMeter `json:"tpm"`
	DailyTokens QuotaMeter `json:"dailyTokens"`
}

type CreatedAPIKey struct {
	APIKey
	Secret string `json:"secret"`
}

type AccessGroup struct {
	ID              string    `json:"id"`
	Name            string    `json:"name"`
	Description     string    `json:"description"`
	ModelPatterns   []string  `json:"modelPatterns"`
	AssignmentCount int64     `json:"assignmentCount"`
	CreatedAt       time.Time `json:"createdAt"`
	UpdatedAt       time.Time `json:"updatedAt"`
}

type Budget struct {
	ID              string    `json:"id"`
	Name            string    `json:"name"`
	Description     string    `json:"description"`
	RPM             int64     `json:"rpm"`
	TPM             int64     `json:"tpm"`
	DailyTokens     int64     `json:"dailyTokens"`
	Enabled         bool      `json:"enabled"`
	AssignmentCount int64     `json:"assignmentCount"`
	CreatedAt       time.Time `json:"createdAt"`
	UpdatedAt       time.Time `json:"updatedAt"`
}

type Principal struct {
	Key           APIKey
	User          *User
	Team          *Team
	ModelPatterns []string
	Budgets       []Budget
}

type UsageEvent struct {
	ID               string         `json:"id"`
	RequestID        string         `json:"requestId"`
	KeyID            string         `json:"keyId"`
	UserID           string         `json:"userId,omitempty"`
	TeamID           string         `json:"teamId,omitempty"`
	Model            string         `json:"model"`
	StatusCode       int            `json:"statusCode"`
	PromptTokens     int64          `json:"promptTokens"`
	CompletionTokens int64          `json:"completionTokens"`
	TotalTokens      int64          `json:"totalTokens"`
	LatencyMS        int64          `json:"latencyMs"`
	TTFTMS           int64          `json:"ttftMs,omitempty"`
	ErrorCode        string         `json:"errorCode,omitempty"`
	Metadata         map[string]any `json:"metadata,omitempty"`
	CreatedAt        time.Time      `json:"createdAt"`
}

type UsageSummary struct {
	Granularity      string       `json:"granularity"`
	Requests         int64        `json:"requests"`
	Successful       int64        `json:"successful"`
	Failed           int64        `json:"failed"`
	PromptTokens     int64        `json:"promptTokens"`
	CompletionTokens int64        `json:"completionTokens"`
	TotalTokens      int64        `json:"totalTokens"`
	ActiveKeys       int64        `json:"activeKeys"`
	AverageLatencyMS int64        `json:"averageLatencyMs"`
	P95LatencyMS     int64        `json:"p95LatencyMs"`
	AverageTTFTMS    int64        `json:"averageTtftMs"`
	P95TTFTMS        int64        `json:"p95TtftMs"`
	Series           []UsagePoint `json:"series"`
	ByModel          []UsageSlice `json:"byModel"`
	ByUser           []UsageSlice `json:"byUser"`
	ByTeam           []UsageSlice `json:"byTeam"`
	ByKey            []UsageSlice `json:"byKey"`
}

type UsagePoint struct {
	Bucket           time.Time `json:"bucket"`
	Requests         int64     `json:"requests"`
	Successful       int64     `json:"successful"`
	Failed           int64     `json:"failed"`
	PromptTokens     int64     `json:"promptTokens"`
	CompletionTokens int64     `json:"completionTokens"`
	TotalTokens      int64     `json:"totalTokens"`
	AverageLatencyMS int64     `json:"averageLatencyMs"`
}

type UsageSlice struct {
	ID               string `json:"id"`
	Requests         int64  `json:"requests"`
	Successful       int64  `json:"successful"`
	Failed           int64  `json:"failed"`
	PromptTokens     int64  `json:"promptTokens"`
	CompletionTokens int64  `json:"completionTokens"`
	TotalTokens      int64  `json:"totalTokens"`
	AverageLatencyMS int64  `json:"averageLatencyMs"`
	P95LatencyMS     int64  `json:"p95LatencyMs"`
}

type AuditEvent struct {
	ID           string         `json:"id"`
	ActorID      string         `json:"actorId,omitempty"`
	ActorEmail   string         `json:"actorEmail,omitempty"`
	Action       string         `json:"action"`
	ResourceType string         `json:"resourceType"`
	ResourceID   string         `json:"resourceId,omitempty"`
	Details      map[string]any `json:"details,omitempty"`
	CreatedAt    time.Time      `json:"createdAt"`
}

type Overview struct {
	Users           int64 `json:"users"`
	Teams           int64 `json:"teams"`
	ActiveKeys      int64 `json:"activeKeys"`
	ExpiringKeys    int64 `json:"expiringKeys"`
	AccessGroups    int64 `json:"accessGroups"`
	EnabledBudgets  int64 `json:"enabledBudgets"`
	RequestsToday   int64 `json:"requestsToday"`
	SuccessfulToday int64 `json:"successfulToday"`
	TokensToday     int64 `json:"tokensToday"`
	P95LatencyMS    int64 `json:"p95LatencyMs"`
}

type ListFilter struct {
	Limit                 int
	Offset                int
	Query                 string
	UserID                string
	TeamID                string
	KeyID                 string
	Model                 string
	Granularity           string
	TimezoneOffsetMinutes int
	From                  *time.Time
	To                    *time.Time
}
