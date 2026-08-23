// Package accessmanagement owns credential-free effective-policy, live quota,
// routing-context, and access-simulation reads for the Router Management API.
package accessmanagement

import (
	"net/netip"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accessprojection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/quotaruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingsnapshot"
)

type Subject struct {
	Kind accesscontrol.SubjectKind
	ID   string
}

func (subject Subject) Validate() error {
	if !subject.Kind.Valid() || subject.ID == "" {
		return ErrInvalidRequest
	}
	return nil
}

type ClaimDefinition struct {
	Kind      string `json:"kind"`
	Minimum   *int64 `json:"minimum,omitempty"`
	Maximum   *int64 `json:"maximum,omitempty"`
	MaxLength *int64 `json:"maxLength,omitempty"`
}

type RoutingClaimSchema struct {
	Revision    uint64
	Definitions map[string]ClaimDefinition
}

type StoredClaim struct {
	Name      string
	Value     routingsnapshot.ClaimValue
	Revision  uint64
	UpdatedAt time.Time
}

type EffectiveClaim struct {
	StoredClaim
	Source Subject
}

type RoutingContext struct {
	Subject        Subject
	Revision       uint64
	SchemaRevision uint64
	Stored         []StoredClaim
	Effective      []EffectiveClaim
}

type LayerSubjects struct {
	Key  *Subject
	User *Subject
	Team *Subject
}

func (subjects LayerSubjects) Source(layer accesscontrol.InheritanceLayer) Subject {
	switch layer {
	case accesscontrol.InheritanceLayerKey:
		if subjects.Key != nil {
			return *subjects.Key
		}
	case accesscontrol.InheritanceLayerUser:
		if subjects.User != nil {
			return *subjects.User
		}
	case accesscontrol.InheritanceLayerTeam:
		if subjects.Team != nil {
			return *subjects.Team
		}
	}
	return Subject{}
}

type PolicySnapshot struct {
	NamespaceID     string
	QuotaPartition  string
	BillingCurrency string
	Subject         Subject
	SubjectRevision uint64
	DesiredRevision uint64
	AppliedRevision uint64
	RevisionTime    time.Time
	Projection      accessprojection.Projection
	LayerSubjects   LayerSubjects
	Schema          RoutingClaimSchema
	Context         RoutingContext
}

type AuthorizationContext struct {
	Subject      Subject
	Ancestors    []Subject
	RateBindings []BindingAuthorizationContext
}

type BindingAuthorizationContext struct {
	BindingID string
	Subject   Subject
}

type GrantView struct {
	Grant  accessprojection.Grant
	Source Subject
}

type QuotaMeterView struct {
	Binding accessprojection.RateBinding
	Rule    accessprojection.ProjectedRateRule
	Source  Subject
	Meter   quotaruntime.Meter
}

type EffectiveQuota struct {
	Meters         []QuotaMeterView
	LimitingRuleID string
	FenceIDs       []string
	AsOf           time.Time
}

type EffectivePolicy struct {
	Subject         Subject
	DesiredRevision uint64
	AppliedRevision uint64
	Access          []GrantView
	Quota           EffectiveQuota
}

type AccessCheckRequest struct {
	NamespaceID     string
	Subject         Subject
	Resource        accesscontrol.GrantResource
	Permission      accesscontrol.GrantPermission
	Path            string
	Override        map[string]routingsnapshot.ClaimValue
	OverridePresent bool
}

type AccessCheckResult struct {
	Subject         Subject
	Resource        accesscontrol.GrantResource
	Permission      accesscontrol.GrantPermission
	Decision        accesscontrol.AccessDecision
	Matched         []GrantView
	RoutingContext  []EffectiveClaim
	Simulation      bool
	DesiredRevision uint64
	AppliedRevision uint64
}

type Actor struct {
	PrincipalID string
	ActorChain  []string
	RequestID   string
	SourceIP    netip.Addr
}

type UpdateRoutingContextRequest struct {
	NamespaceID      string
	Subject          Subject
	ExpectedRevision uint64
	Values           map[string]routingsnapshot.ClaimValue
	Actor            Actor
}

type RoutingContextMutation struct {
	DesiredRevision uint64
	QuotaPartition  string
}
