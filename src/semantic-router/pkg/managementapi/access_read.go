package managementapi

import "time"

// RoutingClaimValue is the wire representation of a typed routing-context
// claim. It intentionally has no dependency on the routing snapshot domain.
type RoutingClaimValue struct {
	Kind    string `json:"kind"`
	String  string `json:"string,omitempty"`
	Boolean bool   `json:"boolean,omitempty"`
	Integer int64  `json:"integer,omitempty"`
}

type RoutingContextSource struct {
	SubjectType string `json:"subjectType"`
	SubjectID   string `json:"subjectId"`
}

type RoutingContextStoredValue struct {
	Name      string            `json:"name"`
	Value     RoutingClaimValue `json:"value"`
	Revision  int64             `json:"revision"`
	UpdatedAt time.Time         `json:"updatedAt"`
}

type RoutingContextEffectiveValue struct {
	Name      string               `json:"name"`
	Value     RoutingClaimValue    `json:"value"`
	Source    RoutingContextSource `json:"source"`
	Revision  int64                `json:"revision,omitempty"`
	UpdatedAt *time.Time           `json:"updatedAt,omitempty"`
}

type RoutingContext struct {
	Subject        PolicySubject                  `json:"subject"`
	Revision       int64                          `json:"revision"`
	SchemaRevision int64                          `json:"schemaRevision"`
	Stored         []RoutingContextStoredValue    `json:"stored"`
	Effective      []RoutingContextEffectiveValue `json:"effective"`
}

type RoutingContextPutRequest struct {
	Values map[string]RoutingClaimValue `json:"values"`
}

type AccessCheckResource struct {
	Type string `json:"type"`
	ID   string `json:"id"`
}

type AccessCheckRequest struct {
	Subject                PolicySubject                 `json:"subject"`
	Resource               AccessCheckResource           `json:"resource"`
	Permission             string                        `json:"permission"`
	Path                   string                        `json:"path,omitempty"`
	RoutingContextOverride *map[string]RoutingClaimValue `json:"routingContextOverride,omitempty"`
}

type AccessCheckResponse struct {
	Subject         PolicySubject                  `json:"subject"`
	Resource        AccessCheckResource            `json:"resource"`
	Permission      string                         `json:"permission"`
	Decision        string                         `json:"decision"`
	MatchedGrants   []EffectiveGrant               `json:"matchedGrants"`
	RoutingContext  []RoutingContextEffectiveValue `json:"routingContext"`
	Simulation      bool                           `json:"simulation"`
	Revision        int64                          `json:"revision"`
	AppliedRevision int64                          `json:"appliedRevision"`
}
