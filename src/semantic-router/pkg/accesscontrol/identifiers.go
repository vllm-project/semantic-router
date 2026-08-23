package accesscontrol

// Domain identifiers are distinct types so ownership and binding code cannot
// accidentally exchange unrelated IDs. Persistence adapters remain responsible
// for parsing their concrete UUID representation.
type (
	NamespaceID             string
	QuotaPartitionID        string
	SubjectID               string
	UserID                  string
	TeamID                  string
	APIKeyID                string
	CredentialVersionID     string
	AccessPolicyID          string
	RateLimitPolicyID       string
	RateLimitRuleID         string
	PolicyBindingID         string
	ManagementPrincipalID   string
	ManagementRoleID        string
	ManagementRoleBindingID string
	ResourceID              string
	Revision                uint64
)
