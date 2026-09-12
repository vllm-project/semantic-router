package authz

// TrustedIdentity is the request identity snapshot established at the ingress
// boundary. UserID, Groups, TenantID, and TeamID are populated only from
// headers accepted from the configured external authorization boundary;
// downstream consumers must not reconstruct identity from raw request
// headers, protocol metadata, or query parameters.
//
// SessionID and ConversationID are included because Router Learning uses the
// same ingress boundary for its client-declared continuity identifiers. They
// are not authentication claims and must not be treated as proof of access.
type TrustedIdentity struct {
	UserID         string
	Groups         []string
	TenantID       string
	TeamID         string
	SessionID      string
	ConversationID string
}

// Clone returns an independent identity snapshot suitable for copying into
// request-scoped or retained state.
func (i TrustedIdentity) Clone() TrustedIdentity {
	i.Groups = append([]string(nil), i.Groups...)
	return i
}
