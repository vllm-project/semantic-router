package accesscontrol

import "time"

// ManagementPrincipal is a global immutable issuer/subject identity. Namespace
// authority is supplied only by scoped role bindings and explicit User links.
type ManagementPrincipal struct {
	ID         ManagementPrincipalID
	Issuer     string
	Subject    string
	Status     PrincipalStatus
	Attributes map[string]string
	CreatedAt  time.Time
	UpdatedAt  time.Time
}

func (p ManagementPrincipal) Validate() error {
	var statusErr error
	if !p.Status.Valid() {
		statusErr = invalid("status", "is not a valid principal status")
	}
	return joinValidation(
		validateRequired("id", string(p.ID)),
		validateRequired("issuer", p.Issuer),
		validateRequired("subject", p.Subject),
		statusErr,
		validateTimestamps(p.CreatedAt, p.UpdatedAt),
	)
}
