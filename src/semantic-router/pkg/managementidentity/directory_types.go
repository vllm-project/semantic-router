package managementidentity

import "github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"

// PrincipalDirectoryEntry is the namespace-safe projection used to select a
// Management principal for a User link. Authentication subjects, attributes,
// sessions, role bindings, and links from other namespaces are deliberately
// absent.
type PrincipalDirectoryEntry struct {
	PrincipalID   accesscontrol.ManagementPrincipalID
	DisplayName   string
	VerifiedEmail string
	Status        accesscontrol.PrincipalStatus
	UserID        accesscontrol.UserID
	LinkRevision  accesscontrol.Revision
}

func (entry PrincipalDirectoryEntry) Linked() bool {
	return entry.UserID != "" && entry.LinkRevision > 0
}

type PrincipalDirectoryRequest struct {
	NamespaceID string
	Search      string
	AfterID     string
	Limit       int
}

type PrincipalDirectoryPage struct {
	Items      []PrincipalDirectoryEntry
	NextCursor string
}

type PrincipalUserLinkListRequest struct {
	NamespaceID string
	PrincipalID string
	UserID      string
	AfterID     string
	Limit       int
}

type PrincipalUserLinkPage struct {
	Items      []PrincipalUserLink
	NextCursor string
}
