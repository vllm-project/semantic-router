package postgres

import (
	"fmt"

	"github.com/google/uuid"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
)

var membershipEventNamespace = uuid.NewSHA1(
	uuid.NameSpaceOID,
	[]byte("vllm.semantic-router.accesscontrol.team-membership"),
)

// membershipEventAggregateID is an internal UUIDv5 adapter for event tables
// whose aggregate_id column is UUID. TeamMembership remains a composite domain
// resource and this identifier is never returned as its public identity.
func membershipEventAggregateID(membership accesscontrol.TeamMembership) string {
	return uuid.NewSHA1(
		membershipEventNamespace,
		[]byte(membershipResourceReference(membership)),
	).String()
}

func membershipResourceReference(membership accesscontrol.TeamMembership) string {
	return fmt.Sprintf(
		"namespaces/%s/teams/%s/members/%s",
		membership.NamespaceID,
		membership.TeamID,
		membership.UserID,
	)
}
