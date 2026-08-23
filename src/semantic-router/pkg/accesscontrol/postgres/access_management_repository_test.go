package postgres

import (
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accessmanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingsnapshot"
)

func TestCompileRoutingContextUsesFieldwiseKeyUserTeamPrecedence(t *testing.T) {
	key := accessmanagement.Subject{Kind: accesscontrol.SubjectKindAPIKey, ID: "key-1"}
	user := accessmanagement.Subject{Kind: accesscontrol.SubjectKindUser, ID: "user-1"}
	team := accessmanagement.Subject{Kind: accesscontrol.SubjectKindTeam, ID: "team-1"}
	base := time.Date(2026, 8, 23, 10, 0, 0, 0, time.UTC)
	claims := map[string]map[string]storedRoutingClaim{
		team.ID: {
			"shared": {Value: routingsnapshot.ClaimValue{Kind: "string", String: "team"}, Revision: 1, UpdatedAt: base},
			"team":   {Value: routingsnapshot.ClaimValue{Kind: "boolean", Boolean: true}, Revision: 1, UpdatedAt: base},
		},
		user.ID: {
			"shared": {Value: routingsnapshot.ClaimValue{Kind: "string", String: "user"}, Revision: 2, UpdatedAt: base.Add(time.Minute)},
			"user":   {Value: routingsnapshot.ClaimValue{Kind: "integer", Integer: 2}, Revision: 2, UpdatedAt: base.Add(time.Minute)},
		},
		key.ID: {
			"shared": {Value: routingsnapshot.ClaimValue{Kind: "string", String: "key"}, Revision: 3, UpdatedAt: base.Add(2 * time.Minute)},
		},
	}
	result := compileRoutingContext(key, 9, accessmanagement.RoutingClaimSchema{Revision: 4}, claims,
		accessmanagement.LayerSubjects{Key: &key, User: &user, Team: &team})
	if result.Revision != 9 || result.SchemaRevision != 4 || len(result.Stored) != 1 || len(result.Effective) != 3 {
		t.Fatalf("unexpected routing context shape: %#v", result)
	}
	values := make(map[string]accessmanagement.EffectiveClaim, len(result.Effective))
	for _, claim := range result.Effective {
		values[claim.Name] = claim
	}
	if values["shared"].Value.String != "key" || values["shared"].Source != key ||
		values["user"].Source != user || values["team"].Source != team {
		t.Fatalf("fieldwise precedence was not preserved: %#v", values)
	}
}
