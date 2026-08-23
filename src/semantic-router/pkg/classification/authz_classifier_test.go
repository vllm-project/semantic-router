package classification

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingsnapshot"
)

func TestAuthzClassifierUsesOnlyTrustedTenantContext(t *testing.T) {
	classifier, err := NewAuthzClassifier([]config.RoleBinding{
		{Name: "user", Role: "user-role", Subjects: []config.Subject{{Kind: "User", Name: "usr-1"}}},
		{Name: "team", Role: "team-role", Subjects: []config.Subject{{Kind: "Team", Name: "team-1"}}},
		{Name: "tier", Role: "premium-role", Subjects: []config.Subject{{Kind: "Group", Name: "premium"}}},
		{Name: "flag", Role: "beta-role", Subjects: []config.Subject{{Kind: "Group", Name: "beta"}}},
	})
	if err != nil {
		t.Fatal(err)
	}
	result, err := classifier.Classify(TrustedRoutingIdentity{
		UserID: "usr-1",
		TeamID: "team-1",
		Claims: map[string]routingsnapshot.ClaimValue{
			"routing_tier": {Kind: "string", String: "premium"},
			"beta":         {Kind: "boolean", Boolean: true},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	want := []string{"user-role", "team-role", "premium-role", "beta-role"}
	if len(result.MatchedRules) != len(want) {
		t.Fatalf("matched roles = %v, want %v", result.MatchedRules, want)
	}
	for index := range want {
		if result.MatchedRules[index] != want[index] {
			t.Fatalf("matched roles = %v, want %v", result.MatchedRules, want)
		}
	}
}

func TestAuthzClassifierFailsClosedWithoutTrustedIdentity(t *testing.T) {
	classifier, err := NewAuthzClassifier([]config.RoleBinding{{
		Name: "private", Role: "private", Subjects: []config.Subject{{Kind: "User", Name: "usr-1"}},
	}})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := classifier.Classify(TrustedRoutingIdentity{}); err == nil {
		t.Fatal("expected missing TenantContext to fail closed")
	}
}

func TestAuthzClassifierRejectsUnknownSubjectKind(t *testing.T) {
	_, err := NewAuthzClassifier([]config.RoleBinding{{
		Name: "bad", Role: "bad", Subjects: []config.Subject{{Kind: "Header", Name: "x-user"}},
	}})
	if err == nil {
		t.Fatal("expected client-header subject kind to be rejected")
	}
}
