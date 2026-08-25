package postgres

import (
	"strings"
	"testing"
)

func TestBackchannelLogoutSelectorPrefersIssuerSessionID(t *testing.T) {
	plan := backchannelLogoutPlanFor(
		"issuer-id", "https://issuer.example", "issuer-session", "subject",
	)
	if !strings.Contains(plan.expireQuery, "issuer_session_id=$2") || strings.Contains(plan.expireQuery, "management_principals") ||
		len(plan.arguments) != 2 || plan.arguments[0] != "issuer-id" || plan.arguments[1] != "issuer-session" {
		t.Fatalf("query=%q arguments=%v", plan.expireQuery, plan.arguments)
	}
}

func TestBackchannelLogoutSelectorUsesIssuerAndSubjectWithoutSID(t *testing.T) {
	plan := backchannelLogoutPlanFor(
		"issuer-id", "https://issuer.example", "", "subject",
	)
	if !strings.Contains(plan.expireQuery, "management_principals") || !strings.Contains(plan.expireQuery, "issuer=$2") ||
		!strings.Contains(plan.expireQuery, "subject=$3") || len(plan.arguments) != 3 ||
		plan.arguments[0] != "issuer-id" || plan.arguments[1] != "https://issuer.example" || plan.arguments[2] != "subject" {
		t.Fatalf("query=%q arguments=%v", plan.expireQuery, plan.arguments)
	}
}
