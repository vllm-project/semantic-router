package postgres

import (
	"strings"
	"testing"
)

func TestBackchannelLogoutSelectorPrefersIssuerSessionID(t *testing.T) {
	query, arguments := backchannelLogoutSelector(
		"issuer-id", "https://issuer.example", "issuer-session", "subject",
	)
	if !strings.Contains(query, "issuer_session_id=$2") || strings.Contains(query, "management_principals") ||
		len(arguments) != 2 || arguments[0] != "issuer-id" || arguments[1] != "issuer-session" {
		t.Fatalf("selector=%q arguments=%v", query, arguments)
	}
}

func TestBackchannelLogoutSelectorUsesIssuerAndSubjectWithoutSID(t *testing.T) {
	query, arguments := backchannelLogoutSelector(
		"issuer-id", "https://issuer.example", "", "subject",
	)
	if !strings.Contains(query, "management_principals") || !strings.Contains(query, "issuer=$2") ||
		!strings.Contains(query, "subject=$3") || len(arguments) != 3 ||
		arguments[0] != "issuer-id" || arguments[1] != "https://issuer.example" || arguments[2] != "subject" {
		t.Fatalf("selector=%q arguments=%v", query, arguments)
	}
}
