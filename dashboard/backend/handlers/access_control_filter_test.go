package handlers

import (
	"net/http/httptest"
	"testing"
)

func TestAccessListFilterUsesExplicitInferenceScopes(t *testing.T) {
	request := httptest.NewRequest(
		"GET",
		"/api/v1/access-control/usage?userId=user-a&teamId=team-a&keyId=key-a&model=vllm-sr%2Fmom-v1-lite&limit=25",
		nil,
	)
	filter := accessListFilter(request)
	if filter.UserID != "user-a" || filter.TeamID != "team-a" || filter.KeyID != "key-a" {
		t.Fatalf("subject filter = %#v", filter)
	}
	if filter.Model != "vllm-sr/mom-v1-lite" || filter.Limit != 25 {
		t.Fatalf("usage filter = %#v", filter)
	}
}
