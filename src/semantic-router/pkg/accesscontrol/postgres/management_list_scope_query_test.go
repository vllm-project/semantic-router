package postgres

import (
	"strings"
	"testing"
)

func TestManagementListQueriesApplyAuthorizationBeforeKeysetAndLimit(t *testing.T) {
	tests := map[string]struct {
		query  string
		scope  string
		cursor string
	}{
		"users":                {subjectListUsersQuery, "id = ANY($4::uuid[])", "$5::timestamptz IS NULL"},
		"teams":                {subjectListTeamsQuery, "id = ANY($4::uuid[])", "$5::timestamptz IS NULL"},
		"user memberships":     {subjectListUserMembershipsQuery, "m.team_id = ANY($5::uuid[])", "$6::timestamptz IS NULL"},
		"team members":         {subjectListTeamMembersQuery, "m.user_id = ANY($5::uuid[])", "$6::timestamptz IS NULL"},
		"API keys":             {managementListAPIKeysQuery, "id = ANY($6::uuid[])", "$9::timestamptz IS NULL"},
		"access policies":      {listManagedAccessPoliciesQuery, "p.id = ANY($4::uuid[])", "$5::timestamptz IS NULL"},
		"rate policies":        {listManagedRatePoliciesQuery, "p.id = ANY($4::uuid[])", "$5::timestamptz IS NULL"},
		"access bindings":      {listManagedAccessBindingsQuery, "b.policy_id=ANY($7::uuid[])", "$8::timestamptz IS NULL"},
		"rate bindings":        {listManagedRateBindingsQuery, "b.policy_id=ANY($8::uuid[])", "$9::timestamptz IS NULL"},
		"provider credentials": {listProviderCredentialsQuery, "id = ANY($5::uuid[])", "$6 = ''"},
	}
	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			scope := strings.Index(test.query, test.scope)
			cursor := strings.Index(test.query, test.cursor)
			order := strings.Index(test.query, "ORDER BY")
			limit := strings.Index(test.query, "LIMIT")
			if scope < 0 || cursor < 0 || order < 0 || limit < 0 || (scope >= cursor || cursor >= order || order >= limit) {
				t.Fatalf("authorization/keyset/order/limit sequence is invalid: %d/%d/%d/%d", scope, cursor, order, limit)
			}
		})
	}
}

func TestManagementSearchQueriesApplyScopeBeforeIndexedSearchAndKeyset(t *testing.T) {
	tests := map[string]struct {
		query  string
		scope  string
		search string
		cursor string
	}{
		"users":           {subjectSearchUsersQuery, "id = ANY($4::uuid[])", "lower(email) LIKE $5", "$6::timestamptz IS NULL"},
		"teams":           {subjectSearchTeamsQuery, "id = ANY($4::uuid[])", "lower(name) LIKE $5", "$6::timestamptz IS NULL"},
		"API keys":        {managementSearchAPIKeysQuery, "id = ANY($6::uuid[])", "lower(name) LIKE $9", "$10::timestamptz IS NULL"},
		"access policies": {searchManagedAccessPoliciesQuery, "p.id = ANY($4::uuid[])", "lower(p.name) LIKE $5", "$6::timestamptz IS NULL"},
		"rate policies":   {searchManagedRatePoliciesQuery, "p.id = ANY($4::uuid[])", "lower(p.name) LIKE $5", "$6::timestamptz IS NULL"},
	}
	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			scope := strings.Index(test.query, test.scope)
			search := strings.Index(test.query, test.search)
			cursor := strings.Index(test.query, test.cursor)
			limit := strings.Index(test.query, "LIMIT")
			if scope < 0 || search < 0 || cursor < 0 || limit < 0 ||
				(scope >= search || search >= cursor || cursor >= limit) {
				t.Fatalf("authorization/search/keyset/limit sequence is invalid: %d/%d/%d/%d", scope, search, cursor, limit)
			}
		})
	}
	if strings.Contains(managementSearchAPIKeysQuery, "access_api_key_credentials") ||
		strings.Contains(managementSearchAPIKeysQuery, "secret_") ||
		strings.Contains(managementSearchAPIKeysQuery, "kid") {
		t.Fatal("API-key search must not inspect credential material")
	}
}

func TestEligibleKeySearchRemainsInsideSelfEligibilityAndCursorBoundary(t *testing.T) {
	eligibility := strings.Index(eligibleKeyList, "k.owner_user_id = l.user_id")
	search := strings.Index(eligibleKeyList, "lower(k.name) LIKE $3")
	cursor := strings.Index(eligibleKeyList, "$4::timestamptz IS NULL")
	limit := strings.Index(eligibleKeyList, "LIMIT $6")
	if eligibility < 0 || search < 0 || cursor < 0 || limit < 0 ||
		eligibility >= search || search >= cursor || cursor >= limit {
		t.Fatalf("eligibility/search/keyset/limit sequence is invalid: %d/%d/%d/%d",
			eligibility, search, cursor, limit)
	}
	if strings.Contains(eligibleKeyList, "access_api_key_credentials") ||
		strings.Contains(eligibleKeyList, "secret_") || strings.Contains(eligibleKeyList, "kid") {
		t.Fatal("self key search must not inspect credential material")
	}
}

func TestManagedBindingQueriesNormalizeOptionalUUIDFilters(t *testing.T) {
	for name, query := range map[string]string{
		"access bindings": listManagedAccessBindingsQuery,
		"rate bindings":   listManagedRateBindingsQuery,
	} {
		t.Run(name, func(t *testing.T) {
			if !strings.Contains(query, "policy_id=NULLIF($2,'')::uuid") ||
				!strings.Contains(query, "NULLIF($4,'')::uuid IS NULL") ||
				!strings.Contains(query, "subject_id=NULLIF($4,'')::uuid") {
				t.Fatal("optional UUID filters are not normalized before PostgreSQL casts")
			}
		})
	}
}

func TestAuthoritativeRelationshipCountsKeepNamespaceAndVisibilityPredicates(t *testing.T) {
	for name, test := range map[string]struct {
		query string
		scope string
	}{
		"user memberships":   {subjectCountUserMembershipsQuery, "m.team_id = ANY($5::uuid[])"},
		"team members":       {subjectCountTeamMembersQuery, "m.user_id = ANY($5::uuid[])"},
		"API keys":           {managementCountAPIKeysQuery, "id = ANY($6::uuid[])"},
		"access assignments": {countFilteredManagedAccessBindingsQuery, "b.policy_id=ANY($7::uuid[])"},
		"budget assignments": {countFilteredManagedRateBindingsQuery, "b.policy_id=ANY($8::uuid[])"},
	} {
		t.Run(name, func(t *testing.T) {
			if !strings.Contains(test.query, "namespace_id") || !strings.Contains(test.query, "$1") ||
				!strings.Contains(test.query, test.scope) {
				t.Fatalf("count query lost namespace or permission scope: %s", test.query)
			}
		})
	}
}
