package managementserver

import (
	"net/http/httptest"
	"testing"
)

func TestParseAgentToolListQueryPreservesBoundedSearchAndKeyset(t *testing.T) {
	request := httptest.NewRequest("GET", "https://management.local/management/v1/agent-tools?pageSize=17&cursor=opaque&search=router.read", nil)
	parsed, err := parseAgentToolListQuery(request)
	if err != nil {
		t.Fatal(err)
	}
	if parsed.PageSize != 17 || parsed.Cursor != "opaque" || parsed.Search != "router.read" {
		t.Fatalf("parsed query = %+v", parsed)
	}

	duplicate := httptest.NewRequest("GET", "https://management.local/management/v1/agent-tools?search=one&search=two", nil)
	if _, err := parseAgentToolListQuery(duplicate); err == nil {
		t.Fatal("duplicate search was accepted")
	}
	unknown := httptest.NewRequest("GET", "https://management.local/management/v1/agent-tools?offset=1", nil)
	if _, err := parseAgentToolListQuery(unknown); err == nil {
		t.Fatal("offset pagination was accepted")
	}
}

func TestParseAgentResourceSearchRejectsSearchOnNonSearchableHistory(t *testing.T) {
	searchable := httptest.NewRequest(
		"GET", "https://management.local/management/v1/agent-profiles?pageSize=17&cursor=opaque&search=recipe", nil,
	)
	parsed, err := parseAgentSearchListQuery(searchable)
	if err != nil {
		t.Fatal(err)
	}
	if parsed.PageSize != 17 || parsed.Cursor != "opaque" || parsed.Search != "recipe" {
		t.Fatalf("parsed query = %+v", parsed)
	}

	history := httptest.NewRequest(
		"GET", "https://management.local/management/v1/agent-sessions/session/turns?search=hidden", nil,
	)
	if _, err := parseAgentListQuery(history); err == nil {
		t.Fatal("turn history accepted an undocumented search filter")
	}
}
