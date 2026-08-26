package testcases

import (
	"net/http"
	"net/http/httptest"
	"testing"
)

type managedAccessEntrypointConflictFixture struct {
	t           *testing.T
	namespaceID string
	step        int
}

func (fixture *managedAccessEntrypointConflictFixture) ServeHTTP(
	response http.ResponseWriter,
	request *http.Request,
) {
	fixture.t.Helper()
	fixture.step++
	if request.Header.Get(managedAccessNamespaceHeader) != fixture.namespaceID {
		fixture.t.Fatalf("namespace header = %q", request.Header.Get(managedAccessNamespaceHeader))
	}
	switch fixture.step {
	case 1:
		fixture.serveInitialConflict(response, request)
	case 2:
		fixture.serveRevisionRefresh(response, request)
	case 3:
		fixture.serveSuccessfulRetry(response, request)
	default:
		fixture.t.Fatalf("unexpected request %d", fixture.step)
	}
}

func (fixture *managedAccessEntrypointConflictFixture) serveInitialConflict(
	response http.ResponseWriter,
	request *http.Request,
) {
	fixture.t.Helper()
	if request.Method != http.MethodPost || request.Header.Get("If-Match") != `"ep:1"` {
		fixture.t.Fatalf("first request = %s If-Match %q", request.Method, request.Header.Get("If-Match"))
	}
	response.WriteHeader(http.StatusPreconditionFailed)
	_, _ = response.Write([]byte(`{"error":{"code":"revision_conflict"}}`))
}

func (fixture *managedAccessEntrypointConflictFixture) serveRevisionRefresh(
	response http.ResponseWriter,
	request *http.Request,
) {
	fixture.t.Helper()
	if request.Method != http.MethodGet {
		fixture.t.Fatalf("refresh request method = %s", request.Method)
	}
	response.Header().Set("ETag", `"ep:2"`)
	response.WriteHeader(http.StatusOK)
}

func (fixture *managedAccessEntrypointConflictFixture) serveSuccessfulRetry(
	response http.ResponseWriter,
	request *http.Request,
) {
	fixture.t.Helper()
	if request.Method != http.MethodPost || request.Header.Get("If-Match") != `"ep:2"` {
		fixture.t.Fatalf("retry request = %s If-Match %q", request.Method, request.Header.Get("If-Match"))
	}
	response.WriteHeader(http.StatusAccepted)
}

func TestPublishManagedAccessEntrypointRefreshesRevisionAfterConflict(t *testing.T) {
	const (
		namespaceID  = "10000000-0000-4000-8000-000000000001"
		entrypointID = "managed_access_entrypoint"
	)
	fixture := &managedAccessEntrypointConflictFixture{t: t, namespaceID: namespaceID}
	server := httptest.NewServer(fixture)
	defer server.Close()

	client := &managedAccessClient{baseURL: server.URL, client: server.Client()}
	if err := publishManagedAccessEntrypoint(t.Context(), client, namespaceID, entrypointID, 1); err != nil {
		t.Fatal(err)
	}
	if fixture.step != 3 {
		t.Fatalf("request count = %d, want 3", fixture.step)
	}
}

func TestValidManagedAccessEntrypointETag(t *testing.T) {
	for value, want := range map[string]bool{
		`"ep:1"`:    true,
		`"ep:0"`:    false,
		`"ep:x"`:    false,
		`ep:1`:      false,
		`"model:1"`: false,
	} {
		if got := validManagedAccessEntrypointETag(value); got != want {
			t.Errorf("validManagedAccessEntrypointETag(%q) = %v, want %v", value, got, want)
		}
	}
}
