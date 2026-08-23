package testcases

import (
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestPublishManagedAccessEntrypointRefreshesRevisionAfterConflict(t *testing.T) {
	const (
		namespaceID  = "10000000-0000-4000-8000-000000000001"
		entrypointID = "managed_access_entrypoint"
	)
	step := 0
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		step++
		if request.Header.Get(managedAccessNamespaceHeader) != namespaceID {
			t.Fatalf("namespace header = %q", request.Header.Get(managedAccessNamespaceHeader))
		}
		switch step {
		case 1:
			if request.Method != http.MethodPost || request.Header.Get("If-Match") != `"ep:1"` {
				t.Fatalf("first request = %s If-Match %q", request.Method, request.Header.Get("If-Match"))
			}
			response.WriteHeader(http.StatusPreconditionFailed)
			_, _ = response.Write([]byte(`{"error":{"code":"revision_conflict"}}`))
		case 2:
			if request.Method != http.MethodGet {
				t.Fatalf("refresh request method = %s", request.Method)
			}
			response.Header().Set("ETag", `"ep:2"`)
			response.WriteHeader(http.StatusOK)
		case 3:
			if request.Method != http.MethodPost || request.Header.Get("If-Match") != `"ep:2"` {
				t.Fatalf("retry request = %s If-Match %q", request.Method, request.Header.Get("If-Match"))
			}
			response.WriteHeader(http.StatusAccepted)
		default:
			t.Fatalf("unexpected request %d", step)
		}
	}))
	defer server.Close()

	client := &managedAccessClient{baseURL: server.URL, client: server.Client()}
	if err := publishManagedAccessEntrypoint(t.Context(), client, namespaceID, entrypointID, 1); err != nil {
		t.Fatal(err)
	}
	if step != 3 {
		t.Fatalf("request count = %d, want 3", step)
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
