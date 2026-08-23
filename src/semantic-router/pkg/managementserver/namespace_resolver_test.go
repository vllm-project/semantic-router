package managementserver

import (
	"context"
	"errors"
	"net/http/httptest"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementapi"
)

const resolverNamespaceID = "11111111-1111-4111-8111-111111111111"

func TestExplicitNamespaceResolverRequiresOneCanonicalSelector(t *testing.T) {
	resolver := ExplicitNamespaceResolver{}
	tests := []struct {
		name      string
		path      string
		headers   []string
		want      string
		wantError error
	}{
		{name: "header", path: managementapi.BasePath + "/providers", headers: []string{resolverNamespaceID}, want: resolverNamespaceID},
		{name: "path", path: managementapi.BasePath + "/namespaces/" + resolverNamespaceID + "/routing/snapshots", want: resolverNamespaceID},
		{name: "matching selectors", path: managementapi.BasePath + "/namespaces/" + resolverNamespaceID, headers: []string{resolverNamespaceID}, want: resolverNamespaceID},
		{name: "missing", path: managementapi.BasePath + "/providers", wantError: ErrNamespaceRequired},
		{name: "noncanonical", path: managementapi.BasePath + "/providers", headers: []string{"{11111111-1111-4111-8111-111111111111}"}, wantError: ErrNamespaceRequired},
		{name: "duplicate", path: managementapi.BasePath + "/providers", headers: []string{resolverNamespaceID, resolverNamespaceID}, wantError: ErrNamespaceRequired},
		{name: "conflict", path: managementapi.BasePath + "/namespaces/" + resolverNamespaceID, headers: []string{"22222222-2222-4222-8222-222222222222"}, wantError: ErrNamespaceConflict},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			request := httptest.NewRequest("GET", test.path, nil)
			for _, value := range test.headers {
				request.Header.Add(managementapi.HeaderNamespaceID, value)
			}
			got, err := resolver.ResolveNamespace(context.Background(), request)
			if !errors.Is(err, test.wantError) {
				t.Fatalf("error = %v, want %v", err, test.wantError)
			}
			if got != test.want {
				t.Fatalf("namespace = %q, want %q", got, test.want)
			}
		})
	}
}
