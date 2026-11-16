package connector

import (
	"context"
	"net/http"
	"net/http/httptest"
	"net/url"
	"testing"
)

// TestDoRequestPreservesEscapedBasePathSegments covers issue #6150: a base URL
// is operator supplied, so a percent-encoded path segment such as
// tenant%2Fblue names one segment. Appending an operation path must keep it
// escaped; decoding it would turn one segment into two and route the call
// somewhere else.
func TestDoRequestPreservesEscapedBasePathSegments(t *testing.T) {
	tests := []struct {
		name       string
		baseSuffix string
		opPath     string
		wantPath   string
		wantURI    string
	}{
		{
			name:       "encoded slash in the base path",
			baseSuffix: "/gateway/tenant%2Fblue",
			opPath:     "/v1/models",
			wantPath:   "/gateway/tenant%2Fblue/v1/models",
			wantURI:    "/gateway/tenant%2Fblue/v1/models",
		},
		{
			name:       "encoded slash with a multi-segment operation",
			baseSuffix: "/gateway/tenant%2Fblue",
			opPath:     "/v1/chat/completions",
			wantPath:   "/gateway/tenant%2Fblue/v1/chat/completions",
			wantURI:    "/gateway/tenant%2Fblue/v1/chat/completions",
		},
		{
			name:       "ordinary base path is unchanged",
			baseSuffix: "/models",
			opPath:     "/classify",
			wantPath:   "/models/classify",
			wantURI:    "/models/classify",
		},
		{
			name:       "trailing slash on the base path is unchanged",
			baseSuffix: "/models/",
			opPath:     "/classify",
			wantPath:   "/models/classify",
			wantURI:    "/models/classify",
		},
		{
			name:       "encoded space in the base path stays encoded",
			baseSuffix: "/team%20blue",
			opPath:     "/v1/models",
			wantPath:   "/team%20blue/v1/models",
			wantURI:    "/team%20blue/v1/models",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			var gotPath, gotURI string
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				gotPath, gotURI = r.URL.EscapedPath(), r.RequestURI
				_, _ = w.Write([]byte(`{}`))
			}))
			defer server.Close()

			client, err := New(server.URL+test.baseSuffix, nil, testOptions())
			if err != nil {
				t.Fatalf("New() error = %v", err)
			}
			t.Cleanup(func() { _ = client.Close() })

			operation := testOperation
			operation.Path = test.opPath
			if _, err := client.Do(context.Background(), operation, nil); err != nil {
				t.Fatalf("Do() error = %v", err)
			}
			if gotPath != test.wantPath {
				t.Fatalf("outgoing EscapedPath() = %q, want %q", gotPath, test.wantPath)
			}
			if gotURI != test.wantURI {
				t.Fatalf("outgoing RequestURI = %q, want %q", gotURI, test.wantURI)
			}
		})
	}
}

// TestJoinPathKeepsRawPathConsistent pins the invariant behind the test above:
// RawPath carries the escaped path only when the base URL actually escaped
// something, so Path and RawPath never disagree and an ordinary base path
// keeps its previous, empty-RawPath representation.
func TestJoinPathKeepsRawPathConsistent(t *testing.T) {
	tests := []struct {
		name        string
		baseURL     string
		operation   string
		wantPath    string
		wantRawPath string
	}{
		{
			name:        "encoded base segment",
			baseURL:     "https://provider.example/gateway/tenant%2Fblue",
			operation:   "/v1/models",
			wantPath:    "/gateway/tenant/blue/v1/models",
			wantRawPath: "/gateway/tenant%2Fblue/v1/models",
		},
		{
			name:        "ordinary base path",
			baseURL:     "https://provider.example/models",
			operation:   "/classify",
			wantPath:    "/models/classify",
			wantRawPath: "",
		},
		{
			name:        "base URL without a path",
			baseURL:     "https://provider.example",
			operation:   "/v1/models",
			wantPath:    "/v1/models",
			wantRawPath: "",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			base, err := url.Parse(test.baseURL)
			if err != nil {
				t.Fatalf("url.Parse() error = %v", err)
			}
			decoded, escaped := joinPath(base, test.operation)
			if decoded != test.wantPath || escaped != test.wantRawPath {
				t.Fatalf("joinPath() = (%q, %q), want (%q, %q)", decoded, escaped, test.wantPath, test.wantRawPath)
			}
			wantEmitted := test.wantRawPath
			if wantEmitted == "" {
				wantEmitted = test.wantPath
			}
			joined := &url.URL{Path: decoded, RawPath: escaped}
			if got := joined.EscapedPath(); got != wantEmitted {
				t.Fatalf("EscapedPath() = %q, want %q", got, wantEmitted)
			}
		})
	}
}
