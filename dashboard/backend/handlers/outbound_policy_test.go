package handlers

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/netip"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/dashboard/backend/safefetch"
)

// inwardResolver answers every name with an internal address, standing in for
// a caller who supplies a public-looking host that points at the cluster.
type inwardResolver struct{ address string }

func (r inwardResolver) LookupNetIP(context.Context, string, string) ([]netip.Addr, error) {
	return []netip.Addr{netip.MustParseAddr(r.address)}, nil
}

func withInwardResolver(t *testing.T, address string) {
	t.Helper()
	previous := outboundResolver
	outboundResolver = inwardResolver{address: address}
	t.Cleanup(func() { outboundResolver = previous })
}

// The address classes an SSRF against this dashboard would aim at.
var inwardTargets = map[string]string{
	"loopback":       "127.0.0.1",
	"private":        "10.0.0.1",
	"link-local":     "169.254.1.1",
	"cloud metadata": "169.254.169.254",
	"IPv6 loopback":  "::1",
}

// FetchRaw must refuse an inward destination and must not disclose why.
func TestFetchRawRefusesInwardDestinations(t *testing.T) {
	for name, address := range inwardTargets {
		t.Run(name, func(t *testing.T) {
			withInwardResolver(t, address)

			body, _ := json.Marshal(FetchRawRequest{URL: "https://public-looking-name.invalid/config.yaml"})
			recorder := httptest.NewRecorder()
			FetchRawHandler()(recorder, httptest.NewRequest(http.MethodPost, "/api/fetch-raw", bytes.NewReader(body)))

			if recorder.Code != http.StatusBadRequest {
				t.Fatalf("status = %d, want 400: %s", recorder.Code, recorder.Body.String())
			}

			var response FetchRawResponse
			if err := json.Unmarshal(recorder.Body.Bytes(), &response); err != nil {
				t.Fatalf("decode response: %v", err)
			}
			if response.Content != "" {
				t.Error("a refused fetch returned content")
			}
			assertNoNetworkDisclosure(t, response.Error, address)
		})
	}
}

// OpenWeb must refuse the same destinations, and must not retry a refused URL
// through the reader fallback.
func TestOpenWebRefusesInwardDestinations(t *testing.T) {
	for name, address := range inwardTargets {
		t.Run(name, func(t *testing.T) {
			withInwardResolver(t, address)

			body, _ := json.Marshal(OpenWebRequest{URL: "https://public-looking-name.invalid/"})
			recorder := httptest.NewRecorder()
			OpenWebHandler()(recorder, httptest.NewRequest(http.MethodPost, "/api/openweb", bytes.NewReader(body)))

			if recorder.Code == http.StatusOK {
				t.Fatalf("a refused destination returned 200: %s", recorder.Body.String())
			}

			var response OpenWebResponse
			if err := json.Unmarshal(recorder.Body.Bytes(), &response); err != nil {
				t.Fatalf("decode response: %v", err)
			}
			if response.Content != "" {
				t.Error("a refused fetch returned content")
			}
			assertNoNetworkDisclosure(t, response.Error, address)
		})
	}
}

// A refused URL fails closed rather than being retried through Jina, which
// would make the policy advisory.
func TestOpenWebDoesNotFallBackAfterAPolicyRefusal(t *testing.T) {
	withInwardResolver(t, "127.0.0.1")

	plan := buildOpenWebFetchPlan(OpenWebRequest{URL: "https://public-looking-name.invalid/"})
	if plan.forceJina {
		t.Fatal("the fixture should take the direct path first")
	}

	result, err := fetchOpenWeb(plan)
	if err == nil {
		t.Fatalf("a refused destination succeeded: %+v", result)
	}
	if err != errOpenWebForbiddenTarget {
		t.Errorf("error = %v, want the terminal policy refusal", err)
	}
}

// Both endpoints reject a scheme that never belongs to a web fetch, before any
// resolution happens.
func TestHandlersRejectNonHTTPSchemes(t *testing.T) {
	for _, raw := range []string{"file:///etc/passwd", "gopher://example.com/", "ftp://example.com/x"} {
		t.Run(raw, func(t *testing.T) {
			body, _ := json.Marshal(FetchRawRequest{URL: raw})
			recorder := httptest.NewRecorder()
			FetchRawHandler()(recorder, httptest.NewRequest(http.MethodPost, "/api/fetch-raw", bytes.NewReader(body)))
			if recorder.Code != http.StatusBadRequest {
				t.Errorf("FetchRaw status = %d, want 400", recorder.Code)
			}

			if response, rejected := validateOpenWebRequest(OpenWebRequest{URL: raw}); !rejected {
				t.Errorf("OpenWeb accepted %q: %+v", raw, response)
			}
		})
	}
}

// isForbiddenFetchTarget must not swallow an ordinary upstream failure, which
// would suppress the reader fallback for a legitimate URL.
func TestIsForbiddenFetchTargetIgnoresTransportFailures(t *testing.T) {
	if isForbiddenFetchTarget(context.DeadlineExceeded) {
		t.Error("a timeout was classified as a policy refusal")
	}
	if !isForbiddenFetchTarget(safefetch.ErrDestinationForbidden) {
		t.Error("a policy refusal was not recognised")
	}
}

// A refusal must not tell the caller which address was reached, or that the
// reason was an address class at all. That is how an SSRF probe maps a network
// even when every fetch fails.
func assertNoNetworkDisclosure(t *testing.T, message string, address string) {
	t.Helper()
	if strings.Contains(message, address) {
		t.Errorf("error names the resolved address %q: %s", address, message)
	}
	for _, leak := range []string{"loopback", "private", "link-local", "169.254", "127.0.0", "10.0.0"} {
		if strings.Contains(strings.ToLower(message), leak) {
			t.Errorf("error leaks network detail %q: %s", leak, message)
		}
	}
}
