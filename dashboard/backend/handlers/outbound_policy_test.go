package handlers

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"net/netip"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/vllm-project/semantic-router/dashboard/backend/safefetch"
	"github.com/vllm-project/semantic-router/dashboard/backend/setupmode"
)

// inwardResolver answers every name with an internal address, standing in for
// a caller who supplies a public-looking host that points at the cluster.
type inwardResolver struct{ address string }

func (r inwardResolver) LookupNetIP(context.Context, string, string) ([]netip.Addr, error) {
	return []netip.Addr{netip.MustParseAddr(r.address)}, nil
}

type timeoutResolver struct{ calls atomic.Int32 }

func (r *timeoutResolver) LookupNetIP(context.Context, string, string) ([]netip.Addr, error) {
	r.calls.Add(1)
	return nil, errors.New("resolver timeout at 10.0.0.1")
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

	result, revoked, err := fetchOpenWeb(plan, func() bool { return false })
	if revoked {
		t.Fatal("policy refusal must not be treated as authorization revocation")
	}
	if err == nil {
		t.Fatalf("a refused destination succeeded: %+v", result)
	}
	if !errors.Is(err, errOpenWebForbiddenTarget) {
		t.Errorf("error = %v, want the terminal policy refusal", err)
	}
}

// A resolver timeout is wrapped by the outbound policy's refusal. The timeout
// text must not erase that refusal or send the same target through Jina.
func TestOpenWebResolverTimeoutKeepsPolicyRefusal(t *testing.T) {
	resolver := &timeoutResolver{}
	previous := outboundResolver
	outboundResolver = resolver
	t.Cleanup(func() { outboundResolver = previous })

	plan := buildOpenWebFetchPlan(OpenWebRequest{URL: "https://public-looking-name.invalid/?secret=token"})
	result, revoked, err := fetchOpenWeb(plan, func() bool { return false })
	if revoked || result != nil || !errors.Is(err, errOpenWebForbiddenTarget) {
		t.Fatalf("fetchOpenWeb() = result=%+v revoked=%v err=%v, want terminal refusal", result, revoked, err)
	}
	if got := resolver.calls.Load(); got != 1 {
		t.Fatalf("resolver calls = %d, want one direct request and no Jina retry", got)
	}
	assertNoNetworkDisclosure(t, err.Error(), "10.0.0.1")
	if strings.Contains(err.Error(), "secret=token") || strings.Contains(err.Error(), "resolver timeout") {
		t.Fatalf("policy refusal disclosed URL or resolver detail: %v", err)
	}
}

// Reader mode sends the target URL to an external service. Reject private IPs
// and ambiguous local spellings before any request to that service.
func TestOpenWebReaderModeRejectsLiteralPrivateTargets(t *testing.T) {
	tests := []struct {
		name string
		url  string
		req  OpenWebRequest
	}{
		{"IPv4 loopback", "http://127.0.0.1/private", OpenWebRequest{ForceJina: true}},
		{"IPv6 loopback", "http://[::1]/private", OpenWebRequest{ForceJina: true}},
		{"metadata", "http://169.254.169.254/latest", OpenWebRequest{ForceJina: true}},
		{"numeric shorthand", "http://127.1/private", OpenWebRequest{ForceJina: true}},
		{"decimal IPv4", "http://2130706433/private", OpenWebRequest{ForceJina: true}},
		{"hexadecimal IPv4", "http://0x7f000001/private", OpenWebRequest{ForceJina: true}},
		{"dotted localhost", "http://localhost../private", OpenWebRequest{ForceJina: true}},
		{"local name", "http://service.local/private", OpenWebRequest{WithImages: true}},
		{"internal name", "http://service.internal/private", OpenWebRequest{ForceJina: true}},
		{"PDF target", "http://10.0.0.1/guide.pdf", OpenWebRequest{}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			resolver := &timeoutResolver{}
			previous := outboundResolver
			outboundResolver = resolver
			t.Cleanup(func() { outboundResolver = previous })

			req := tt.req
			req.URL = tt.url
			body, _ := json.Marshal(req)
			recorder := httptest.NewRecorder()
			OpenWebHandler()(recorder, httptest.NewRequest(http.MethodPost, "/api/openweb", bytes.NewReader(body)))
			if recorder.Code != http.StatusBadRequest {
				t.Fatalf("status = %d, want 400: %s", recorder.Code, recorder.Body.String())
			}
			var response OpenWebResponse
			if err := json.Unmarshal(recorder.Body.Bytes(), &response); err != nil {
				t.Fatalf("decode response: %v", err)
			}
			if response.Error != errOpenWebForbiddenTarget.Error() || response.Content != "" {
				t.Fatalf("response = %+v, want empty content and generic refusal", response)
			}
			if calls := resolver.calls.Load(); calls != 0 {
				t.Fatalf("resolver calls = %d, want no outbound request", calls)
			}
		})
	}
}

func TestOpenWebReaderModeAcceptsPublicTargets(t *testing.T) {
	for _, raw := range []string{"https://example.com/", "http://8.8.8.8/"} {
		if response, rejected := validateOpenWebRequest(OpenWebRequest{URL: raw, ForceJina: true}); rejected {
			t.Errorf("public target %q rejected: %+v", raw, response)
		}
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

// The setup import endpoint takes a caller-supplied URL too, and was the one
// path still building its own client (review on #3617).
func TestSetupImportRemoteRefusesInwardDestinations(t *testing.T) {
	for name, address := range inwardTargets {
		t.Run(name, func(t *testing.T) {
			withInwardResolver(t, address)

			directory := t.TempDir()
			configPath := filepath.Join(directory, "config.yaml")
			if err := os.WriteFile(configPath, []byte(setupBootstrapConfig), 0o600); err != nil {
				t.Fatalf("write bootstrap config: %v", err)
			}

			body, _ := json.Marshal(SetupImportRemoteRequest{URL: "https://public-looking-name.invalid/config.yaml"})
			recorder := httptest.NewRecorder()
			SetupImportRemoteHandler(configPath, setupmode.New(configPath, true))(
				recorder,
				httptest.NewRequest(http.MethodPost, "/api/setup/import-remote", bytes.NewReader(body)),
			)

			if recorder.Code == http.StatusOK {
				t.Fatalf("a refused destination returned 200: %s", recorder.Body.String())
			}
			if recorder.Code != http.StatusBadRequest {
				t.Fatalf("status = %d, want 400: %s", recorder.Code, recorder.Body.String())
			}
			assertNoNetworkDisclosure(t, recorder.Body.String(), address)
		})
	}
}

// normalizeRemoteConfigURL keeps its existing messages while delegating the
// syntactic checks, so the setup UI's error text is unchanged.
func TestNormalizeRemoteConfigURLRejectsUnsafeInput(t *testing.T) {
	tests := []struct {
		raw  string
		want string
	}{
		{"", "remote config URL is required"},
		{"file:///etc/passwd", "remote config URL must use http or https"},
		{"gopher://example.com/", "remote config URL must use http or https"},
		{"https://", "invalid remote config URL"},
		{"/relative/path", "invalid remote config URL"},
		{"https://user:pass@example.com/c.yaml", "invalid remote config URL"},
	}

	for _, tt := range tests {
		t.Run(tt.raw, func(t *testing.T) {
			got, err := normalizeRemoteConfigURL(tt.raw)
			if err == nil {
				t.Fatalf("normalizeRemoteConfigURL(%q) = %q, want an error", tt.raw, got)
			}
			if !strings.Contains(err.Error(), tt.want) {
				t.Errorf("error = %q, want it to contain %q", err.Error(), tt.want)
			}
		})
	}
}

func TestNormalizeRemoteConfigURLAcceptsPublicURLs(t *testing.T) {
	for _, raw := range []string{"https://example.com/config.yaml", "http://example.com/c.yaml"} {
		if _, err := normalizeRemoteConfigURL(raw); err != nil {
			t.Errorf("normalizeRemoteConfigURL(%q) = %v, want accepted", raw, err)
		}
	}
}

const setupBootstrapConfig = `version: v0.3
listeners:
  - name: http-8899
    address: 0.0.0.0
    port: 8899
`

// allowLoopbackForTest declares the loopback fixture server as a permitted
// destination, the same way an operator would declare a real internal target.
// It does not disable the check: everything outside the prefix stays refused.
func allowLoopbackForTest(t *testing.T) {
	t.Helper()
	previous := outboundAllowedPrivatePrefixes
	outboundAllowedPrivatePrefixes = []netip.Prefix{
		netip.MustParsePrefix("127.0.0.0/8"),
		netip.MustParsePrefix("::1/128"),
	}
	t.Cleanup(func() { outboundAllowedPrivatePrefixes = previous })
}

// The allowlist is narrow: declaring loopback does not admit other private
// ranges.
func TestAllowlistDoesNotWidenBeyondItsPrefixes(t *testing.T) {
	allowLoopbackForTest(t)
	withInwardResolver(t, "10.0.0.1")

	body, _ := json.Marshal(FetchRawRequest{URL: "https://public-looking-name.invalid/c.yaml"})
	recorder := httptest.NewRecorder()
	FetchRawHandler()(recorder, httptest.NewRequest(http.MethodPost, "/api/fetch-raw", bytes.NewReader(body)))

	if recorder.Code != http.StatusBadRequest {
		t.Fatalf("a private address outside the allowlist was accepted: %d %s", recorder.Code, recorder.Body.String())
	}
}
