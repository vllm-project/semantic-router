package routerauth

import (
	"errors"
	"net/http"
	"os"
	"testing"
)

type testCredentialProvider struct {
	token string
	err   error
}

func (provider testCredentialProvider) ManagementCredential() (string, error) {
	return provider.token, provider.err
}

func TestRewriteAuthorizationReplacesAndStripsEveryBrowserCredential(t *testing.T) {
	request, err := http.NewRequest(http.MethodGet, "http://router/config/hash?authToken=query-jwt&view=full", nil)
	if err != nil {
		t.Fatal(err)
	}
	request.Header.Set("Authorization", "Bearer dashboard-user-jwt")
	request.Header.Set("Proxy-Authorization", "Bearer proxy-user-jwt")
	request.Header.Set("Cookie", "vsr_session=cookie-user-jwt")
	if err := RewriteAuthorization(request, testCredentialProvider{token: "service-token"}); err != nil {
		t.Fatal(err)
	}
	if got := request.Header.Get("Authorization"); got != "Bearer service-token" {
		t.Fatalf("Authorization = %q", got)
	}
	if request.Header.Get("Proxy-Authorization") != "" || request.Header.Get("Cookie") != "" {
		t.Fatalf("browser credential headers leaked: %#v", request.Header)
	}
	if request.URL.Query().Get("authToken") != "" || request.URL.Query().Get("view") != "full" {
		t.Fatalf("browser query credential was not isolated: %q", request.URL.RawQuery)
	}
}

func TestRewriteAuthorizationStripsBrowserCredentialWhenManagedAuthIsDisabled(t *testing.T) {
	request, err := http.NewRequest(http.MethodGet, "http://router/health", nil)
	if err != nil {
		t.Fatal(err)
	}
	request.Header.Set("Authorization", "Bearer dashboard-user-jwt")
	if err := RewriteAuthorization(request, testCredentialProvider{err: os.ErrNotExist}); err != nil {
		t.Fatal(err)
	}
	if got := request.Header.Get("Authorization"); got != "" {
		t.Fatalf("Authorization leaked: %q", got)
	}
}

func TestRewriteAuthorizationFailsClosedOnCredentialReadError(t *testing.T) {
	request, err := http.NewRequest(http.MethodGet, "http://router/config/hash", nil)
	if err != nil {
		t.Fatal(err)
	}
	request.Header.Set("Authorization", "Bearer dashboard-user-jwt")
	if err := RewriteAuthorization(request, testCredentialProvider{err: errors.New("corrupt credential")}); err == nil {
		t.Fatal("RewriteAuthorization() unexpectedly succeeded")
	}
	if got := request.Header.Get("Authorization"); got != "" {
		t.Fatalf("Authorization leaked after provider failure: %q", got)
	}
}
