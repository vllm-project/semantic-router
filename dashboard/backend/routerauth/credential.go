package routerauth

import (
	"errors"
	"net/http"
	"os"
	"strings"
)

// CredentialProvider resolves the Dashboard-owned Router management identity.
// Implementations must keep the credential server-side and return os.ErrNotExist
// when management auth is disabled or no managed credential has been created.
type CredentialProvider interface {
	ManagementCredential() (string, error)
}

// RewriteAuthorization replaces any browser credential with the Dashboard's
// scoped Router identity. An absent managed credential intentionally produces
// no Authorization header so auth-disabled Router configurations keep working.
func RewriteAuthorization(request *http.Request, provider CredentialProvider) error {
	if request == nil {
		return errors.New("router request is nil")
	}
	StripBrowserCredentials(request)
	if provider == nil {
		return nil
	}
	token, err := provider.ManagementCredential()
	if errors.Is(err, os.ErrNotExist) {
		return nil
	}
	if err != nil {
		return err
	}
	token = strings.TrimSpace(token)
	if token == "" {
		return errors.New("router management credential is empty")
	}
	request.Header.Set("Authorization", "Bearer "+token)
	return nil
}

// StripBrowserCredentials removes every Dashboard session credential form
// before a request leaves the Dashboard trust boundary for Router or Envoy.
func StripBrowserCredentials(request *http.Request) {
	if request == nil {
		return
	}
	request.Header.Del("Authorization")
	request.Header.Del("Proxy-Authorization")
	request.Header.Del("Cookie")
	if request.URL != nil {
		query := request.URL.Query()
		query.Del("authToken")
		request.URL.RawQuery = query.Encode()
	}
}
