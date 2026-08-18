package handlers

import (
	"net/http"
	"time"

	"github.com/vllm-project/semantic-router/dashboard/backend/routerauth"
)

func routerManagementGET(rawURL string, timeout time.Duration, providers ...routerauth.CredentialProvider) (*http.Response, error) {
	request, err := http.NewRequest(http.MethodGet, rawURL, nil)
	if err != nil {
		return nil, err
	}
	var provider routerauth.CredentialProvider
	if len(providers) > 0 {
		provider = providers[0]
	}
	if err := routerauth.RewriteAuthorization(request, provider); err != nil {
		return nil, err
	}
	return (&http.Client{Timeout: timeout}).Do(request)
}
