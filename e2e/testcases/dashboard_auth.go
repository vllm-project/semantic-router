package testcases

import (
	"context"
	"io"
	"net/http"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
)

func dashboardAuthToken(ctx context.Context, client *http.Client, baseURL string, verbose bool) (string, error) {
	return fixtures.EnsureDashboardAdmin(ctx, client, baseURL, verbose)
}

func setDashboardAuth(req *http.Request, token string) {
	req.AddCookie(&http.Cookie{Name: "vsr_session", Value: token})
}

func newAuthenticatedDashboardRequest(
	ctx context.Context,
	client *http.Client,
	method string,
	url string,
	body io.Reader,
	baseURL string,
	verbose bool,
) (*http.Request, error) {
	token, err := dashboardAuthToken(ctx, client, baseURL, verbose)
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(ctx, method, url, body)
	if err != nil {
		return nil, err
	}
	setDashboardAuth(req, token)
	return req, nil
}
