package handlers

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"strings"
	"time"
)

func newEnvoyActivationVerifier(readyURL string, client *http.Client) func(context.Context) error {
	endpoint, endpointErr := envoyReadyEndpoint(readyURL)
	if client == nil {
		client = &http.Client{
			Transport: &http.Transport{Proxy: nil},
			Timeout:   5 * time.Second,
			CheckRedirect: func(_ *http.Request, _ []*http.Request) error {
				return http.ErrUseLastResponse
			},
		}
	}
	return func(ctx context.Context) error {
		if endpointErr != nil {
			return endpointErr
		}
		var lastStatus int
		for {
			request, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint, nil)
			if err != nil {
				return err
			}
			response, err := client.Do(request)
			if err == nil {
				lastStatus = response.StatusCode
				_ = response.Body.Close()
				if response.StatusCode >= http.StatusOK && response.StatusCode < http.StatusMultipleChoices {
					return nil
				}
			}
			select {
			case <-ctx.Done():
				return errors.Join(ctx.Err(), fmt.Errorf("envoy readiness did not become healthy (last status %d)", lastStatus))
			case <-time.After(500 * time.Millisecond):
			}
		}
	}
}

func envoyReadyEndpoint(raw string) (string, error) {
	parsed, err := url.Parse(strings.TrimSpace(raw))
	if err != nil || parsed.Host == "" || (parsed.Scheme != "http" && parsed.Scheme != "https") || parsed.User != nil {
		return "", errors.New("envoy readiness endpoint is not configured")
	}
	parsed.RawQuery = ""
	parsed.Fragment = ""
	return parsed.String(), nil
}
