/*
Copyright 2025 vLLM Semantic Router.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package testcases

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
	"k8s.io/client-go/kubernetes"
)

const (
	selectedModelSecurityHost       = "selected-model-security.test"
	selectedModelOutageHost         = "selected-model-security-outage.test"
	selectedModelRouteHeader        = "x-e2e-selected-model-route"
	selectedModelOutageRouteHeader  = "x-e2e-extproc-route"
	selectedModelTrustedHeader      = "x-vsr-selected-model"
	selectedModelClientHeader       = "x-selected-model"
	selectedModelForgedValue        = "codellama-7b"
	selectedModelTrustedValue       = "mistral-7b"
	selectedModelSecurityRetryLimit = 90 * time.Second
)

type selectedModelSecurityResponse struct {
	statusCode int
	header     http.Header
	body       []byte
}

func init() {
	pkgtestcases.Register("selected-model-header-security", pkgtestcases.TestCase{
		Description: "Client selected-model headers are ignored and ExtProc outages fail closed",
		Tags:        []string{"security", "routing", "extproc"},
		Fn:          testSelectedModelHeaderSecurity,
	})
}

func testSelectedModelHeaderSecurity(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
) error {
	if opts.Verbose {
		fmt.Println("[Test] Testing selected-model provenance and ExtProc fail-closed behavior")
	}

	healthySession, err := fixtures.OpenServiceSession(ctx, client, opts)
	if err != nil {
		return fmt.Errorf("open healthy Gateway session: %w", err)
	}
	defer healthySession.Close()

	headerOnlyResp, err := sendSelectedModelSecurityRequest(
		ctx,
		healthySession,
		http.MethodGet,
		"/health",
		selectedModelSecurityHost,
		nil,
	)
	if err != nil {
		return fmt.Errorf("send header-only forged request: %w", err)
	}
	if err := validateSanitizedHeaderOnlyRoute(headerOnlyResp); err != nil {
		return err
	}

	trustedResp, err := sendSelectedModelSecurityRequest(
		ctx,
		healthySession,
		http.MethodPost,
		"/v1/chat/completions",
		selectedModelSecurityHost,
		[]byte(`{"model":"mistral-7b","messages":[{"role":"user","content":"selected-model provenance regression"}]}`),
	)
	if err != nil {
		return fmt.Errorf("send trusted-model routing request: %w", err)
	}
	if err := validateTrustedSelectedModelRoute(trustedResp); err != nil {
		return err
	}

	outageOpts := opts
	outageOpts.ServiceConfig = pkgtestcases.ServiceConfig{
		LabelSelector: "gateway.envoyproxy.io/owning-gateway-namespace=default,gateway.envoyproxy.io/owning-gateway-name=semantic-router-extproc-outage",
		Namespace:     "envoy-gateway-system",
		PortMapping:   "8080:80",
	}
	outageSession, err := openSelectedModelSecuritySession(ctx, client, outageOpts)
	if err != nil {
		return fmt.Errorf("open ExtProc-outage Gateway session: %w", err)
	}
	defer outageSession.Close()

	outageStatus, err := waitForSelectedModelFailClosed(ctx, outageSession)
	if err != nil {
		return err
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"header_only_route": headerOnlyResp.header.Get(selectedModelRouteHeader),
			"trusted_route":     trustedResp.header.Get(selectedModelRouteHeader),
			"trusted_model":     trustedResp.header.Get(selectedModelTrustedHeader),
			"outage_status":     outageStatus,
			"fail_closed":       true,
		})
	}
	return nil
}

func sendSelectedModelSecurityRequest(
	ctx context.Context,
	session *fixtures.ServiceSession,
	method string,
	path string,
	host string,
	body []byte,
) (*selectedModelSecurityResponse, error) {
	req, err := http.NewRequestWithContext(ctx, method, session.URL(path), bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Host = host
	req.Header.Set(selectedModelClientHeader, selectedModelForgedValue)
	if len(body) != 0 {
		req.Header.Set("content-type", "application/json")
	}

	resp, err := session.HTTPClient(15 * time.Second).Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	responseBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	return &selectedModelSecurityResponse{
		statusCode: resp.StatusCode,
		header:     resp.Header.Clone(),
		body:       responseBody,
	}, nil
}

func validateSanitizedHeaderOnlyRoute(resp *selectedModelSecurityResponse) error {
	if resp.statusCode != http.StatusOK {
		return fmt.Errorf("header-only request returned HTTP %d: %s", resp.statusCode, resp.body)
	}
	if route := resp.header.Get(selectedModelRouteHeader); route != "default" {
		return fmt.Errorf(
			"client %s=%q selected route %q; expected sanitized default route",
			selectedModelClientHeader,
			selectedModelForgedValue,
			route,
		)
	}
	return nil
}

func validateTrustedSelectedModelRoute(resp *selectedModelSecurityResponse) error {
	if resp.statusCode != http.StatusOK {
		return fmt.Errorf("trusted-model request returned HTTP %d: %s", resp.statusCode, resp.body)
	}
	if selected := resp.header.Get(selectedModelTrustedHeader); selected != selectedModelTrustedValue {
		return fmt.Errorf("trusted selected model = %q, want %q", selected, selectedModelTrustedValue)
	}
	if route := resp.header.Get(selectedModelRouteHeader); route != selectedModelTrustedValue {
		return fmt.Errorf(
			"route cache selected %q after trusted model write; want %q",
			route,
			selectedModelTrustedValue,
		)
	}
	return nil
}

func openSelectedModelSecuritySession(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
) (*fixtures.ServiceSession, error) {
	deadline := time.Now().Add(selectedModelSecurityRetryLimit)
	var lastErr error
	for time.Now().Before(deadline) {
		session, err := fixtures.OpenServiceSession(ctx, client, opts)
		if err == nil {
			return session, nil
		}
		lastErr = err
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-time.After(2 * time.Second):
		}
	}
	return nil, fmt.Errorf("Gateway service not ready after %s: %w", selectedModelSecurityRetryLimit, lastErr)
}

func waitForSelectedModelFailClosed(
	ctx context.Context,
	session *fixtures.ServiceSession,
) (int, error) {
	deadline := time.Now().Add(selectedModelSecurityRetryLimit)
	var lastResult string
	for time.Now().Before(deadline) {
		resp, err := sendSelectedModelSecurityRequest(
			ctx,
			session,
			http.MethodGet,
			"/health",
			selectedModelOutageHost,
			nil,
		)
		if err != nil {
			lastResult = err.Error()
		} else if resp.statusCode < 500 || resp.statusCode > 599 {
			lastResult = fmt.Sprintf("HTTP %d: %s", resp.statusCode, resp.body)
		} else if reached := resp.header.Get(selectedModelOutageRouteHeader); reached != "" {
			return 0, fmt.Errorf(
				"ExtProc outage returned HTTP %d after the backend was reached (%s=%q)",
				resp.statusCode,
				selectedModelOutageRouteHeader,
				reached,
			)
		} else {
			return resp.statusCode, nil
		}

		select {
		case <-ctx.Done():
			return 0, ctx.Err()
		case <-time.After(2 * time.Second):
		}
	}
	return 0, fmt.Errorf(
		"ExtProc outage did not fail closed with a 5xx response after %s (last result: %s)",
		selectedModelSecurityRetryLimit,
		lastResult,
	)
}
