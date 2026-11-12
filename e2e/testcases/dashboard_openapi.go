package testcases

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"k8s.io/client-go/kubernetes"

	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

func init() {
	pkgtestcases.Register("dashboard-openapi", pkgtestcases.TestCase{
		Description: "Verify the Dashboard serves an authenticated OpenAPI document rendered from its route registration",
		Tags:        []string{"dashboard", "api"},
		Fn:          testDashboardOpenAPI,
	})
}

type dashboardOpenAPIOperation struct {
	OperationID  string                     `json:"operationId"`
	Permission   string                     `json:"x-vllm-sr-permission"`
	SchemaStatus string                     `json:"x-vllm-sr-schema-status"`
	Responses    map[string]json.RawMessage `json:"responses"`
}

type dashboardOpenAPIDocument struct {
	OpenAPI string                                          `json:"openapi"`
	Paths   map[string]map[string]dashboardOpenAPIOperation `json:"paths"`
}

func testDashboardOpenAPI(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	localPort, stop, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stop()

	httpClient := &http.Client{Timeout: 15 * time.Second}
	url := fmt.Sprintf("http://localhost:%s/openapi.json", localPort)

	status, _, err := getDashboardOpenAPI(ctx, httpClient, url, "")
	if err != nil {
		return err
	}
	if status != http.StatusUnauthorized {
		return fmt.Errorf("anonymous GET /openapi.json = %d, want %d", status, http.StatusUnauthorized)
	}

	token, err := dashboardAuthToken(ctx, httpClient, fmt.Sprintf("http://localhost:%s", localPort), opts.Verbose)
	if err != nil {
		return err
	}
	status, body, err := getDashboardOpenAPI(ctx, httpClient, url, token)
	if err != nil {
		return err
	}
	if status != http.StatusOK {
		return fmt.Errorf("GET /openapi.json = %d: %s", status, truncateString(string(body), 200))
	}

	var document dashboardOpenAPIDocument
	if err := json.Unmarshal(body, &document); err != nil {
		return fmt.Errorf("decode OpenAPI document: %w", err)
	}
	if err := checkDashboardOpenAPIDocument(document); err != nil {
		return err
	}

	if opts.Verbose {
		fmt.Printf("[Dashboard] OpenAPI OK: %d paths\n", len(document.Paths))
	}
	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{"paths": len(document.Paths)})
	}
	return nil
}

func getDashboardOpenAPI(ctx context.Context, client *http.Client, url, token string) (int, []byte, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return 0, nil, fmt.Errorf("create OpenAPI request: %w", err)
	}
	if token != "" {
		setDashboardAuth(req, token)
	}
	resp, err := client.Do(req)
	if err != nil {
		return 0, nil, fmt.Errorf("OpenAPI request failed: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return 0, nil, fmt.Errorf("read OpenAPI response: %w", err)
	}
	return resp.StatusCode, body, nil
}

func checkDashboardOpenAPIDocument(document dashboardOpenAPIDocument) error {
	if document.OpenAPI != "3.0.3" {
		return fmt.Errorf("openapi = %q, want 3.0.3", document.OpenAPI)
	}
	for path, item := range document.Paths {
		for method, operation := range item {
			if operation.OperationID == "" || operation.SchemaStatus == "" {
				return fmt.Errorf("%s %s lacks an operation id or schema status", method, path)
			}
		}
	}
	for _, typed := range []struct{ path, method, operationID, permission string }{
		{"/api/settings", "get", "getSettings", "config.read"},
		{"/api/setup/state", "get", "getSetupState", ""},
		{"/openapi.json", "get", "getDashboardOpenAPI", "config.read"},
	} {
		operation, ok := document.Paths[typed.path][typed.method]
		if !ok {
			return fmt.Errorf("document omits %s %s", typed.method, typed.path)
		}
		if operation.OperationID != typed.operationID || operation.Permission != typed.permission || operation.SchemaStatus != "typed" {
			return fmt.Errorf("%s %s = %+v", typed.method, typed.path, operation)
		}
		if _, ok := operation.Responses["200"]; !ok {
			return fmt.Errorf("%s %s has no 200 response", typed.method, typed.path)
		}
	}
	return nil
}
