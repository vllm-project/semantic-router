package testcases

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
	"k8s.io/client-go/kubernetes"
)

func init() {
	pkgtestcases.Register("apiserver-config-validate", pkgtestcases.TestCase{
		Description: "Verify POST /api/v1/config/validate returns the v1 diagnostic contract without mutating the live config (issue #3477)",
		Tags:        []string{"apiserver", "config", "api", "validate"},
		Fn:          testAPIServerConfigValidate,
	})
}

type routerConfigValidateResponse struct {
	Valid           bool   `json:"valid"`
	ContractVersion string `json:"contract_version"`
	NormalizedYAML  string `json:"normalized_yaml"`
	Errors          []struct {
		Code  string `json:"code"`
		Field string `json:"field"`
	} `json:"errors"`
}

func testAPIServerConfigValidate(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
) error {
	session, err := fixtures.OpenRouterAPISession(ctx, client, opts)
	if err != nil {
		return err
	}
	defer session.Close()

	httpClient := session.HTTPClient(30 * time.Second)
	hashBefore, err := fetchConfigHash(ctx, httpClient, session.URL("/api/v1/config/hash"))
	if err != nil {
		return err
	}

	validURL := session.URL("/api/v1/config/validate")
	validBody, err := postRouterConfigValidate(ctx, httpClient, validURL, map[string]any{
		"yaml": validateE2EValidYAML,
	})
	if err != nil {
		return err
	}
	if err := assertValidValidateContract(validBody); err != nil {
		return err
	}

	invalidBody, err := postRouterConfigValidate(ctx, httpClient, validURL, map[string]any{
		"yaml": validateE2EInvalidYAML,
	})
	if err != nil {
		return err
	}
	if err := assertInvalidValidateContract(invalidBody); err != nil {
		return err
	}

	hashAfter, err := fetchConfigHash(ctx, httpClient, session.URL("/api/v1/config/hash"))
	if err != nil {
		return err
	}
	if hashBefore != hashAfter {
		return fmt.Errorf("validate mutated live config hash: before=%s after=%s", hashBefore, hashAfter)
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"contract_version": validBody.ContractVersion,
			"invalid_code":     invalidBody.Errors[0].Code,
			"invalid_field":    invalidBody.Errors[0].Field,
			"config_hash":      hashAfter,
		})
	}
	return nil
}

func assertValidValidateContract(body *routerConfigValidateResponse) error {
	if body.Valid && body.ContractVersion == "v1" {
		return nil
	}
	return fmt.Errorf("expected valid=true contract_version=v1, got valid=%v version=%q", body.Valid, body.ContractVersion)
}

func assertInvalidValidateContract(body *routerConfigValidateResponse) error {
	if body.Valid {
		return fmt.Errorf("expected valid=false for an undeclared signal reference")
	}
	if len(body.Errors) == 0 || body.Errors[0].Code == "" || body.Errors[0].Field == "" {
		return fmt.Errorf("expected a field-addressable error, got %+v", body.Errors)
	}
	return nil
}

func postRouterConfigValidate(
	ctx context.Context,
	httpClient *http.Client,
	url string,
	payload map[string]any,
) (*routerConfigValidateResponse, error) {
	encoded, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("marshal validate request: %w", err)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(encoded))
	if err != nil {
		return nil, fmt.Errorf("create validate request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("send validate request: %w", err)
	}
	defer func() {
		_ = resp.Body.Close()
	}()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read validate response: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("expected /api/v1/config/validate status 200, got %d: %s", resp.StatusCode, string(body))
	}
	var decoded routerConfigValidateResponse
	if err := json.Unmarshal(body, &decoded); err != nil {
		return nil, fmt.Errorf("decode validate response: %w", err)
	}
	return &decoded, nil
}

func fetchConfigHash(ctx context.Context, httpClient *http.Client, url string) (string, error) {
	resp, err := getJSON(ctx, httpClient, url)
	if err != nil {
		return "", err
	}
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("expected /api/v1/config/hash status 200, got %d: %s", resp.StatusCode, string(resp.Body))
	}
	var decoded struct {
		Hash string `json:"source_config_hash"`
	}
	if err := json.Unmarshal(resp.Body, &decoded); err != nil {
		return "", fmt.Errorf("decode /api/v1/config/hash: %w", err)
	}
	if decoded.Hash == "" {
		return "", fmt.Errorf("expected /api/v1/config/hash to include a source hash")
	}
	return decoded.Hash, nil
}

const validateE2EValidYAML = `version: v0.3
listeners: []
providers:
  defaults:
    model: m1
  models:
    - name: m1
      backend_refs:
        - endpoint: 127.0.0.1:8000
          provider: vllm
routing:
  modelCards:
    - name: m1
  decisions:
    - name: d1
      priority: 1
      rules: {operator: AND, conditions: []}
      modelRefs:
        - model: m1
          use_reasoning: false
`

const validateE2EInvalidYAML = `version: v0.3
listeners: []
providers:
  defaults:
    model: m1
  models:
    - name: m1
      backend_refs:
        - endpoint: 127.0.0.1:8000
          provider: vllm
routing:
  modelCards:
    - name: m1
  signals:
    keywords:
      - name: hello
        operator: OR
        keywords: ["hi"]
  decisions:
    - name: triage
      priority: 1
      rules:
        operator: OR
        conditions:
          - type: keyword
            name: nope
      modelRefs:
        - model: m1
          use_reasoning: false
`
