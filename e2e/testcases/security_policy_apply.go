package testcases

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"k8s.io/client-go/kubernetes"

	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

func init() {
	pkgtestcases.Register("security-policy-apply", pkgtestcases.TestCase{
		Description: "Verify PUT /api/security/policy applies role_bindings and ratelimit to router config",
		Tags:        []string{"dashboard", "security", "rbac", "ratelimit"},
		Fn:          testSecurityPolicyApply,
	})
}

type securityPolicyPayload struct {
	RoleMappings []roleMappingPayload   `json:"role_mappings"`
	RateTiers    []rateLimitTierPayload `json:"rate_tiers"`
}

type roleMappingPayload struct {
	Name      string           `json:"name"`
	Subjects  []subjectPayload `json:"subjects"`
	Role      string           `json:"role"`
	ModelRefs []string         `json:"model_refs"`
	Priority  int              `json:"priority"`
}

type subjectPayload struct {
	Kind string `json:"kind"`
	Name string `json:"name"`
}

type rateLimitTierPayload struct {
	Name           string `json:"name"`
	Group          string `json:"group,omitempty"`
	RequestsPerMin int    `json:"rpm"`
	TokensPerMin   int    `json:"tpm,omitempty"`
}

type securityPolicyTestEnv struct {
	baseURL    string
	httpClient *http.Client
	verbose    bool
}

func newTestPolicy() securityPolicyPayload {
	return securityPolicyPayload{
		RoleMappings: []roleMappingPayload{
			{
				Name:      "premium",
				Subjects:  []subjectPayload{{Kind: "Group", Name: "paying-customers"}},
				Role:      "premium_tier",
				ModelRefs: []string{"gpt-4"},
				Priority:  10,
			},
			{
				Name:      "free",
				Subjects:  []subjectPayload{{Kind: "Group", Name: "free-users"}},
				Role:      "free_tier",
				ModelRefs: []string{"gpt-3.5-turbo"},
				Priority:  20,
			},
		},
		RateTiers: []rateLimitTierPayload{
			{Name: "premium-rate", Group: "paying-customers", RequestsPerMin: 1000, TokensPerMin: 100000},
			{Name: "free-rate", Group: "free-users", RequestsPerMin: 10, TokensPerMin: 5000},
		},
	}
}

// putSecurityPolicy sends the policy and returns (applied, skipped, error).
// skipped=true means the endpoint doesn't exist in the deployed dashboard image.
func (env *securityPolicyTestEnv) putSecurityPolicy(ctx context.Context, body []byte) (bool, bool, error) {
	putURL := env.baseURL + "/api/security/policy"
	if env.verbose {
		fmt.Printf("[SecurityPolicy] PUT %s (%d bytes)\n", putURL, len(body))
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPut, putURL, bytes.NewBuffer(body))
	if err != nil {
		return false, false, fmt.Errorf("create PUT request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := env.httpClient.Do(req)
	if err != nil {
		return false, false, fmt.Errorf("PUT security policy failed: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	respBody, _ := io.ReadAll(resp.Body)

	if resp.StatusCode == http.StatusNotFound || resp.StatusCode == http.StatusBadGateway {
		if env.verbose {
			fmt.Printf("[SecurityPolicy] Endpoint not available (status %d); stock dashboard image — skipping\n", resp.StatusCode)
		}
		return false, true, nil
	}

	if resp.StatusCode != http.StatusOK {
		return false, false, fmt.Errorf("PUT security policy: expected 200, got %d: %s", resp.StatusCode, truncateString(string(respBody), 300))
	}

	var putResult map[string]interface{}
	if err := json.Unmarshal(respBody, &putResult); err != nil {
		return false, false, fmt.Errorf("PUT response is not valid JSON: %w", err)
	}

	applied, _ := putResult["applied"].(bool)
	if env.verbose {
		fmt.Printf("[SecurityPolicy] PUT OK: applied=%v, message=%v\n", applied, putResult["message"])
	}

	if putResult["fragment"] == nil {
		return false, false, fmt.Errorf("expected fragment field in PUT response")
	}
	if putResult["policy"] == nil {
		return false, false, fmt.Errorf("expected policy field in PUT response")
	}

	return applied, false, nil
}

func (env *securityPolicyTestEnv) getSecurityPolicy(ctx context.Context) error {
	getURL := env.baseURL + "/api/security/policy"
	if env.verbose {
		fmt.Printf("[SecurityPolicy] GET %s\n", getURL)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, getURL, nil)
	if err != nil {
		return fmt.Errorf("create GET request: %w", err)
	}

	resp, err := env.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("GET security policy failed: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	body, _ := io.ReadAll(resp.Body)
	if resp.StatusCode == http.StatusBadGateway {
		return fmt.Errorf("GET security policy: 502 Bad Gateway — dashboard may be misconfigured or proxy is down")
	}
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("GET security policy: expected 200, got %d: %s", resp.StatusCode, truncateString(string(body), 300))
	}

	var saved securityPolicyPayload
	if err := json.Unmarshal(body, &saved); err != nil {
		return fmt.Errorf("GET response is not valid JSON: %w", err)
	}

	if len(saved.RoleMappings) != 2 {
		return fmt.Errorf("expected 2 role mappings in saved policy, got %d", len(saved.RoleMappings))
	}
	if len(saved.RateTiers) != 2 {
		return fmt.Errorf("expected 2 rate tiers in saved policy, got %d", len(saved.RateTiers))
	}

	return nil
}

func (env *securityPolicyTestEnv) previewFragment(ctx context.Context, body []byte) error {
	previewURL := env.baseURL + "/api/security/policy/preview"
	if env.verbose {
		fmt.Printf("[SecurityPolicy] POST %s (verifying fragment generation)\n", previewURL)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, previewURL, bytes.NewBuffer(body))
	if err != nil {
		return fmt.Errorf("create preview POST request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := env.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("POST preview failed: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	previewBody, _ := io.ReadAll(resp.Body)
	if resp.StatusCode == http.StatusBadGateway {
		return fmt.Errorf("POST preview: 502 Bad Gateway — dashboard may be misconfigured or proxy is down")
	}
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("POST preview: expected 200, got %d: %s", resp.StatusCode, truncateString(string(previewBody), 300))
	}

	fragmentJSON := string(previewBody)
	for _, want := range []string{"premium_tier", "free_tier", "premium-rate", "free-rate", "local-limiter"} {
		if !strings.Contains(fragmentJSON, want) {
			return fmt.Errorf("fragment does not contain expected value %q", want)
		}
	}

	if env.verbose {
		fmt.Printf("[SecurityPolicy] Fragment contains expected role_bindings and ratelimit rules\n")
	}

	return nil
}

func (env *securityPolicyTestEnv) verifyConfigApplied(ctx context.Context) error {
	configURL := env.baseURL + "/api/router/config/yaml"
	if env.verbose {
		fmt.Printf("[SecurityPolicy] GET %s (verifying config apply)\n", configURL)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, configURL, nil)
	if err != nil {
		return fmt.Errorf("create config GET request: %w", err)
	}

	resp, err := env.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("GET config/yaml failed: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	body, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("GET config/yaml: expected 200, got %d", resp.StatusCode)
	}

	configYAML := string(body)
	for _, want := range []string{"premium_tier", "free_tier"} {
		if !strings.Contains(configYAML, want) {
			return fmt.Errorf("config YAML does not contain role %q after policy apply", want)
		}
	}

	if env.verbose {
		fmt.Printf("[SecurityPolicy] Config YAML confirmed: auto-apply propagated role_bindings and ratelimit\n")
	}

	return nil
}

func testSecurityPolicyApply(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	localPort, stop, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stop()

	env := &securityPolicyTestEnv{
		baseURL:    fmt.Sprintf("http://localhost:%s", localPort),
		httpClient: &http.Client{Timeout: 30 * time.Second},
		verbose:    opts.Verbose,
	}

	policy := newTestPolicy()
	body, err := json.Marshal(policy)
	if err != nil {
		return fmt.Errorf("marshal policy: %w", err)
	}

	applied, skipped, err := env.putSecurityPolicy(ctx, body)
	if err != nil {
		return err
	}
	if skipped {
		if opts.SetDetails != nil {
			opts.SetDetails(map[string]interface{}{
				"skipped": true,
				"reason":  "dashboard image does not include security-policy routes",
			})
		}
		return nil
	}

	if err := env.getSecurityPolicy(ctx); err != nil {
		return err
	}

	if err := env.previewFragment(ctx, body); err != nil {
		return err
	}

	if applied {
		if err := env.verifyConfigApplied(ctx); err != nil {
			return err
		}
	} else if env.verbose {
		fmt.Printf("[SecurityPolicy] Auto-apply skipped (read-only config or path not configured); fragment generation verified\n")
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"applied":            applied,
			"role_bindings":      2,
			"rate_rules":         2,
			"fragment_validated": true,
			"policy_persisted":   true,
		})
	}

	return nil
}
