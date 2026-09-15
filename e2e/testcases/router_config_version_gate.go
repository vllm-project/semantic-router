package testcases

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	"k8s.io/client-go/kubernetes"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

func init() {
	pkgtestcases.Register("router-config-version-gate", pkgtestcases.TestCase{
		Description: "Verify the running Router accepts the supported canonical version and rejects an unsupported one before interpreting the document",
		Tags:        []string{"apiserver", "config", "validation"},
		Fn:          testRouterConfigVersionGate,
	})
}

// versionGateDocument is one canonical document parameterized only by `version`.
// Both cases submit the same bytes apart from that field, so a difference in the
// outcome is attributable to the version gate and nothing else.
const versionGateDocument = `version: %s
listeners:
  - name: http-8899
    address: 0.0.0.0
    port: 8899
    timeout: 300s
providers:
  defaults:
    model: local/general
  models:
    - name: local/general
      provider_model_id: my-served-model
      backend_refs:
        - name: primary
          endpoint: host.docker.internal:8000
          protocol: http
          provider: vllm
routing:
  strategy: priority
  modelCards:
    - name: local/general
      modality: text
      capabilities: [chat]
  decisions:
    - name: default_answer
      description: Default route.
      priority: 100
      rules:
        operator: AND
        conditions: []
      modelRefs:
        - model: local/general
`

// unsupportedCanonicalVersion is a contract this build must not read. It is a
// past contract rather than a future one, so the case stays valid after a bump.
const unsupportedCanonicalVersion = "v0.2"

type routerConfigValidateResponse struct {
	Valid          bool   `json:"valid"`
	NormalizedYAML string `json:"normalized_yaml"`
}

type routerConfigErrorResponse struct {
	Error struct {
		Code    string `json:"code"`
		Message string `json:"message"`
	} `json:"error"`
}

// testRouterConfigVersionGate drives POST /config/router/validate on the running
// Router. That endpoint runs the same canonical parse as serve and hot-reload but
// writes nothing, so the case asserts the real runtime contract without mutating
// the deployed configuration. See issue #2469.
func testRouterConfigVersionGate(
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
	url := session.URL("/config/router/validate")

	supportedVersion, err := validateVersionedDocument(ctx, httpClient, url, "v0.3", opts.Verbose)
	if err != nil {
		return err
	}

	if err := rejectUnsupportedVersion(ctx, httpClient, url, opts.Verbose); err != nil {
		return err
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"supported_version":   supportedVersion,
			"unsupported_version": unsupportedCanonicalVersion,
		})
	}

	return nil
}

// validateVersionedDocument asserts the supported contract is accepted, so a
// later rejection cannot be blamed on the rest of the document.
func validateVersionedDocument(
	ctx context.Context,
	httpClient *http.Client,
	url string,
	version string,
	verbose bool,
) (string, error) {
	resp, err := postVersionedDocument(ctx, httpClient, url, version)
	if err != nil {
		return "", err
	}

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf(
			"expected version %q to validate with 200, got %d: %s",
			version, resp.StatusCode, truncateString(string(resp.Body), 300),
		)
	}

	var decoded routerConfigValidateResponse
	if err := json.Unmarshal(resp.Body, &decoded); err != nil {
		return "", fmt.Errorf("decode validate response: %w", err)
	}
	if !decoded.Valid {
		return "", fmt.Errorf("expected valid=true for version %q, got %+v", version, decoded)
	}

	if verbose {
		fmt.Printf("[ConfigVersion] version %s accepted by /config/router/validate\n", version)
	}
	return version, nil
}

// rejectUnsupportedVersion asserts the same document is refused for an
// unsupported contract, with an error that names the field and the accepted set.
func rejectUnsupportedVersion(
	ctx context.Context,
	httpClient *http.Client,
	url string,
	verbose bool,
) error {
	resp, err := postVersionedDocument(ctx, httpClient, url, unsupportedCanonicalVersion)
	if err != nil {
		return err
	}

	if resp.StatusCode != http.StatusUnprocessableEntity {
		return fmt.Errorf(
			"expected version %q to be rejected with 422, got %d: %s",
			unsupportedCanonicalVersion, resp.StatusCode, truncateString(string(resp.Body), 300),
		)
	}

	var decoded routerConfigErrorResponse
	if err := json.Unmarshal(resp.Body, &decoded); err != nil {
		return fmt.Errorf("decode validation error response: %w", err)
	}
	if decoded.Error.Code != "CONFIG_VALIDATION_ERROR" {
		return fmt.Errorf(
			"expected error code CONFIG_VALIDATION_ERROR, got %q: %s",
			decoded.Error.Code, truncateString(string(resp.Body), 300),
		)
	}

	// The message must identify the offending field, otherwise a rejection for an
	// unrelated reason would pass this case.
	message := decoded.Error.Message
	for _, needle := range []string{"version:", unsupportedCanonicalVersion} {
		if !strings.Contains(message, needle) {
			return fmt.Errorf("validation error does not mention %q: %s", needle, truncateString(message, 300))
		}
	}

	if verbose {
		fmt.Printf("[ConfigVersion] version %s rejected: %s\n", unsupportedCanonicalVersion, truncateString(message, 200))
	}
	return nil
}

func postVersionedDocument(
	ctx context.Context,
	httpClient *http.Client,
	url string,
	version string,
) (*httpResponse, error) {
	payload, err := json.Marshal(map[string]string{
		"yaml": fmt.Sprintf(versionGateDocument, version),
	})
	if err != nil {
		return nil, fmt.Errorf("marshal validate payload: %w", err)
	}
	return postJSON(ctx, httpClient, http.MethodPost, url, payload)
}
