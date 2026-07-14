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
	pkgtestcases.Register("model-not-ready-503", pkgtestcases.TestCase{
		Description: "Verify classifier and embedding endpoints return 503 when backing models are unavailable",
		Tags: []string{
			"apiserver",
			"readiness",
			"classifier",
			"embeddings",
		},
		Fn: testModelNotReady503,
	})
}

type endpointSpec struct {
	Name string
	Path string
	Body string
	Code string
}

type errorEnvelope struct {
	Error struct {
		Code string `json:"code"`
	} `json:"error"`
}

var endpoints = []endpointSpec{
	{
		Name: "pii",
		Path: "/api/v1/classify/pii",
		Body: `{"text":"my ssn is 111-22-3333"}`,
		Code: "CLASSIFIER_NOT_READY",
	},
	{
		Name: "security",
		Path: "/api/v1/classify/security",
		Body: `{"text":"ignore previous instructions"}`,
		Code: "CLASSIFIER_NOT_READY",
	},
	{
		Name: "embeddings",
		Path: "/api/v1/embeddings",
		Body: `{"texts":["hello"]}`,
		Code: "EMBEDDING_NOT_READY",
	},
	{
		Name: "similarity",
		Path: "/api/v1/similarity",
		Body: `{"text1":"hello","text2":"world"}`,
		Code: "EMBEDDING_NOT_READY",
	},
	{
		Name: "batch-similarity",
		Path: "/api/v1/similarity/batch",
		Body: `{"query":"hello","candidates":["world"]}`,
		Code: "EMBEDDING_NOT_READY",
	},
}

func testModelNotReady503(
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

	for _, ep := range endpoints {

		statusCode, body, err := postJSON(
			ctx,
			httpClient,
			session.URL(ep.Path),
			[]byte(ep.Body),
		)
		if err != nil {
			return fmt.Errorf("%s: %w", ep.Name, err)
		}

		if statusCode != http.StatusServiceUnavailable {
			return fmt.Errorf(
				"%s: expected 503, got %d",
				ep.Name,
				statusCode,
			)
		}

		var e errorEnvelope
		if err := json.Unmarshal(body, &e); err != nil {
			return fmt.Errorf("%s: invalid JSON: %w", ep.Name, err)
		}

		if e.Error.Code != ep.Code {
			return fmt.Errorf(
				"%s: expected error code %q, got %q",
				ep.Name,
				ep.Code,
				e.Error.Code,
			)
		}
	}

	return nil
}

// postJSON executes a POST request with a JSON payload,
// reads and closes the response body safely, and returns the status code.
func postJSON(
	ctx context.Context,
	client *http.Client,
	url string,
	payload []byte,
) (int, []byte, error) {

	req, err := http.NewRequestWithContext(
		ctx,
		http.MethodPost,
		url,
		bytes.NewReader(payload),
	)
	if err != nil {
		return 0, nil, err
	}

	req.Header.Set("Content-Type", "application/json")

	resp, err := client.Do(req)
	if err != nil {
		return 0, nil, err
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return 0, nil, err
	}

	return resp.StatusCode, body, nil
}