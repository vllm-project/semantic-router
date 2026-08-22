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
	pkgtestcases.Register("rag-external-api", pkgtestcases.TestCase{
		Description: "Verify typed external RAG requests and an exact-limit successful response",
		Tags:        []string{"rag", "external-api", "safety"},
		Fn:          externalRAGAPITestCase,
	})
}

func externalRAGAPITestCase(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
) error {
	gateway, err := fixtures.OpenServiceSession(ctx, client, opts)
	if err != nil {
		return fmt.Errorf("open gateway session: %w", err)
	}
	defer gateway.Close()

	if err := verifyExternalRAGTypedRequest(ctx, client, opts, gateway); err != nil {
		return err
	}
	if err := verifyExternalRAGOversizeRejection(ctx, gateway); err != nil {
		return err
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"typed_request_verified":       true,
			"exact_limit_success_verified": true,
			"one_byte_over_rejected":       true,
		})
	}
	return nil
}

func verifyExternalRAGTypedRequest(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
	gateway *fixtures.ServiceSession,
) error {
	query := `__EXTERNAL_RAG_E2E__ quote: "; slash: \\; unicode: 雪; marker: ${top_k}`
	response, err := sendExternalRAGChat(ctx, gateway, query)
	if err != nil {
		return err
	}
	if response.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(response.Body, 4096))
		response.Body.Close()
		return fmt.Errorf("chat request status %d: %s", response.StatusCode, body)
	}
	response.Body.Close()

	outbound, err := readExternalRAGMockRequest(ctx, client, opts)
	if err != nil {
		return err
	}
	if outbound.Query != query {
		return fmt.Errorf("external RAG query = %q, want exact user content", outbound.Query)
	}
	if outbound.TopK != json.Number("7") {
		return fmt.Errorf("external RAG top_k = %q, want typed number 7", outbound.TopK)
	}
	if outbound.Threshold != json.Number("0.625") {
		return fmt.Errorf("external RAG threshold = %q, want typed number 0.625", outbound.Threshold)
	}
	return nil
}

func verifyExternalRAGOversizeRejection(
	ctx context.Context,
	gateway *fixtures.ServiceSession,
) error {
	oversizedResponse, err := sendExternalRAGChat(ctx, gateway, "__EXTERNAL_RAG_E2E_OVERSIZE__")
	if err != nil {
		return err
	}
	defer oversizedResponse.Body.Close()
	oversizedBody, err := io.ReadAll(io.LimitReader(oversizedResponse.Body, 4096))
	if err != nil {
		return fmt.Errorf("read oversized-response rejection: %w", err)
	}
	if oversizedResponse.StatusCode != http.StatusServiceUnavailable {
		return fmt.Errorf(
			"oversized external RAG response status = %d, want %d: %s",
			oversizedResponse.StatusCode,
			http.StatusServiceUnavailable,
			oversizedBody,
		)
	}
	if !bytes.Contains(oversizedBody, []byte("exceeded configured limit of 38 bytes")) {
		return fmt.Errorf("oversized-response rejection omitted byte-limit detail: %s", oversizedBody)
	}
	return nil
}

func sendExternalRAGChat(
	ctx context.Context,
	gateway *fixtures.ServiceSession,
	query string,
) (*http.Response, error) {
	requestBody, err := json.Marshal(map[string]interface{}{
		"model": "MoM",
		"messages": []map[string]string{
			{"role": "user", "content": query},
		},
	})
	if err != nil {
		return nil, fmt.Errorf("marshal chat request: %w", err)
	}

	request, err := http.NewRequestWithContext(
		ctx,
		http.MethodPost,
		gateway.URL("/v1/chat/completions"),
		bytes.NewReader(requestBody),
	)
	if err != nil {
		return nil, fmt.Errorf("create chat request: %w", err)
	}
	request.Header.Set("Content-Type", "application/json")

	response, err := gateway.HTTPClient(60 * time.Second).Do(request)
	if err != nil {
		return nil, fmt.Errorf("execute chat request: %w", err)
	}
	return response, nil
}

type externalRAGMockRequest struct {
	Query     string      `json:"query"`
	TopK      json.Number `json:"top_k"`
	Threshold json.Number `json:"threshold"`
}

func readExternalRAGMockRequest(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
) (externalRAGMockRequest, error) {
	mockOptions := opts
	mockOptions.ServiceConfig = pkgtestcases.ServiceConfig{
		Name:        "external-rag-mock",
		Namespace:   "default",
		ServicePort: "8080",
	}
	mock, err := fixtures.OpenServiceSession(ctx, client, mockOptions)
	if err != nil {
		return externalRAGMockRequest{}, fmt.Errorf("open external RAG mock session: %w", err)
	}
	defer mock.Close()

	request, err := http.NewRequestWithContext(ctx, http.MethodGet, mock.URL("/last-request"), nil)
	if err != nil {
		return externalRAGMockRequest{}, fmt.Errorf("create mock inspection request: %w", err)
	}
	response, err := mock.HTTPClient(15 * time.Second).Do(request)
	if err != nil {
		return externalRAGMockRequest{}, fmt.Errorf("inspect external RAG request: %w", err)
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		return externalRAGMockRequest{}, fmt.Errorf("mock inspection status %d", response.StatusCode)
	}

	decoder := json.NewDecoder(response.Body)
	decoder.UseNumber()
	var outbound externalRAGMockRequest
	if err := decoder.Decode(&outbound); err != nil {
		return externalRAGMockRequest{}, fmt.Errorf("decode captured external RAG request: %w", err)
	}
	return outbound, nil
}
