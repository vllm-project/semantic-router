package testcases

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
	"k8s.io/client-go/kubernetes"
)

const (
	embeddingE2EInputLimit = 100
	embeddingE2ETextLimit  = 128 * 1024
)

type embeddingInputLimitAPIError struct {
	Error struct {
		Code string `json:"code"`
	} `json:"error"`
}

func init() {
	pkgtestcases.Register("apiserver-embedding-input-limits", pkgtestcases.TestCase{
		Description: "Reject excessive embedding and similarity input counts or text sizes before native inference",
		Tags:        []string{"kubernetes", "apiserver", "embedding", "security"},
		Fn:          testAPIServerEmbeddingInputLimits,
	})
}

func testAPIServerEmbeddingInputLimits(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
) error {
	session, err := fixtures.OpenRouterAPISession(ctx, client, opts)
	if err != nil {
		return err
	}
	defer session.Close()

	overLimitTexts := repeatedE2EStrings(embeddingE2EInputLimit+1, "bounded-input")
	overLimitCandidates := repeatedE2EStrings(embeddingE2EInputLimit+1, "candidate")
	oversizedText := strings.Repeat("private-oversized-text-", embeddingE2ETextLimit/23+1)
	requests := []struct {
		name    string
		path    string
		payload interface{}
	}{
		{name: "embeddings-101-inputs", path: "/api/v1/embeddings", payload: map[string]interface{}{"texts": overLimitTexts}},
		{name: "embeddings-oversized-text", path: "/api/v1/embeddings", payload: map[string]interface{}{"texts": []string{oversizedText}}},
		{name: "similarity-oversized-text", path: "/api/v1/similarity", payload: map[string]interface{}{"text1": oversizedText, "text2": "valid"}},
		{name: "batch-similarity-101-candidates", path: "/api/v1/similarity/batch", payload: map[string]interface{}{"query": "query", "candidates": overLimitCandidates}},
	}

	httpClient := session.HTTPClient(30 * time.Second)
	checked := make([]string, 0, len(requests))
	for _, request := range requests {
		body, err := json.Marshal(request.payload)
		if err != nil {
			return fmt.Errorf("marshal %s request: %w", request.name, err)
		}
		resp, err := postJSON(ctx, httpClient, http.MethodPost, session.URL(request.path), body)
		if err != nil {
			return err
		}
		if err := validateEmbeddingInputLimitRejection(request.name, resp, oversizedText); err != nil {
			return err
		}
		checked = append(checked, request.name)
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{"validated_contracts": checked})
	}
	return nil
}

func validateEmbeddingInputLimitRejection(name string, resp *httpResponse, forbiddenText string) error {
	if resp.StatusCode != http.StatusRequestEntityTooLarge {
		return fmt.Errorf("expected %s status 413, got %d: %s", name, resp.StatusCode, string(resp.Body))
	}
	var document embeddingInputLimitAPIError
	if err := json.Unmarshal(resp.Body, &document); err != nil {
		return fmt.Errorf("decode %s response: %w", name, err)
	}
	if document.Error.Code != "EMBEDDING_INPUT_TOO_LARGE" {
		return fmt.Errorf("expected %s code EMBEDDING_INPUT_TOO_LARGE, got %q", name, document.Error.Code)
	}
	if forbiddenText != "" && strings.Contains(string(resp.Body), forbiddenText) {
		return fmt.Errorf("%s reflected oversized input", name)
	}
	return nil
}

func repeatedE2EStrings(count int, value string) []string {
	values := make([]string, count)
	for i := range values {
		values[i] = value
	}
	return values
}
