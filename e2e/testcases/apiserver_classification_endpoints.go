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
	pkgtestcases.Register("apiserver-classification-endpoints", pkgtestcases.TestCase{
		Description: "Verify the standalone immutable-config boundary, metrics, and combined classification",
		Tags:        []string{"kubernetes", "apiserver", "classification", "api"},
		Fn:          testAPIServerClassificationEndpoints,
	})
}

type classificationMetricsDocument struct {
	DecisionCount    int `json:"decision_count"`
	SignalGroupCount int `json:"signal_group_count"`
}

func testAPIServerClassificationEndpoints(
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
	if err := assertStandaloneConfigRoutesAbsent(ctx, httpClient, session); err != nil {
		return err
	}

	metricsDoc, err := fetchClassificationMetricsDocument(ctx, httpClient, session.URL("/metrics/classification"))
	if err != nil {
		return err
	}
	combinedKeys, err := fetchCombinedClassificationKeys(
		ctx,
		httpClient,
		session.URL("/api/v1/classify/combined"),
		map[string]string{"text": "Briefly explain what an API is."},
	)
	if err != nil {
		return err
	}
	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"decision_count":     metricsDoc.DecisionCount,
			"signal_group_count": metricsDoc.SignalGroupCount,
			"immutable_config":   true,
			"combined_keys":      combinedKeys,
		})
	}

	return nil
}

func assertStandaloneConfigRoutesAbsent(
	ctx context.Context,
	httpClient *http.Client,
	session *fixtures.ServiceSession,
) error {
	tests := []struct {
		method string
		path   string
	}{
		{http.MethodGet, "/config/router"},
		{http.MethodPost, "/config/router/validate"},
		{http.MethodPatch, "/config/router"},
		{http.MethodPut, "/config/router"},
		{http.MethodPost, "/config/router/rollback"},
		{http.MethodGet, "/config/router/versions"},
		{http.MethodGet, "/config/router/recipes"},
		{http.MethodPost, "/config/router/recipes/validate"},
		{http.MethodGet, "/config/router/recipes/example"},
		{http.MethodPut, "/config/router/recipes/example"},
		{http.MethodDelete, "/config/router/recipes/example"},
		{http.MethodGet, "/config/hash"},
		{http.MethodGet, "/config/kbs"},
		{http.MethodPost, "/config/kbs"},
		{http.MethodGet, "/config/kbs/example"},
		{http.MethodPut, "/config/kbs/example"},
		{http.MethodDelete, "/config/kbs/example"},
	}
	for _, test := range tests {
		response, err := postJSON(ctx, httpClient, test.method, session.URL(test.path), []byte(`{}`))
		if err != nil {
			return err
		}
		if response.StatusCode != http.StatusNotFound {
			return fmt.Errorf("expected removed route %s %s to return 404, got %d: %s", test.method, test.path, response.StatusCode, string(response.Body))
		}
	}
	return nil
}

func fetchClassificationMetricsDocument(
	ctx context.Context,
	httpClient *http.Client,
	url string,
) (*classificationMetricsDocument, error) {
	resp, err := getJSON(ctx, httpClient, url)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("expected /metrics/classification status 200, got %d: %s", resp.StatusCode, string(resp.Body))
	}

	var doc classificationMetricsDocument
	if err := json.Unmarshal(resp.Body, &doc); err != nil {
		return nil, fmt.Errorf("decode /metrics/classification response: %w", err)
	}
	return &doc, nil
}

func fetchCombinedClassificationKeys(
	ctx context.Context,
	httpClient *http.Client,
	url string,
	payload map[string]string,
) ([]string, error) {
	body, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("marshal /api/v1/classify/combined payload: %w", err)
	}

	resp, err := postJSON(ctx, httpClient, http.MethodPost, url, body)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("expected /api/v1/classify/combined status 200, got %d: %s", resp.StatusCode, string(resp.Body))
	}

	var document map[string]json.RawMessage
	if err := json.Unmarshal(resp.Body, &document); err != nil {
		return nil, fmt.Errorf("decode /api/v1/classify/combined response: %w", err)
	}
	keys := []string{"intent", "pii", "security", "processing_time_ms"}
	for _, key := range keys {
		if _, ok := document[key]; !ok {
			return nil, fmt.Errorf("expected /api/v1/classify/combined response to include %q", key)
		}
	}
	return keys, nil
}

func postJSON(
	ctx context.Context,
	httpClient *http.Client,
	method string,
	url string,
	body []byte,
) (*httpResponse, error) {
	req, err := http.NewRequestWithContext(ctx, method, url, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("create %s request %s: %w", method, url, err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("send %s request %s: %w", method, url, err)
	}
	defer func() {
		_ = resp.Body.Close()
	}()

	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read %s response %s: %w", method, url, err)
	}

	return &httpResponse{
		StatusCode: resp.StatusCode,
		Body:       data,
	}, nil
}
