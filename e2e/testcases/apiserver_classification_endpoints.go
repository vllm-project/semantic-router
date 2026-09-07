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
		Description: "Verify router config, metrics, and combined-classification endpoints against the live router runtime",
		Tags:        []string{"kubernetes", "apiserver", "classification", "config", "api"},
		Fn:          testAPIServerClassificationEndpoints,
	})
}

type routerConfigDocument struct {
	Routing struct {
		Decisions []struct {
			Name string `json:"name"`
		} `json:"decisions"`
		Signals map[string]interface{} `json:"signals"`
	} `json:"routing"`
}

type classificationMetricsDocument struct {
	DecisionCount    int  `json:"decision_count"`
	SignalGroupCount int  `json:"signal_group_count"`
	RouterConfigAPI  bool `json:"router_config_api"`
	ClassifierReady  bool `json:"classifier_ready"`
	PIIReady         bool `json:"pii_ready"`
	SecurityReady    bool `json:"security_ready"`
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
	_, configDoc, err := fetchRouterConfigDocument(ctx, httpClient, session.URL("/config/router"))
	if err != nil {
		return err
	}
	// The Kubernetes baseline intentionally mounts its source document from a
	// read-only ConfigMap. PUT/PATCH persistence and merge semantics are covered
	// by the API server's focused tests; mutating the shared live fixture here
	// would be deployment-specific and could contaminate later E2E cases.

	metricsDoc, err := fetchClassificationMetricsDocument(ctx, httpClient, session.URL("/metrics/classification"))
	if err != nil {
		return err
	}
	if metricsDoc.DecisionCount != len(configDoc.Routing.Decisions) {
		return fmt.Errorf("expected /metrics/classification decision_count=%d, got %d", len(configDoc.Routing.Decisions), metricsDoc.DecisionCount)
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
			"router_config_api":  metricsDoc.RouterConfigAPI,
			"combined_keys":      combinedKeys,
		})
	}

	return nil
}

func fetchRouterConfigDocument(
	ctx context.Context,
	httpClient *http.Client,
	url string,
) ([]byte, *routerConfigDocument, error) {
	resp, err := getJSON(ctx, httpClient, url)
	if err != nil {
		return nil, nil, err
	}
	if resp.StatusCode != http.StatusOK {
		return nil, nil, fmt.Errorf("expected /config/router status 200, got %d: %s", resp.StatusCode, string(resp.Body))
	}

	var doc routerConfigDocument
	if err := json.Unmarshal(resp.Body, &doc); err != nil {
		return nil, nil, fmt.Errorf("decode /config/router response: %w", err)
	}
	if len(doc.Routing.Decisions) == 0 {
		return nil, nil, fmt.Errorf("expected /config/router to include routing decisions")
	}
	return resp.Body, &doc, nil
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
	if !doc.RouterConfigAPI {
		return nil, fmt.Errorf("expected /metrics/classification to advertise router_config_api=true")
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
