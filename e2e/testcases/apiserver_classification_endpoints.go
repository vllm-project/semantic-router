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
		Tags:        []string{"ai-gateway", "apiserver", "classification", "config", "api"},
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
	configBody, configDoc, err := fetchRouterConfigDocument(ctx, httpClient, session.URL("/config/router"))
	if err != nil {
		return err
	}
	if err := assertRouterConfigMergeSemantics(ctx, httpClient, session.URL("/config/router"), configBody); err != nil {
		return err
	}

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
	if err := assertRouterConfigReplaceSemantics(ctx, httpClient, session.URL("/config/router"), configBody); err != nil {
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

func assertRouterConfigMergeSemantics(
	ctx context.Context,
	httpClient *http.Client,
	url string,
	originalBody []byte,
) error {
	var payload map[string]interface{}
	if err := json.Unmarshal(originalBody, &payload); err != nil {
		return fmt.Errorf("decode original /config/router document: %w", err)
	}

	routing, ok := payload["routing"].(map[string]interface{})
	if !ok {
		return fmt.Errorf("expected original /config/router document to include routing")
	}
	if _, ok := routing["projections"].(map[string]interface{}); !ok {
		return fmt.Errorf("expected original /config/router document to include routing.projections")
	}

	patchBody, err := json.Marshal(map[string]interface{}{
		"routing": map[string]interface{}{
			"decisions": routing["decisions"],
		},
	})
	if err != nil {
		return fmt.Errorf("marshal merge PATCH /config/router body: %w", err)
	}

	resp, err := postJSON(ctx, httpClient, http.MethodPatch, url, patchBody)
	if err != nil {
		return err
	}
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("expected PATCH /config/router status 200, got %d: %s", resp.StatusCode, string(resp.Body))
	}

	var updated map[string]interface{}
	if err := json.Unmarshal(resp.Body, &updated); err != nil {
		return fmt.Errorf("decode PATCH /config/router response: %w", err)
	}
	updatedRouting, ok := updated["routing"].(map[string]interface{})
	if !ok {
		return fmt.Errorf("expected PATCH /config/router response to include routing")
	}
	if _, ok := updatedRouting["projections"].(map[string]interface{}); !ok {
		return fmt.Errorf("expected PATCH /config/router to preserve omitted routing.projections")
	}
	return nil
}

func assertRouterConfigReplaceSemantics(
	ctx context.Context,
	httpClient *http.Client,
	url string,
	originalBody []byte,
) error {
	var payload map[string]interface{}
	if err := json.Unmarshal(originalBody, &payload); err != nil {
		return fmt.Errorf("decode original /config/router document: %w", err)
	}

	routing, ok := payload["routing"].(map[string]interface{})
	if !ok {
		return fmt.Errorf("expected original /config/router document to include routing")
	}
	if _, ok := routing["projections"].(map[string]interface{}); !ok {
		return fmt.Errorf("expected original /config/router document to include routing.projections")
	}

	delete(routing, "projections")
	modifiedBody, err := json.Marshal(payload)
	if err != nil {
		return fmt.Errorf("marshal modified /config/router document: %w", err)
	}

	resp, err := postJSON(ctx, httpClient, http.MethodPut, url, modifiedBody)
	if err != nil {
		return err
	}
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("expected replace PUT /config/router status 200, got %d: %s", resp.StatusCode, string(resp.Body))
	}

	var updated map[string]interface{}
	if err := json.Unmarshal(resp.Body, &updated); err != nil {
		return fmt.Errorf("decode replace PUT /config/router response: %w", err)
	}
	updatedRouting, ok := updated["routing"].(map[string]interface{})
	if !ok {
		return fmt.Errorf("expected replace PUT /config/router response to include routing")
	}
	if _, ok := updatedRouting["projections"]; ok {
		return fmt.Errorf("expected replace PUT /config/router to drop omitted routing.projections")
	}

	restoreResp, err := postJSON(ctx, httpClient, http.MethodPut, url, originalBody)
	if err != nil {
		return err
	}
	if restoreResp.StatusCode != http.StatusOK {
		return fmt.Errorf("expected restore PUT /config/router status 200, got %d: %s", restoreResp.StatusCode, string(restoreResp.Body))
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
