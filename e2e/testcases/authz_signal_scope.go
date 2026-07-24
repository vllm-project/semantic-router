package testcases

import (
	"bufio"
	"context"
	"fmt"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
	"k8s.io/client-go/kubernetes"
)

const (
	authzSignalScopeName   = "admin_only_marker"
	authzSignalScopePrompt = "authz-scope-marker-2647"
	authzSignalScopeModel  = "Qwen/Qwen2.5-7B-Instruct"
	authzSignalScopeMetric = "llm_signal_extraction_total"
	authzSignalScopeType   = "keyword"
)

func init() {
	pkgtestcases.Register("authz-signal-scope", pkgtestcases.TestCase{
		Description: "Verify signals used only by authz-ineligible decisions are not evaluated",
		Tags:        []string{"authz-rbac", "routing", "metrics", "regression"},
		Fn:          testAuthzSignalScope,
	})
}

func testAuthzSignalScope(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
) error {
	traffic, err := fixtures.OpenServiceSession(ctx, client, opts)
	if err != nil {
		return err
	}
	defer traffic.Close()

	metricsSession, err := fixtures.OpenSemanticRouterMetricsSession(ctx, client, opts)
	if err != nil {
		return err
	}
	defer metricsSession.Close()

	metricsBefore, err := fetchMetrics(ctx, metricsSession)
	if err != nil {
		return err
	}
	if !strings.Contains(metricsBefore, "# HELP "+authzSignalScopeMetric+" ") {
		return fmt.Errorf("metrics body missing %s descriptor", authzSignalScopeMetric)
	}
	before, err := signalExtractionMetricValue(metricsBefore, authzSignalScopeType, authzSignalScopeName)
	if err != nil {
		return fmt.Errorf("read signal metric before request: %w", err)
	}

	chat := fixtures.NewChatCompletionsClient(traffic, 30*time.Second)
	resp, err := chat.Create(ctx, fixtures.ChatCompletionsRequest{
		Model: "MoM",
		Messages: []fixtures.ChatMessage{
			{Role: "user", Content: authzSignalScopePrompt},
		},
	}, nil)
	if err != nil {
		return fmt.Errorf("send unauthenticated marker request: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("marker request: expected 200, got %d: %s", resp.StatusCode, string(resp.Body))
	}
	if decision := resp.Headers.Get("x-vsr-selected-decision"); decision != "" {
		return fmt.Errorf("marker request selected authz-protected decision %q", decision)
	}

	var responseBody struct {
		Model string `json:"model"`
	}
	if err := resp.DecodeJSON(&responseBody); err != nil {
		return fmt.Errorf("decode marker response: %w", err)
	}
	if responseBody.Model != authzSignalScopeModel {
		return fmt.Errorf("marker request model: got %q, want %q", responseBody.Model, authzSignalScopeModel)
	}

	metricsAfter, err := fetchMetrics(ctx, metricsSession)
	if err != nil {
		return err
	}
	if !strings.Contains(metricsAfter, "# HELP "+authzSignalScopeMetric+" ") {
		return fmt.Errorf("metrics body missing %s descriptor after request", authzSignalScopeMetric)
	}
	after, err := signalExtractionMetricValue(metricsAfter, authzSignalScopeType, authzSignalScopeName)
	if err != nil {
		return fmt.Errorf("read signal metric after request: %w", err)
	}
	if delta := after - before; delta != 0 {
		return fmt.Errorf("authz-ineligible signal %q was evaluated: metric delta=%v", authzSignalScopeName, delta)
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"signal_name":   authzSignalScopeName,
			"metric_before": before,
			"metric_after":  after,
			"metric_delta":  after - before,
			"model":         responseBody.Model,
		})
	}

	return nil
}

func signalExtractionMetricValue(metrics, signalType, signalName string) (float64, error) {
	prefix := authzSignalScopeMetric + "{"
	typeLabel := `signal_type="` + signalType + `"`
	nameLabel := `signal_name="` + signalName + `"`

	scanner := bufio.NewScanner(strings.NewReader(metrics))
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if !strings.HasPrefix(line, prefix) {
			continue
		}

		closingBrace := strings.IndexByte(line, '}')
		if closingBrace < len(prefix) {
			continue
		}

		labels := strings.Split(line[len(prefix):closingBrace], ",")
		hasType := false
		hasName := false
		for _, label := range labels {
			switch strings.TrimSpace(label) {
			case typeLabel:
				hasType = true
			case nameLabel:
				hasName = true
			}
		}
		if !hasType || !hasName {
			continue
		}

		fields := strings.Fields(line[closingBrace+1:])
		if len(fields) == 0 {
			return 0, fmt.Errorf("metric %s has no sample value", authzSignalScopeMetric)
		}
		value, err := strconv.ParseFloat(fields[0], 64)
		if err != nil {
			return 0, fmt.Errorf("parse %s sample %q: %w", authzSignalScopeMetric, fields[0], err)
		}
		return value, nil
	}
	if err := scanner.Err(); err != nil {
		return 0, fmt.Errorf("scan Prometheus metrics: %w", err)
	}

	return 0, nil
}
