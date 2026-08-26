package testcases

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math/big"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
	stackgateway "github.com/vllm-project/semantic-router/e2e/pkg/stacks/gateway"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
	"k8s.io/client-go/kubernetes"
)

const (
	managedAccessRequestLimit      = int64(12)
	managedAccessContractTimeout   = 8 * time.Minute
	managedAccessSettlementTimeout = 90 * time.Second
)

type managedAccessInvocation struct {
	status int
	input  int64
	output int64
	err    error
}

type managedAccessKeyDetail struct {
	Data struct {
		KeyID      string     `json:"keyId"`
		Status     string     `json:"status"`
		Revision   uint64     `json:"revision"`
		LastUsedAt *time.Time `json:"lastUsedAt,omitempty"`
	} `json:"data"`
}

type managedAccessQuotaMeter struct {
	Metric       string  `json:"metric"`
	Algorithm    string  `json:"algorithm"`
	Accounting   string  `json:"accounting"`
	Enforcement  string  `json:"enforcement"`
	Currency     string  `json:"currency,omitempty"`
	Limit        string  `json:"limit"`
	Used         string  `json:"used"`
	Remaining    *string `json:"remaining"`
	Completeness string  `json:"completeness"`
	Capacity     string  `json:"capacityState"`
}

type managedAccessQuota struct {
	Meters []managedAccessQuotaMeter `json:"meters"`
}

type managedAccessUsage struct {
	Totals struct {
		Requests           string `json:"requests"`
		SuccessfulRequests string `json:"successfulRequests"`
		InputTokens        string `json:"inputTokens"`
		OutputTokens       string `json:"outputTokens"`
		TotalTokens        string `json:"totalTokens"`
		Completeness       string `json:"completeness"`
		Costs              []struct {
			Currency     string `json:"currency"`
			KnownAmount  string `json:"knownAmount"`
			Completeness string `json:"completeness"`
		} `json:"costs"`
	} `json:"totals"`
	Grain string `json:"grain"`
}

func init() {
	pkgtestcases.Register("dashboard-managed-access-lifecycle", pkgtestcases.TestCase{
		Description: "Public Management resources enforce discovery, global quota, actual settlement, and credential lifecycle",
		Tags:        []string{"dashboard", "managed-access", "authorization", "quota", "usage", "ha"},
		Fn:          testDashboardManagedAccessLifecycle,
	})
}

func testDashboardManagedAccessLifecycle(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
) error {
	if opts.Profile != "dashboard" {
		return fmt.Errorf("dashboard-managed-access-lifecycle requires the dashboard E2E profile")
	}
	if err := waitDeploymentReadyReplicas(
		ctx, client, managedAccessNamespace, "semantic-router", 2, 5*time.Minute, opts.Verbose,
	); err != nil {
		return fmt.Errorf("managed-access contract requires two ready Router replicas: %w", err)
	}
	contractContext, cancel := context.WithTimeout(ctx, managedAccessContractTimeout)
	defer cancel()
	startedAt := time.Now().UTC()

	dashboardPort, stopDashboard, err := setupServiceConnection(contractContext, client, opts)
	if err != nil {
		return err
	}
	defer stopDashboard()
	replicaSessions, closeReplicaSessions, err := openManagedAccessReplicaSessions(
		contractContext, client, opts,
	)
	if err != nil {
		return err
	}
	defer closeReplicaSessions()
	dashboardURL := "http://127.0.0.1:" + dashboardPort
	dashboardClient := &http.Client{Timeout: 45 * time.Second}
	management, closeManagement, err := openManagedAccessClient(
		contractContext, client, opts, dashboardClient, dashboardURL,
	)
	if err != nil {
		return err
	}
	defer closeManagement()
	fixture, err := createManagedAccessFixture(contractContext, management)
	if err != nil {
		return err
	}
	defer func() {
		fixture.secret = ""
	}()
	publicationRevision, err := waitManagedAccessReplicaConvergence(
		contractContext, management, fixture.namespaceID, 0,
	)
	if err != nil {
		return err
	}

	gatewayOptions := opts
	gatewayOptions.ServiceConfig = stackgateway.DefaultServiceConfig()
	gateway, err := fixtures.OpenServiceSession(contractContext, client, gatewayOptions)
	if err != nil {
		return fmt.Errorf("open public Router gateway: %w", err)
	}
	defer gateway.Close()
	gatewayClient := gateway.HTTPClient(45 * time.Second)
	lifecycle := managedAccessLifecycle{
		ctx: contractContext, management: management, fixture: &fixture,
		replicaSessions: replicaSessions, gatewayClient: gatewayClient, gatewayBaseURL: gateway.BaseURL(),
		publicationRevision: publicationRevision, startedAt: startedAt, setDetails: opts.SetDetails,
	}
	return lifecycle.run()
}

func managedAccessModelsRequest(
	ctx context.Context,
	client *http.Client,
	baseURL string,
	secret string,
) (int, []string, error) {
	request, err := http.NewRequestWithContext(
		ctx, http.MethodGet, strings.TrimRight(baseURL, "/")+"/v1/models", nil,
	)
	if err != nil {
		return 0, nil, fmt.Errorf("create /v1/models request: %w", err)
	}
	request.Header.Set("Accept", "application/json")
	if secret != "" {
		request.Header.Set("Authorization", "Bearer "+secret)
	}
	response, err := client.Do(request)
	if err != nil {
		return 0, nil, fmt.Errorf("request /v1/models: %w", err)
	}
	defer func() { _ = response.Body.Close() }()
	body, err := io.ReadAll(io.LimitReader(response.Body, 1<<20))
	if err != nil {
		return response.StatusCode, nil, fmt.Errorf("read /v1/models response: %w", err)
	}
	defer clear(body)
	if response.StatusCode != http.StatusOK {
		return response.StatusCode, nil, nil
	}
	var result openAIModelsResponse
	if err := json.Unmarshal(body, &result); err != nil {
		return response.StatusCode, nil, fmt.Errorf("decode /v1/models response: %w", err)
	}
	if result.Object != "list" {
		return response.StatusCode, nil, fmt.Errorf("/v1/models object = %q, want list", result.Object)
	}
	models := make([]string, 0, len(result.Data))
	for _, model := range result.Data {
		models = append(models, model.ID)
	}
	return response.StatusCode, models, nil
}

func waitManagedAccessDiscovery(
	ctx context.Context,
	client *http.Client,
	baseURL string,
	secret string,
	authorized string,
	hidden string,
) error {
	deadline := time.Now().Add(90 * time.Second)
	var lastStatus int
	var lastModels []string
	for time.Now().Before(deadline) {
		status, models, err := managedAccessModelsRequest(ctx, client, baseURL, secret)
		if err != nil {
			return err
		}
		lastStatus, lastModels = status, models
		if status == http.StatusOK {
			if dashboardBuilderContains(models, hidden) {
				return fmt.Errorf("authorized /v1/models leaked hidden Entrypoint %q", hidden)
			}
			if len(models) == 1 && models[0] == authorized {
				return nil
			}
		}
		timer := time.NewTimer(time.Second)
		select {
		case <-ctx.Done():
			timer.Stop()
			return ctx.Err()
		case <-timer.C:
		}
	}
	return fmt.Errorf(
		"authorized discovery did not converge: status=%d models=%v want [%s]",
		lastStatus, lastModels, authorized,
	)
}

func waitManagedAccessCredentialDenied(
	ctx context.Context,
	client *http.Client,
	baseURL string,
	secret string,
) error {
	deadline := time.Now().Add(30 * time.Second)
	lastStatus := 0
	consecutiveUnauthorized := 0
	for time.Now().Before(deadline) {
		status, _, err := managedAccessModelsRequest(ctx, client, baseURL, secret)
		if err != nil {
			return err
		}
		lastStatus = status
		switch status {
		case http.StatusUnauthorized:
			consecutiveUnauthorized++
			if consecutiveUnauthorized == 3 {
				return nil
			}
		case http.StatusOK, http.StatusServiceUnavailable:
			consecutiveUnauthorized = 0
		default:
			return fmt.Errorf("credential transition returned status %d", status)
		}
		timer := time.NewTimer(250 * time.Millisecond)
		select {
		case <-ctx.Done():
			timer.Stop()
			return ctx.Err()
		case <-timer.C:
		}
	}
	return fmt.Errorf("credential status did not converge to 401 (last status %d)", lastStatus)
}

func invokeManagedAccessModel(
	ctx context.Context,
	client *http.Client,
	baseURL string,
	secret string,
	model string,
	ordinal int,
) managedAccessInvocation {
	payload, err := json.Marshal(map[string]interface{}{
		"model": model,
		"messages": []map[string]string{{
			"role": "user", "content": fmt.Sprintf("Reply with the number %d.", ordinal),
		}},
		"max_tokens": 16,
		"stream":     false,
	})
	if err != nil {
		return managedAccessInvocation{err: fmt.Errorf("marshal managed-access invocation: %w", err)}
	}
	defer clear(payload)
	request, err := http.NewRequestWithContext(
		ctx, http.MethodPost, strings.TrimRight(baseURL, "/")+"/v1/chat/completions", bytes.NewReader(payload),
	)
	if err != nil {
		return managedAccessInvocation{err: fmt.Errorf("create managed-access invocation: %w", err)}
	}
	request.Header.Set("Accept", "application/json")
	request.Header.Set("Authorization", "Bearer "+secret)
	request.Header.Set("Content-Type", "application/json")
	response, err := client.Do(request)
	if err != nil {
		return managedAccessInvocation{err: fmt.Errorf("invoke managed-access model: %w", err)}
	}
	defer func() { _ = response.Body.Close() }()
	body, err := io.ReadAll(io.LimitReader(response.Body, 4<<20))
	if err != nil {
		return managedAccessInvocation{status: response.StatusCode, err: fmt.Errorf("read managed-access response: %w", err)}
	}
	defer clear(body)
	result := managedAccessInvocation{status: response.StatusCode}
	if response.StatusCode == http.StatusTooManyRequests {
		return result
	}
	if response.StatusCode != http.StatusOK {
		result.err = fmt.Errorf(
			"managed-access invocation status = %d: %s",
			response.StatusCode, truncateString(string(body), 240),
		)
		return result
	}
	var completion struct {
		Choices []json.RawMessage `json:"choices"`
		Usage   struct {
			PromptTokens     int64 `json:"prompt_tokens"`
			CompletionTokens int64 `json:"completion_tokens"`
			TotalTokens      int64 `json:"total_tokens"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(body, &completion); err != nil {
		result.err = fmt.Errorf("decode managed-access response: %w", err)
		return result
	}
	if len(completion.Choices) == 0 || completion.Usage.PromptTokens <= 0 ||
		completion.Usage.CompletionTokens <= 0 ||
		completion.Usage.TotalTokens != completion.Usage.PromptTokens+completion.Usage.CompletionTokens {
		result.err = fmt.Errorf("managed-access response omitted authoritative usage")
		return result
	}
	result.input = completion.Usage.PromptTokens
	result.output = completion.Usage.CompletionTokens
	return result
}

func waitManagedAccessRequestMeter(
	ctx context.Context,
	client *managedAccessClient,
	namespaceID string,
	keyID string,
) (managedAccessQuota, managedAccessQuotaMeter, error) {
	deadline := time.Now().Add(30 * time.Second)
	for time.Now().Before(deadline) {
		quota, err := readManagedAccessQuota(ctx, client, namespaceID, keyID)
		if err != nil {
			return managedAccessQuota{}, managedAccessQuotaMeter{}, err
		}
		for _, meter := range quota.Meters {
			if meter.Metric == "requests" && meter.Algorithm == "sliding_log" &&
				meter.Accounting == "request" && meter.Enforcement == "enforce" &&
				meter.Limit == strconv.FormatInt(managedAccessRequestLimit, 10) {
				return quota, meter, nil
			}
		}
		timer := time.NewTimer(time.Second)
		select {
		case <-ctx.Done():
			timer.Stop()
			return managedAccessQuota{}, managedAccessQuotaMeter{}, ctx.Err()
		case <-timer.C:
		}
	}
	return managedAccessQuota{}, managedAccessQuotaMeter{}, fmt.Errorf("live request quota meter did not become visible")
}

func waitManagedAccessRequestBoundary(
	ctx context.Context,
	client *managedAccessClient,
	namespaceID string,
	keyID string,
) (managedAccessQuota, error) {
	deadline := time.Now().Add(10 * time.Second)
	var lastErr error
	for time.Now().Before(deadline) {
		quota, err := readManagedAccessQuota(ctx, client, namespaceID, keyID)
		if err != nil {
			return managedAccessQuota{}, err
		}
		lastErr = assertManagedAccessRequestQuota(quota)
		if lastErr == nil {
			return quota, nil
		}
		timer := time.NewTimer(250 * time.Millisecond)
		select {
		case <-ctx.Done():
			timer.Stop()
			return managedAccessQuota{}, ctx.Err()
		case <-timer.C:
		}
	}
	return managedAccessQuota{}, fmt.Errorf("global RPM boundary did not converge: %w", lastErr)
}

func readManagedAccessQuota(
	ctx context.Context,
	client *managedAccessClient,
	namespaceID string,
	keyID string,
) (managedAccessQuota, error) {
	var quota managedAccessQuota
	if _, _, err := client.request(
		ctx, namespaceID, http.MethodGet, "/api-keys/"+keyID+"/quota", "", nil, nil,
		[]int{http.StatusOK}, &quota,
	); err != nil {
		return managedAccessQuota{}, fmt.Errorf("read live API-key quota: %w", err)
	}
	return quota, nil
}

func waitManagedAccessSettlement(
	ctx context.Context,
	client *managedAccessClient,
	fixture managedAccessFixture,
	startedAt time.Time,
	expectedRequests int64,
	expectedInput int64,
	expectedOutput int64,
	expectedCost int64,
) (managedAccessUsage, managedAccessQuota, error) {
	deadline := time.Now().Add(managedAccessSettlementTimeout)
	var lastReason string
	for time.Now().Before(deadline) {
		var detail managedAccessKeyDetail
		if _, _, err := client.request(
			ctx, fixture.namespaceID, http.MethodGet, "/api-keys/"+fixture.keyID,
			"", nil, nil, []int{http.StatusOK}, &detail,
		); err != nil {
			return managedAccessUsage{}, managedAccessQuota{}, err
		}
		usage, err := readManagedAccessUsage(ctx, client, fixture, startedAt)
		if err != nil {
			return managedAccessUsage{}, managedAccessQuota{}, err
		}
		quota, err := readManagedAccessQuota(ctx, client, fixture.namespaceID, fixture.keyID)
		if err != nil {
			return managedAccessUsage{}, managedAccessQuota{}, err
		}
		lastReason = managedAccessSettlementReason(
			detail, usage, quota, expectedRequests, expectedInput, expectedOutput, expectedCost,
		)
		if lastReason == "" {
			return usage, quota, nil
		}
		timer := time.NewTimer(time.Second)
		select {
		case <-ctx.Done():
			timer.Stop()
			return managedAccessUsage{}, managedAccessQuota{}, ctx.Err()
		case <-timer.C:
		}
	}
	return managedAccessUsage{}, managedAccessQuota{}, fmt.Errorf(
		"actual token/cost settlement did not converge: %s", lastReason,
	)
}

func readManagedAccessUsage(
	ctx context.Context,
	client *managedAccessClient,
	fixture managedAccessFixture,
	startedAt time.Time,
) (managedAccessUsage, error) {
	query := url.Values{}
	query.Set("grain", "minute")
	query.Set("start", startedAt.Add(-time.Minute).Format(time.RFC3339Nano))
	query.Set("end", time.Now().UTC().Format(time.RFC3339Nano))
	var usage managedAccessUsage
	if _, _, err := client.request(
		ctx, fixture.namespaceID, http.MethodGet,
		"/api-keys/"+fixture.keyID+"/usage?"+query.Encode(), "", nil, nil,
		[]int{http.StatusOK}, &usage,
	); err != nil {
		return managedAccessUsage{}, fmt.Errorf("read durable API-key Usage: %w", err)
	}
	return usage, nil
}

func managedAccessSettlementReason(
	detail managedAccessKeyDetail,
	usage managedAccessUsage,
	quota managedAccessQuota,
	expectedRequests int64,
	expectedInput int64,
	expectedOutput int64,
	expectedCost int64,
) string {
	if detail.Data.Status != "active" || detail.Data.LastUsedAt == nil {
		return "API-key detail has not observed last use"
	}
	if reason := managedAccessUsageSummaryReason(
		usage, expectedRequests, expectedInput, expectedOutput, expectedCost,
	); reason != "" {
		return reason
	}
	if err := assertManagedAccessActualQuota(quota, expectedInput+expectedOutput, expectedCost); err != nil {
		return err.Error()
	}
	return ""
}

func managedAccessUsageSummaryReason(
	usage managedAccessUsage,
	expectedRequests int64,
	expectedInput int64,
	expectedOutput int64,
	expectedCost int64,
) string {
	if usage.Grain != "minute" || usage.Totals.Completeness != "complete" {
		return "Usage is not a complete minute summary"
	}
	if reason := managedAccessUsageRequestReason(usage, expectedRequests); reason != "" {
		return reason
	}
	if reason := managedAccessUsageTokenReason(usage, expectedInput, expectedOutput); reason != "" {
		return reason
	}
	return managedAccessUsageCostReason(usage, expectedCost)
}

func managedAccessUsageRequestReason(usage managedAccessUsage, expectedRequests int64) string {
	if usage.Totals.Requests != strconv.FormatInt(expectedRequests, 10) ||
		usage.Totals.SuccessfulRequests != strconv.FormatInt(expectedRequests, 10) {
		return fmt.Sprintf(
			"Usage requests are total=%s successful=%s",
			usage.Totals.Requests, usage.Totals.SuccessfulRequests,
		)
	}
	return ""
}

func managedAccessUsageTokenReason(usage managedAccessUsage, expectedInput, expectedOutput int64) string {
	if usage.Totals.InputTokens != strconv.FormatInt(expectedInput, 10) ||
		usage.Totals.OutputTokens != strconv.FormatInt(expectedOutput, 10) ||
		usage.Totals.TotalTokens != strconv.FormatInt(expectedInput+expectedOutput, 10) {
		return fmt.Sprintf(
			"Usage tokens are input=%s output=%s total=%s",
			usage.Totals.InputTokens, usage.Totals.OutputTokens, usage.Totals.TotalTokens,
		)
	}
	return ""
}

func managedAccessUsageCostReason(usage managedAccessUsage, expectedCost int64) string {
	if len(usage.Totals.Costs) != 1 || usage.Totals.Costs[0].Currency != "USD" ||
		usage.Totals.Costs[0].Completeness != "complete" ||
		!managedAccessDecimalEquals(usage.Totals.Costs[0].KnownAmount, strconv.FormatInt(expectedCost, 10)) {
		return "Usage cost has not converged to the actual response cost"
	}
	return ""
}

func assertManagedAccessRequestQuota(quota managedAccessQuota) error {
	for _, meter := range quota.Meters {
		if meter.Metric != "requests" {
			continue
		}
		if meter.Algorithm != "sliding_log" || meter.Accounting != "request" ||
			meter.Enforcement != "enforce" || meter.Limit != strconv.FormatInt(managedAccessRequestLimit, 10) ||
			meter.Used != strconv.FormatInt(managedAccessRequestLimit, 10) ||
			meter.Remaining == nil || *meter.Remaining != "0" ||
			meter.Completeness != "complete" || meter.Capacity != "exhausted" {
			return fmt.Errorf("request meter does not show the exhausted global RPM boundary")
		}
		return nil
	}
	return fmt.Errorf("live quota omitted requests meter")
}

func assertManagedAccessActualQuota(
	quota managedAccessQuota,
	expectedTokens int64,
	expectedCost int64,
) error {
	found := map[string]bool{"served_total_tokens": false, "cost": false}
	for _, meter := range quota.Meters {
		relevant, err := validateManagedAccessActualMeter(meter, expectedTokens, expectedCost)
		if err != nil {
			return err
		}
		if relevant {
			found[meter.Metric] = true
		}
	}
	for metric, present := range found {
		if !present {
			return fmt.Errorf("live quota omitted %s meter", metric)
		}
	}
	return nil
}

func validateManagedAccessActualMeter(
	meter managedAccessQuotaMeter,
	expectedTokens int64,
	expectedCost int64,
) (bool, error) {
	switch meter.Metric {
	case "served_total_tokens":
		if !managedAccessActualMeterShapeMatches(meter, "") ||
			meter.Used != strconv.FormatInt(expectedTokens, 10) {
			return true, fmt.Errorf("served-token meter does not match actual response usage")
		}
		return true, nil
	case "cost":
		if !managedAccessActualMeterShapeMatches(meter, "USD") ||
			!managedAccessDecimalEquals(meter.Used, strconv.FormatInt(expectedCost, 10)) {
			return true, fmt.Errorf("cost meter does not match actual response pricing")
		}
		return true, nil
	default:
		return false, nil
	}
}

func managedAccessActualMeterShapeMatches(meter managedAccessQuotaMeter, currency string) bool {
	commonShape := meter.Algorithm == "calendar_window" && meter.Accounting == "response_actual" &&
		meter.Enforcement == "enforce" && meter.Limit == "100000" &&
		meter.Completeness == "complete" && meter.Capacity == "available"
	return commonShape && (currency == "" || meter.Currency == currency)
}

func managedAccessDecimalEquals(left string, right string) bool {
	leftValue, leftOK := new(big.Rat).SetString(left)
	rightValue, rightOK := new(big.Rat).SetString(right)
	return leftOK && rightOK && leftValue.Cmp(rightValue) == 0
}
