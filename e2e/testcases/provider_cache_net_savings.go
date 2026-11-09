package testcases

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"net/url"
	"strings"
	"time"

	"k8s.io/client-go/kubernetes"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

func init() {
	pkgtestcases.Register("provider-cache-net-savings", pkgtestcases.TestCase{
		Description: "The Replay aggregate nets provider prompt-cache savings over a session without clamping a cache-write loss at zero",
		Tags:        []string{"router-replay", "anthropic", "functional"},
		Fn:          testProviderCacheNetSavings,
	})
}

type providerCacheNetSpend struct {
	Name          string  `json:"name"`
	Currency      string  `json:"currency"`
	NoCacheSpend  float64 `json:"no_cache_spend"`
	ActualSpend   float64 `json:"actual_spend"`
	NetSavings    float64 `json:"net_savings"`
	PricedRecords int     `json:"priced_records"`
}

type providerCacheNetSummary struct {
	Currency        string                  `json:"currency"`
	NoCacheSpend    float64                 `json:"no_cache_spend"`
	ActualSpend     float64                 `json:"actual_spend"`
	NetSavings      float64                 `json:"net_savings"`
	PricedRecords   int                     `json:"priced_records"`
	ExcludedRecords int                     `json:"excluded_records"`
	ByDecision      []providerCacheNetSpend `json:"by_decision"`
	BySelectedModel []providerCacheNetSpend `json:"by_selected_model"`
}

func testProviderCacheNetSavings(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	session, err := fixtures.OpenServiceSession(ctx, client, opts)
	if err != nil {
		return err
	}
	defer session.Close()
	management, err := fixtures.OpenRouterAPISession(ctx, client, opts)
	if err != nil {
		return err
	}
	defer management.Close()
	id := fmt.Sprintf("cache-net-%d", time.Now().UnixNano())
	chat := protocolCodecE2EClients[0]
	// The read-heavy request saves 0.00009; the write-heavy one loses 0.00035.
	for _, prompt := range []string{
		"__mock_provider_cache_cost__ __mock_cache_read_heavy__",
		"__mock_provider_cache_cost__",
	} {
		response, sendErr := sendProtocolMatrixRaw(ctx, session, chat.path, chat.request("vllm-sr/auto", prompt, false), false, map[string]string{"x-session-id": id})
		if sendErr != nil {
			return sendErr
		}
		if response.StatusCode != http.StatusOK {
			return fmt.Errorf("%s returned HTTP %d", id, response.StatusCode)
		}
	}
	summary, err := waitProviderCacheNetSummary(ctx, management, id, 2)
	if err != nil {
		return err
	}
	return assertProviderCacheNetSummary(summary)
}

func waitProviderCacheNetSummary(ctx context.Context, session *fixtures.ServiceSession, id string, records int) (providerCacheNetSummary, error) {
	deadline := time.NewTimer(10 * time.Second)
	defer deadline.Stop()
	for {
		response, err := doRouterReplayManagementGET(ctx, session, "/api/v1/observability/replays/aggregate?session_id="+url.QueryEscape(id))
		if err != nil {
			return providerCacheNetSummary{}, err
		}
		if response.StatusCode != http.StatusOK {
			return providerCacheNetSummary{}, fmt.Errorf("replay aggregate HTTP %d", response.StatusCode)
		}
		var aggregate struct {
			ProviderCache providerCacheNetSummary `json:"provider_cache"`
		}
		if err := json.Unmarshal(response.Body, &aggregate); err != nil {
			return providerCacheNetSummary{}, err
		}
		if aggregate.ProviderCache.PricedRecords >= records {
			return aggregate.ProviderCache, nil
		}
		select {
		case <-ctx.Done():
			return providerCacheNetSummary{}, ctx.Err()
		case <-deadline.C:
			return providerCacheNetSummary{}, fmt.Errorf("aggregate for %s priced %d of %d records", id, aggregate.ProviderCache.PricedRecords, records)
		case <-time.After(100 * time.Millisecond):
		}
	}
}

func assertProviderCacheNetSummary(summary providerCacheNetSummary) error {
	if summary.PricedRecords != 2 || summary.ExcludedRecords != 0 || summary.Currency != "USD" {
		return fmt.Errorf("unexpected settled records: priced=%d excluded=%d currency=%q", summary.PricedRecords, summary.ExcludedRecords, summary.Currency)
	}
	// Clamping each request at zero would report +0.00009.
	for _, check := range []struct {
		name      string
		got, want float64
	}{
		{"no_cache_spend", summary.NoCacheSpend, .00044},
		{"actual_spend", summary.ActualSpend, .0007},
		{"net_savings", summary.NetSavings, -.00026},
	} {
		if math.Abs(check.got-check.want) > 1e-12 {
			return fmt.Errorf("%s=%v, want %v", check.name, check.got, check.want)
		}
	}
	for _, split := range []struct {
		name    string
		entries []providerCacheNetSpend
		owner   func(string) bool
	}{
		{"decision", summary.ByDecision, func(name string) bool { return strings.HasSuffix(name, "provider_cache_cost") }},
		{"selected model", summary.BySelectedModel, func(name string) bool { return name == "MoM" }},
	} {
		if len(split.entries) != 1 || !split.owner(split.entries[0].Name) || split.entries[0].PricedRecords != 2 ||
			math.Abs(split.entries[0].NetSavings-summary.NetSavings) > 1e-12 {
			return fmt.Errorf("%s split does not add up to the total: %+v", split.name, split.entries)
		}
	}
	return nil
}
