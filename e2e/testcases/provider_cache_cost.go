package testcases

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"net/url"
	"time"

	"k8s.io/client-go/kubernetes"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

func init() {
	pkgtestcases.Register("provider-cache-cost-breakdown", pkgtestcases.TestCase{
		Description: "Provider cache component costs and same-model savings survive buffered and streaming Replay management APIs",
		Tags:        []string{"router-replay", "anthropic", "streaming", "functional"},
		Fn:          testProviderCacheCostBreakdown,
	})
}

type providerCacheComponent struct {
	Tokens     *int     `json:"tokens"`
	Provenance string   `json:"provenance"`
	Cost       *float64 `json:"cost"`
}

type providerCacheReplayRecord struct {
	LifecycleState string   `json:"lifecycle_state"`
	ActualCost     *float64 `json:"actual_cost"`
	RoutingSavings *float64 `json:"cost_savings"`
	ProviderCache  *struct {
		State         string                 `json:"state"`
		UncachedInput providerCacheComponent `json:"uncached_input"`
		CacheRead     providerCacheComponent `json:"cache_read"`
		CacheWrite    providerCacheComponent `json:"cache_write"`
		Output        providerCacheComponent `json:"output"`
		Write5m       providerCacheComponent `json:"cache_write_5m"`
		Write1h       providerCacheComponent `json:"cache_write_1h"`
		NoCacheCost   *float64               `json:"no_cache_cost"`
		CacheSavings  *float64               `json:"cache_savings"`
	} `json:"provider_cache"`
}

func testProviderCacheCostBreakdown(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
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
	for _, protocol := range protocolCodecE2EClients {
		for _, stream := range []bool{false, true} {
			for _, tiers := range []bool{false, true} {
				id := fmt.Sprintf("cache-cost-%s-%t-%t-%d", protocol.name, stream, tiers, time.Now().UnixNano())
				prompt := "__mock_provider_cache_cost__"
				if !tiers {
					prompt += " __mock_cache_aggregate__"
				}
				response, err := sendProtocolMatrixRaw(ctx, session, protocol.path, protocol.request("vllm-sr/auto", prompt, stream), stream, map[string]string{"x-session-id": id})
				if err != nil {
					return err
				}
				if response.StatusCode != http.StatusOK {
					return fmt.Errorf("%s returned HTTP %d", id, response.StatusCode)
				}
				record, err := waitProviderCacheReplay(ctx, management, id)
				if err != nil {
					return err
				}
				if err := assertProviderCacheCosts(record, tiers); err != nil {
					return fmt.Errorf("%s: %w", id, err)
				}
			}
		}
	}
	return nil
}

func waitProviderCacheReplay(ctx context.Context, session *fixtures.ServiceSession, id string) (providerCacheReplayRecord, error) {
	deadline := time.NewTimer(10 * time.Second)
	defer deadline.Stop()
	for {
		response, err := doRouterReplayManagementGET(ctx, session, "/api/v1/observability/replays?session_id="+url.QueryEscape(id))
		if err != nil {
			return providerCacheReplayRecord{}, err
		}
		if response.StatusCode != http.StatusOK {
			return providerCacheReplayRecord{}, fmt.Errorf("Replay HTTP %d", response.StatusCode)
		}
		var list struct {
			Data []providerCacheReplayRecord `json:"data"`
		}
		if err := json.Unmarshal(response.Body, &list); err != nil {
			return providerCacheReplayRecord{}, err
		}
		for _, record := range list.Data {
			if record.LifecycleState == "completed" && record.ProviderCache != nil {
				return record, nil
			}
		}
		select {
		case <-ctx.Done():
			return providerCacheReplayRecord{}, ctx.Err()
		case <-deadline.C:
			return providerCacheReplayRecord{}, fmt.Errorf("no terminal provider cache Replay for %s", id)
		case <-time.After(100 * time.Millisecond):
		}
	}
}

func assertProviderCacheCosts(record providerCacheReplayRecord, tiers bool) error {
	accounting := record.ProviderCache
	if accounting == nil || accounting.State != "complete" {
		return fmt.Errorf("provider accounting is incomplete")
	}
	writeCost := .00032
	if tiers {
		writeCost = .00052
	}
	for _, check := range []struct {
		component providerCacheComponent
		tokens    int
		cost      float64
	}{
		{accounting.UncachedInput, 10, .00002}, {accounting.CacheRead, 10, .00001},
		{accounting.CacheWrite, 80, writeCost}, {accounting.Output, 2, .00002},
	} {
		if check.component.Tokens == nil || *check.component.Tokens != check.tokens {
			return fmt.Errorf("unexpected component tokens: %v", check.component.Tokens)
		}
		if check.component.Provenance != "authoritative" && check.component.Provenance != "derived" {
			return fmt.Errorf("missing component provenance")
		}
		if check.component.Cost == nil || math.Abs(*check.component.Cost-check.cost) > 1e-12 {
			return fmt.Errorf("incorrect component cost")
		}
	}
	actual := .00005 + writeCost
	for _, check := range []struct {
		name string
		got  *float64
		want float64
	}{
		{"actual", record.ActualCost, actual}, {"no-cache", accounting.NoCacheCost, .00022},
		{"cache savings", accounting.CacheSavings, .00022 - actual}, {"routing savings", record.RoutingSavings, 0},
	} {
		if check.got == nil || math.Abs(*check.got-check.want) > 1e-12 {
			return fmt.Errorf("%s=%v, want %g", check.name, check.got, check.want)
		}
	}
	if tiers {
		if accounting.Write5m.Tokens == nil || *accounting.Write5m.Tokens != 30 || accounting.Write1h.Tokens == nil || *accounting.Write1h.Tokens != 50 {
			return fmt.Errorf("tier identity lost")
		}
	} else if accounting.Write5m.Tokens != nil || accounting.Write1h.Tokens != nil {
		return fmt.Errorf("aggregate-only writes invented tier counts")
	}
	return nil
}
