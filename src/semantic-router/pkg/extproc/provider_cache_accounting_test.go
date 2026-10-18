package extproc

import (
	"context"
	"encoding/json"
	"math"
	"reflect"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
)

func TestProviderCacheAccountingBufferedStreamingReplay(t *testing.T) {
	const usage = `{"input_tokens":10,"cache_read_input_tokens":10,"cache_creation_input_tokens":80,"cache_creation":{"ephemeral_5m_input_tokens":30,"ephemeral_1h_input_tokens":50},"output_tokens":2}`
	const message = `{"id":"msg_cost","type":"message","role":"assistant","model":"claude","content":[{"type":"text","text":"hello"}],"stop_reason":"end_turn","usage":` + usage + `}`
	engine := protocolcodec.NewBuiltinEngine()
	buffered, _, _, err := engine.DecodeResponse(llmprotocol.AnthropicMessagesV1, []byte(message))
	if err != nil {
		t.Fatal(err)
	}
	stream, _, err := engine.EncodeResponseStream(llmprotocol.AnthropicMessagesV1, buffered, llmprotocol.StreamContext{Context: context.Background(), PublicModel: "claude"})
	if err != nil {
		t.Fatal(err)
	}
	streamed, _, err := engine.DecodeResponseStream(llmprotocol.AnthropicMessagesV1, stream, llmprotocol.StreamContext{Context: context.Background(), PublicModel: "claude"})
	if err != nil {
		t.Fatal(err)
	}
	write, write1h := 4.0, 8.0
	pricing := config.ModelPricing{Currency: "USD", PromptPer1M: 2, CachedInputPer1M: 1, CacheWritePer1M: &write, CacheWrite1hPer1M: &write1h, CompletionPer1M: 10}
	router := &OpenAIRouter{Config: &config.RouterConfig{BackendModels: config.BackendModels{ModelConfig: map[string]config.ModelParams{"claude": {Pricing: pricing}}}}}
	var previous *store.ProviderCacheAccounting
	for _, response := range []llmprotocol.Response{buffered, streamed} {
		ctx := replayCostContext("claude", "claude")
		ctx.SemanticResponse = &response
		snapshot := router.buildReplayUsageCost(ctx, responseUsageFromSemanticUsage(response.Usage))
		accounting := snapshot.ProviderCache
		if accounting == nil || accounting.State != "complete" {
			t.Fatalf("incomplete accounting: %#v", accounting)
		}
		for _, pair := range []struct {
			got  *float64
			want float64
		}{
			{accounting.UncachedInput.Cost, .00002},
			{accounting.CacheRead.Cost, .00001},
			{accounting.CacheWrite.Cost, .00052},
			{accounting.Output.Cost, .00002},
			{accounting.NoCacheCost, .00022},
			{accounting.CacheSavings, -.00035},
			{snapshot.ActualCost, .00057},
		} {
			if pair.got == nil || math.Abs(*pair.got-pair.want) > 1e-12 {
				t.Fatalf("cost=%v, want %v", pair.got, pair.want)
			}
		}
		if snapshot.CostSavings == nil || *snapshot.CostSavings != 0 {
			t.Fatal("same-model routing savings must remain zero")
		}
		memory := store.NewMemoryStore(10, 0)
		id, err := memory.Add(context.Background(), store.Record{})
		if err != nil {
			t.Fatal(err)
		}
		if err = memory.UpdateUsageCost(context.Background(), id, snapshot); err != nil {
			t.Fatal(err)
		}
		record, _, err := memory.Get(context.Background(), id)
		if err != nil {
			t.Fatal(err)
		}
		encoded, err := json.Marshal(record)
		if err != nil {
			t.Fatal(err)
		}
		var decoded store.Record
		if err := json.Unmarshal(encoded, &decoded); err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(decoded.ProviderCache, accounting) {
			t.Fatal("Replay JSON lost provider cache accounting")
		}
		if previous != nil && !reflect.DeepEqual(previous, accounting) {
			t.Fatal("streaming and buffered accounting differ")
		}
		previous = accounting
	}
}

func TestProviderCachePartialUsagePersistsWithoutInventingTotals(t *testing.T) {
	for _, streaming := range []bool{false, true} {
		memory := store.NewMemoryStore(10, 0)
		id, err := memory.Add(context.Background(), store.Record{})
		if err != nil {
			t.Fatal(err)
		}
		router := &OpenAIRouter{Config: &config.RouterConfig{BackendModels: config.BackendModels{ModelConfig: map[string]config.ModelParams{
			"claude": {Pricing: config.ModelPricing{Currency: "USD", PromptPer1M: 2, CompletionPer1M: 10}},
		}}}, ReplayRecorder: routerreplay.NewRecorder(memory)}
		ctx := &RequestContext{RequestModel: "claude", RouterReplayID: id, SemanticResponse: &llmprotocol.Response{Usage: llmprotocol.Usage{
			State: llmprotocol.UsageAvailable, OutputTotal: llmprotocol.TokenCount{Value: llmprotocol.Int64(2), Provenance: llmprotocol.UsageAuthoritative},
		}}}
		usage := router.takeNeutralResponseUsage(ctx)
		if streaming {
			router.reportSemanticStreamingUsage(ctx, 0, usage)
		} else {
			router.reportNonStreamingUsage(ctx, 0, usage)
		}
		record, _, err := memory.Get(context.Background(), id)
		if err != nil {
			t.Fatal(err)
		}
		if record.ProviderCache == nil || record.ProviderCache.State != "partial" || record.ProviderCache.Output.Cost == nil {
			t.Fatal("partial provider evidence was not persisted")
		}
		if record.ActualCost != nil || record.TotalTokens != nil || record.ProviderCache.CacheSavings != nil {
			t.Fatal("partial usage invented settled totals or savings")
		}
		if record.Currency == nil || *record.Currency != "USD" {
			t.Fatal("partial component cost lost its currency")
		}
	}
}

func TestProviderCacheAccountingEvidence(t *testing.T) {
	count := func(value int64) llmprotocol.TokenCount {
		return llmprotocol.TokenCount{Value: llmprotocol.Int64(value), Provenance: llmprotocol.UsageAuthoritative}
	}
	complete := llmprotocol.Usage{State: llmprotocol.UsageAvailable, InputTotal: count(100), InputCacheRead: count(20), InputCacheWrite: count(30), OutputTotal: count(2)}
	pricing := config.ModelPricing{Currency: "USD", PromptPer1M: 2, CachedInputPer1M: 1, CompletionPer1M: 10}
	for _, test := range []struct {
		name   string
		change func(*llmprotocol.Usage)
		state  string
	}{
		{"derived uncached input", func(*llmprotocol.Usage) {}, "complete"},
		{"explicit zero", func(u *llmprotocol.Usage) { u.InputCacheRead = count(0) }, "complete"},
		{"missing read", func(u *llmprotocol.Usage) { u.InputCacheRead = llmprotocol.TokenCount{} }, "partial"},
		{"missing write", func(u *llmprotocol.Usage) { u.InputCacheWrite = llmprotocol.TokenCount{} }, "partial"},
		{"estimated read", func(u *llmprotocol.Usage) { u.InputCacheRead.Provenance = llmprotocol.UsageEstimated }, "partial"},
		{"unavailable", func(u *llmprotocol.Usage) { *u = llmprotocol.Usage{State: llmprotocol.UsageUnavailable} }, "unavailable"},
		{"invalid total", func(u *llmprotocol.Usage) { u.InputCacheRead = count(101) }, "invalid"},
	} {
		t.Run(test.name, func(t *testing.T) {
			usage := complete
			test.change(&usage)
			result := buildProviderCacheAccounting(&RequestContext{SemanticResponse: &llmprotocol.Response{Usage: usage}}, &pricing)
			if result.State != test.state {
				t.Fatalf("state=%s, want %s", result.State, test.state)
			}
			if test.state != "complete" && (result.CacheSavings != nil || result.NoCacheCost != nil) {
				t.Fatal("incomplete usage produced cache savings")
			}
			if usage.InputCacheRead.Value == nil && result.CacheRead.Tokens != nil {
				t.Fatal("missing read invented zero")
			}
			if test.name == "derived uncached input" && (result.UncachedInput.Provenance != "derived" || *result.UncachedInput.Tokens != 50) {
				t.Fatal("uncached input not derived from complete evidence")
			}
		})
	}
	ctx := &RequestContext{SemanticResponse: &llmprotocol.Response{Usage: complete}}
	withoutPrices := buildProviderCacheAccounting(ctx, nil)
	if withoutPrices.CacheRead.Tokens == nil || withoutPrices.CacheRead.Cost != nil || withoutPrices.CacheSavings != nil {
		t.Fatal("missing prices invented costs or lost usage")
	}
	free := buildProviderCacheAccounting(ctx, &config.ModelPricing{Currency: "USD"})
	if free.CacheSavings == nil || *free.CacheSavings != 0 {
		t.Fatal("explicit free pricing must retain zero savings")
	}
}
