package classification

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"net"
	"strings"
	"testing"
	"time"

	"github.com/alicebob/miniredis/v2"
	"github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

func sharedVerdictTestConfig(t *testing.T, address string) *config.GuardVerdictStoreConfig {
	t.Helper()
	t.Setenv("TEST_GUARD_HMAC", strings.Repeat("ab", 32))
	t.Setenv("TEST_GUARD_REDIS_PASSWORD", "guard-test-password")
	return &config.GuardVerdictStoreConfig{
		Backend: "redis", HMACKeyEnv: "TEST_GUARD_HMAC",
		Redis: &config.GuardVerdictRedisConfig{Address: address, PasswordEnv: "TEST_GUARD_REDIS_PASSWORD"},
	}
}

func sharedVerdictTestServer(t *testing.T) (*miniredis.Miniredis, *config.GuardVerdictStoreConfig) {
	t.Helper()
	server := miniredis.RunT(t)
	server.RequireAuth("guard-test-password")
	return server, sharedVerdictTestConfig(t, server.Addr())
}

// Two independently initialized classifiers alternate the same long turn.
// Model reads and public coverage/decision metrics cross the cache boundary.
func TestSharedGuardWindowsAcrossReplicas(t *testing.T) {
	server, storeConfig := sharedVerdictTestServer(t)
	configure := func(cfg *config.RouterConfig) { cfg.GuardVerdicts = storeConfig }
	first, firstModel := refusingGuardClassifier(t, 20*time.Second, "", time.Millisecond, configure)
	second, secondModel := refusingGuardClassifier(t, 20*time.Second, "", time.Millisecond, configure)
	text := scanProse(12_000)
	reads := metrics.SignalScanWindowsTotal.WithLabelValues("primary", "read")
	kept := metrics.SignalScanWindowsTotal.WithLabelValues("primary", "kept")
	beforeReads, beforeKept := testutil.ToFloat64(reads), testutil.ToFloat64(kept)
	baseline := evaluateTurn(first, text)
	count := len(guardStates(firstModel))
	windows := testutil.ToFloat64(reads) - beforeReads
	require.Greater(t, count, 2)
	require.False(t, guarded(baseline))
	for _, replica := range []*Classifier{second, first, second} {
		result := evaluateTurn(replica, text)
		require.Equal(t, baseline.SignalValues, result.SignalValues)
		require.Equal(t, baseline.ScanCoverage, result.ScanCoverage)
		require.False(t, guarded(result))
	}
	require.Empty(t, guardStates(secondModel))
	require.Len(t, guardStates(firstModel), count)
	require.Equal(t, windows, testutil.ToFloat64(reads)-beforeReads)
	require.Equal(t, 3*windows, testutil.ToFloat64(kept)-beforeKept)

	// Local hits renew Redis, and can repopulate entries lost to eviction.
	server.FastForward(windowVerdictTTL - time.Minute)
	agePublishedVerdicts(first, windowVerdictTTL-time.Minute)
	evaluateTurn(first, text)
	require.Eventually(t, func() bool {
		for _, key := range server.Keys() {
			if server.TTL(key) <= windowVerdictTTL-time.Second {
				return false
			}
		}
		return true
	}, 2*time.Second, time.Millisecond)
	server.FlushAll()
	agePublishedVerdicts(first, sharedVerdictRefreshInterval+time.Second)
	evaluateTurn(first, text)
	require.Eventually(t, func() bool { return len(server.Keys()) == int(windows) }, 2*time.Second, time.Millisecond)
	// A new replica starts with empty local memory and still reuses the turn.
	restarted, restartedModel := refusingGuardClassifier(t, 20*time.Second, "", time.Millisecond, configure)
	require.Equal(t, baseline.ScanCoverage, evaluateTurn(restarted, text).ScanCoverage)
	require.Empty(t, guardStates(restartedModel))
}

func TestSharedGuardDecisionsSurviveInvalidEntriesAndOutages(t *testing.T) {
	for _, failure := range []string{"none", "forged", "outage"} {
		t.Run(failure, func(t *testing.T) {
			server, storeConfig := sharedVerdictTestServer(t)
			configure := func(cfg *config.RouterConfig) { cfg.GuardVerdicts = storeConfig }
			first, _ := refusingGuardClassifier(t, 20*time.Second, "", time.Millisecond, configure)
			second, model := refusingGuardClassifier(t, 20*time.Second, "", time.Millisecond, configure)
			baseline, _ := refusingGuardClassifier(t, 20*time.Second, "", time.Millisecond)
			for _, text := range []string{scanProse(8_000), scanInjection + " " + scanProse(8_000)} {
				want := evaluateTurn(baseline, text)
				evaluateTurn(first, text)
				if failure == "forged" {
					for _, key := range server.Keys() {
						require.NoError(t, server.Set(key, `{"payload":{"version":1,"response":{"Answers":{}}},"mac":"AAAA"}`))
					}
				}
				if failure == "outage" {
					server.Close()
				}
				before := len(guardStates(model))
				got := evaluateTurn(second, text)
				require.Equal(t, guarded(want), guarded(got))
				require.Equal(t, want.MatchedJailbreakRules, got.MatchedJailbreakRules)
				if failure != "none" {
					require.Greater(t, len(guardStates(model)), before)
				}
			}
		})
	}
}

func newVerdictTestStore(t *testing.T, cfg *config.GuardVerdictStoreConfig, recipe string) *sharedWindowVerdicts {
	t.Helper()
	store, err := newSharedWindowVerdicts(cfg, recipe)
	require.NoError(t, err)
	t.Cleanup(func() { _ = store.Close() })
	return store
}

func TestSharedVerdictsBindKeyRecipeExpiryAndSigningKey(t *testing.T) {
	server, cfg := sharedVerdictTestServer(t)
	store := newVerdictTestStore(t, cfg, "first")
	otherRecipe := newVerdictTestStore(t, cfg, "second")
	ctx := context.Background()
	key, otherKey := sha256.Sum256([]byte("window")), sha256.Sum256([]byte("other"))
	response := modelservice.Response{Answers: map[string]modelservice.Answer{"guard": {Choice: "benign", InputCoverage: "complete"}}}
	store.put(ctx, key, response)
	got, _, ok := store.get(ctx, key)
	require.True(t, ok)
	require.Equal(t, cloneWindowResponse(response), got)
	data, err := server.Get(store.key(key))
	require.NoError(t, err)
	for _, destination := range []string{store.key(otherKey), otherRecipe.key(key)} {
		require.NoError(t, server.Set(destination, data))
	}
	_, _, ok = store.get(ctx, otherKey)
	require.False(t, ok, "a signed verdict must not be transplanted to another window")
	_, _, ok = otherRecipe.get(ctx, key)
	require.False(t, ok, "recipe caches remain isolated")

	t.Setenv("TEST_GUARD_HMAC", strings.Repeat("cd", 32))
	rotated := newVerdictTestStore(t, cfg, "first")
	_, _, ok = rotated.get(ctx, key)
	require.False(t, ok, "a different Router signing key cannot trust this entry")

	var signed signedWindowVerdict
	require.NoError(t, json.Unmarshal([]byte(data), &signed))
	var entry sharedWindowVerdict
	require.NoError(t, json.Unmarshal(signed.Payload, &entry))
	entry.ExpiresAt = time.Now().Add(-time.Second).UnixMilli()
	signed.Payload, err = json.Marshal(entry)
	require.NoError(t, err)
	signed.MAC = store.signature(store.key(key), signed.Payload)
	expired, err := json.Marshal(signed)
	require.NoError(t, err)
	require.NoError(t, server.Set(store.key(key), string(expired)))
	_, _, ok = store.get(ctx, key)
	require.False(t, ok, "extending Redis TTL cannot revive a signed expired verdict")
	require.NoError(t, server.Set(store.key(key), strings.Repeat("x", sharedVerdictMaxBytes+1)))
	_, _, ok = store.get(ctx, key)
	require.False(t, ok, "oversized entries are ignored")
}

func TestSharedVerdictsDoNotStorePIIText(t *testing.T) {
	server, cfg := sharedVerdictTestServer(t)
	store := newVerdictTestStore(t, cfg, "recipe")
	key := sha256.Sum256([]byte("pii window"))
	response := modelservice.Response{Answers: map[string]modelservice.Answer{
		"pii": {InputCoverage: "complete", Spans: []modelservice.Span{{Start: 1, End: 3, Label: "PERSON", Text: "张三", Probability: 0.99}}},
	}}
	store.put(context.Background(), key, response)
	data, err := server.Get(store.key(key))
	require.NoError(t, err)
	require.NotContains(t, data, "张三")
	require.Equal(t, "张三", response.Answers["pii"].Spans[0].Text, "encoding does not change the model answer")
	got, _, ok := store.get(context.Background(), key)
	require.True(t, ok)
	call := &systemOneCall{scan: &guardScan{text: "找张三吧", windows: []scanWindow{{start: 0, end: len("找张三吧")}}}}
	require.True(t, restoreWindowSpans(call, got))
	require.Equal(t, cloneWindowResponse(response), got)
}

func TestSharedVerdictCredentialsAreValidatedWithoutConnecting(t *testing.T) {
	_, cfg := sharedVerdictTestServer(t)
	for _, secret := range []string{"", "not-hex", strings.Repeat("ab", 31)} {
		t.Setenv("TEST_GUARD_HMAC", secret)
		_, err := newSharedWindowVerdicts(cfg, "recipe")
		require.Error(t, err)
	}
	t.Setenv("TEST_GUARD_HMAC", strings.Repeat("ab", 32))
	t.Setenv("TEST_GUARD_REDIS_PASSWORD", "")
	_, err := newSharedWindowVerdicts(cfg, "recipe")
	require.Error(t, err)
	t.Setenv("TEST_GUARD_REDIS_PASSWORD", strings.Repeat("ab", 32))
	_, err = newSharedWindowVerdicts(cfg, "recipe")
	require.Error(t, err, "Redis must never receive the signing secret as its password")
}

func TestSharedVerdictTimeoutAndCooldown(t *testing.T) {
	// Accept TCP but never answer Redis's handshake or commands.
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	require.NoError(t, err)
	done := make(chan struct{})
	stopped := make(chan struct{})
	go func() {
		defer close(stopped)
		connection, acceptErr := listener.Accept()
		if acceptErr == nil {
			defer connection.Close()
			<-done
		}
	}()
	t.Cleanup(func() { close(done); _ = listener.Close(); <-stopped })
	store := newVerdictTestStore(t, sharedVerdictTestConfig(t, listener.Addr().String()), "recipe")
	key := sha256.Sum256([]byte("window"))
	started := time.Now()
	_, _, ok := store.get(context.Background(), key)
	require.False(t, ok)
	require.Less(t, time.Since(started), 500*time.Millisecond)
	require.Greater(t, store.retryAfter.Load(), time.Now().UnixNano())
	started = time.Now()
	for range 100 {
		_, _, ok = store.get(context.Background(), key)
		require.False(t, ok)
	}
	require.Less(t, time.Since(started), 100*time.Millisecond)
}

func TestSharedVerdictsKeepOnlyCompleteRequestAnswers(t *testing.T) {
	server, cfg := sharedVerdictTestServer(t)
	classifier, _ := refusingGuardClassifier(t, 20*time.Second, "", time.Millisecond, func(router *config.RouterConfig) { router.GuardVerdicts = cfg })
	asked := jailbreakAsked(t, classifier)
	key := windowVerdictKey(sha256.Sum256([]byte("complete window")))
	call := &systemOneCall{verdict: &key, rules: []askedRule{asked}, scan: &guardScan{
		text: "abc", windows: []scanWindow{{start: 0, end: 3}}, reused: make([]bool, 1),
	}}
	ctx := context.Background()
	for _, answer := range []modelservice.Answer{{}, {InputCoverage: "partial"}, {InputCoverage: "complete", Error: "failed"}, {
		InputCoverage: "complete", Spans: []modelservice.Span{{Start: 0, End: 4}},
	}} {
		response := modelservice.Response{Answers: map[string]modelservice.Answer{asked.questionID(): answer}}
		if len(answer.Spans) == 0 {
			classifier.keepWindow(ctx, call, response, nil)
			require.Empty(t, server.Keys())
		}
		// Even correctly signed incomplete answers from a different replica
		// must pass the same keep rule before entering local memory.
		classifier.sharedVerdicts.put(ctx, key, response)
		hits := metrics.GuardVerdictStoreOpsTotal.WithLabelValues("get", "hit")
		invalid := metrics.GuardVerdictStoreOpsTotal.WithLabelValues("get", "invalid")
		beforeHits, beforeInvalid := testutil.ToFloat64(hits), testutil.ToFloat64(invalid)
		_, ok := classifier.keptWindow(ctx, call)
		require.False(t, ok)
		require.Equal(t, beforeHits, testutil.ToFloat64(hits), "a rejected signed verdict is not a cache hit")
		require.Equal(t, beforeInvalid+1, testutil.ToFloat64(invalid))
		server.FlushAll()
	}
	complete := modelservice.Response{Answers: map[string]modelservice.Answer{asked.questionID(): {InputCoverage: "complete"}}}
	classifier.keepWindow(ctx, call, complete, context.DeadlineExceeded)
	require.Empty(t, server.Keys())
	asked.rule.Stage = config.SystemOneStageResponse
	responseScan := &guardScan{rules: []askedRule{asked}}
	responseScan.verdicts = classifier.newScanVerdictKeys(responseScan)
	require.Nil(t, responseScan.verdicts)
	call.verdict = responseScan.verdictKey(scanWindow{})
	classifier.keepWindow(ctx, call, complete, nil)
	require.Empty(t, server.Keys())
	_, ok := classifier.keptWindow(ctx, call)
	require.False(t, ok)
}

// Age the local publication observation alongside the test Redis clock.
func agePublishedVerdicts(c *Classifier, elapsed time.Duration) {
	c.windowVerdicts.mu.Lock()
	defer c.windowVerdicts.mu.Unlock()
	for _, element := range c.windowVerdicts.entries {
		entry := element.Value.(*keptWindowEntry)
		entry.sharedUntil = entry.sharedUntil.Add(-elapsed)
	}
}
