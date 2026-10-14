package main

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/looper"
)

func TestProduceArmKeepsRealAndPlaceboGroundingRequestLocal(t *testing.T) {
	fusion, client := evaluationFusion(t)
	opt := options{judge: "judge", panelModels: []string{"panel-a", "panel-b"}, reference: config.FusionGroundingReferencePanel, placeboSeed: 7}
	it := item{ID: "item-1", Question: "What does the evidence show?"}
	entry := cachedPanelItem{ItemID: it.ID, Panel: []cachedResponse{
		{Model: "panel-a", Content: "Evidence supports this statement."},
		{Model: "panel-b", Content: "The statement is supported by the evidence."},
	}}
	entry.PanelSHA256 = panelSHA256(entry.Panel)

	entered, release := make(chan struct{}), make(chan struct{})
	var startOnce, releaseOnce sync.Once
	unblock := func() { releaseOnce.Do(func() { close(release) }) }
	t.Cleanup(unblock)
	var realCalls, forbiddenCalls atomic.Int32
	forbidden := func(context.Context, string, string, string) (looper.GroundingEvidence, error) {
		forbiddenCalls.Add(1)
		return looper.GroundingEvidence{}, nil
	}
	real := func(ctx context.Context, _, _, _ string) (looper.GroundingEvidence, error) {
		realCalls.Add(1)
		startOnce.Do(func() { close(entered) })
		select {
		case <-release:
			score := float32(0.25)
			return looper.GroundingEvidence{Unsupported: true, Spans: []string{"unsupported"}, Probability: &score}, nil
		case <-ctx.Done():
			return looper.GroundingEvidence{}, ctx.Err()
		}
	}
	done := make(chan answerRecord, 1)
	go func() {
		done <- produceArm(fusion, client, opt, it, entry, "C", &looper.GroundingBackends{Detect: real})
	}()
	select {
	case <-entered:
	case <-time.After(5 * time.Second):
		t.Fatal("real arm did not reach its detector")
	}

	// Keep one real request in flight while another arm and a separate real
	// model use the same FusionLooper. None may replace its scorer.
	placebo := produceArm(fusion, client, opt, it, entry, "D", &looper.GroundingBackends{Detect: forbidden})
	require.Empty(t, placebo.Error)
	require.True(t, placebo.GroundingPresent)
	require.Len(t, placebo.Panel, 2)

	other := produceArm(fusion, client, opt, it, entry, "C", &looper.GroundingBackends{Detect: func(context.Context, string, string, string) (looper.GroundingEvidence, error) {
		score := float32(0.75)
		return looper.GroundingEvidence{Unsupported: true, Spans: []string{"unsupported"}, Probability: &score}, nil
	}})
	assertArmScores(t, other, 0.25)
	unblock()
	select {
	case result := <-done:
		assertArmScores(t, result, 0.75)
	case <-time.After(5 * time.Second):
		t.Fatal("real arm did not finish after releasing its detector")
	}
	assert.Equal(t, int32(2), realCalls.Load(), "the real arm reads each response against its peer once")

	repeated := produceArm(fusion, client, opt, it, entry, "D", &looper.GroundingBackends{Detect: forbidden})
	require.Empty(t, repeated.Error)
	assert.Equal(t, placebo.Panel, repeated.Panel, "placebo remains deterministic for its item and seed")
	plain := produceArm(fusion, client, opt, it, entry, "B", &looper.GroundingBackends{Detect: forbidden})
	require.Empty(t, plain.Error)
	assert.False(t, plain.GroundingPresent)
	assert.Zero(t, forbiddenCalls.Load(), "plain and placebo arms must not use the request's detector")
	assert.Equal(t, entry.PanelSHA256, plain.PanelSHA256)
	assert.Equal(t, entry.PanelSHA256, repeated.PanelSHA256)
}

func TestRouterGroundingRejectsMissingConfig(t *testing.T) {
	backends, owner, err := routerGrounding(filepath.Join(t.TempDir(), "missing.yaml"))
	require.Error(t, err)
	assert.Nil(t, backends)
	assert.Nil(t, owner)
}

func TestOnlyShippedGroundingArmsNeedTheRouterConfig(t *testing.T) {
	reference := config.FusionGroundingReferencePanel
	assert.False(t, needsShippedGrounding([]string{"A", "B", "D"}, reference))
	for _, arm := range []string{"C", "annotate", "filter"} {
		assert.True(t, needsShippedGrounding([]string{"A", arm}, reference), arm)
	}
}

func assertArmScores(t *testing.T, record answerRecord, score float64) {
	t.Helper()
	require.Empty(t, record.Error)
	require.True(t, record.GroundingPresent)
	require.Len(t, record.Panel, 2)
	for _, panel := range record.Panel {
		require.NotNil(t, panel.GroundingScore)
		assert.InDelta(t, score, *panel.GroundingScore, 1e-6)
	}
}

func evaluationFusion(t *testing.T) (*looper.FusionLooper, *looper.Client) {
	t.Helper()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var payload struct {
			Model    string `json:"model"`
			Messages []struct {
				Content string `json:"content"`
			} `json:"messages"`
		}
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			t.Errorf("decode judge request: %v", err)
			http.Error(w, "bad request", http.StatusBadRequest)
			return
		}
		if payload.Model != "judge" {
			t.Errorf("cached panel unexpectedly called %q", payload.Model)
			http.Error(w, "uncached model", http.StatusBadRequest)
			return
		}
		content := "Final answer."
		for _, message := range payload.Messages {
			if strings.Contains(message.Content, "return only valid JSON") {
				content = `{"consensus":[],"contradictions":[],"partial_coverage":[],"unique_insights":[],"blind_spots":[]}`
			}
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"id": "evaluation", "object": "chat.completion", "created": 1, "model": "judge",
			"choices": []map[string]any{{"index": 0, "finish_reason": "stop", "message": map[string]string{"role": "assistant", "content": content}}},
			"usage":   map[string]int{"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
		})
	}))
	t.Cleanup(server.Close)
	cfg := &config.LooperConfig{Endpoint: server.URL}
	client, err := looper.NewConnectorClient(cfg)
	require.NoError(t, err)
	t.Cleanup(func() { assert.NoError(t, client.Close()) })
	algorithm, err := looper.FactoryWithClient(cfg, config.DecisionAlgorithmFusion, client)
	require.NoError(t, err)
	fusion, ok := algorithm.(*looper.FusionLooper)
	require.True(t, ok)
	return fusion, client
}
