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
	var realCalls atomic.Int32
	real := func(ctx context.Context, _, _ string) (float32, float32, error) {
		realCalls.Add(1)
		startOnce.Do(func() { close(entered) })
		select {
		case <-release:
			return 0.75, 0.25, nil
		case <-ctx.Done():
			return 0, 0, ctx.Err()
		}
	}
	done := make(chan answerRecord, 1)
	go func() { done <- produceArm(fusion, client, opt, it, entry, "C", real) }()
	select {
	case <-entered:
	case <-time.After(5 * time.Second):
		t.Fatal("real arm did not reach its NLI backend")
	}

	// Keep one real request in flight while another arm and a separate real
	// model use the same FusionLooper. None may replace its scorer.
	placebo := produceArm(fusion, client, opt, it, entry, "D", real)
	require.Empty(t, placebo.Error)
	require.True(t, placebo.GroundingPresent)
	require.Len(t, placebo.Panel, 2)
	assert.Equal(t, int32(1), realCalls.Load(), "placebo must not call the real model")

	other := produceArm(fusion, client, opt, it, entry, "C", func(context.Context, string, string) (float32, float32, error) {
		return 0.25, 0.75, nil
	})
	assertArmScores(t, other, 0.25)
	unblock()
	select {
	case result := <-done:
		assertArmScores(t, result, 0.75)
	case <-time.After(5 * time.Second):
		t.Fatal("real arm did not finish after releasing its NLI backend")
	}
	assert.Equal(t, int32(2), realCalls.Load())

	repeated := produceArm(fusion, client, opt, it, entry, "D", real)
	require.Empty(t, repeated.Error)
	assert.Equal(t, placebo.Panel, repeated.Panel, "placebo remains deterministic for its item and seed")
	plain := produceArm(fusion, client, opt, it, entry, "B", real)
	require.Empty(t, plain.Error)
	assert.False(t, plain.GroundingPresent)
	assert.Equal(t, int32(2), realCalls.Load(), "plain and placebo arms must not use real NLI")
	assert.Equal(t, entry.PanelSHA256, plain.PanelSHA256)
	assert.Equal(t, entry.PanelSHA256, repeated.PanelSHA256)
}

func TestPrepareNLIRejectsMissingArtifact(t *testing.T) {
	nli, owner, err := prepareNLI(options{nliModel: filepath.Join(t.TempDir(), "missing"), useCPU: true})
	require.Error(t, err)
	assert.Nil(t, nli)
	assert.Nil(t, owner)
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
