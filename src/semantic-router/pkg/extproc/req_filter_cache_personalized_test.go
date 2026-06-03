// Copyright 2025 vLLM Semantic Router Contributors
// SPDX-License-Identifier: Apache-2.0

package extproc

import (
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestDecisionWillPersonalize(t *testing.T) {
	tests := []struct {
		name string
		ctx  *RequestContext
		cfg  *config.RouterConfig
		want bool
	}{
		{
			name: "no decision — not personalized",
			ctx:  &RequestContext{},
			cfg:  &config.RouterConfig{},
			want: false,
		},
		{
			name: "decision with RAG enabled",
			ctx: &RequestContext{
				VSRSelectedDecision: &config.Decision{
					Plugins: []config.DecisionPlugin{
						{Type: "rag", Configuration: config.MustStructuredPayload(map[string]interface{}{"enabled": true})},
					},
				},
			},
			cfg:  &config.RouterConfig{},
			want: true,
		},
		{
			name: "decision with RAG disabled",
			ctx: &RequestContext{
				VSRSelectedDecision: &config.Decision{
					Plugins: []config.DecisionPlugin{
						{Type: "rag", Configuration: config.MustStructuredPayload(map[string]interface{}{"enabled": false})},
					},
				},
			},
			cfg:  &config.RouterConfig{},
			want: false,
		},
		{
			name: "decision with per-decision memory enabled",
			ctx: &RequestContext{
				VSRSelectedDecision: &config.Decision{
					Plugins: []config.DecisionPlugin{
						{Type: "memory", Configuration: config.MustStructuredPayload(map[string]interface{}{"enabled": true})},
					},
				},
			},
			cfg:  &config.RouterConfig{},
			want: true,
		},
		{
			name: "global memory enabled",
			ctx: &RequestContext{
				VSRSelectedDecision: &config.Decision{},
			},
			cfg: &config.RouterConfig{
				Memory: config.MemoryConfig{Enabled: true},
			},
			want: true,
		},
		{
			name: "no plugins — not personalized",
			ctx: &RequestContext{
				VSRSelectedDecision: &config.Decision{},
			},
			cfg:  &config.RouterConfig{},
			want: false,
		},
		{
			name: "decision name lookup without decision object",
			ctx: &RequestContext{
				VSRSelectedDecisionName: "rag-only",
			},
			cfg: func() *config.RouterConfig {
				cfg := &config.RouterConfig{}
				cfg.Decisions = []config.Decision{
					{
						Name:      "rag-only",
						ModelRefs: []config.ModelRef{{Model: "m"}},
						Plugins: []config.DecisionPlugin{
							{Type: "rag", Configuration: config.MustStructuredPayload(map[string]interface{}{
								"enabled": true,
								"backend": "milvus",
							})},
						},
					},
				}
				return cfg
			}(),
			want: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, decisionWillPersonalize(tt.ctx, tt.cfg))
		})
	}
}

func TestDecisionWillPersonalizeForDecision_UsesCategoryFallback(t *testing.T) {
	ctx := &RequestContext{}
	cfg := &config.RouterConfig{}
	cfg.Decisions = []config.Decision{
		{
			Name:      "rag-from-category",
			ModelRefs: []config.ModelRef{{Model: "m"}},
			Plugins: []config.DecisionPlugin{
				{Type: "rag", Configuration: config.MustStructuredPayload(map[string]interface{}{
					"enabled": true,
					"backend": "milvus",
				})},
			},
		},
	}

	assert.True(t, decisionWillPersonalizeForDecision(ctx, cfg, "rag-from-category"))
}

func TestUpdateResponseCacheSkipsPersonalizedContext(t *testing.T) {
	mockCache := &mockStreamingCache{}
	router := &OpenAIRouter{
		Cache: mockCache,
		Config: &config.RouterConfig{
			SemanticCache: config.SemanticCache{Enabled: true},
		},
	}

	tests := []struct {
		name       string
		ctx        *RequestContext
		wantUpdate bool
	}{
		{
			name:       "generic response is cached",
			ctx:        &RequestContext{RequestID: "req-1"},
			wantUpdate: true,
		},
		{
			name:       "RAG response is not cached",
			ctx:        &RequestContext{RequestID: "req-2", RAGRetrievedContext: "private docs"},
			wantUpdate: false,
		},
		{
			name:       "memory response is not cached",
			ctx:        &RequestContext{RequestID: "req-3", MemoryContext: "user history"},
			wantUpdate: false,
		},
		{
			name:       "PII response is not cached",
			ctx:        &RequestContext{RequestID: "req-4", PIIDetected: true},
			wantUpdate: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mockCache.updateCalled = false
			router.updateResponseCache(tt.ctx, []byte(`{"choices":[]}`))
			assert.Equal(t, tt.wantUpdate, mockCache.updateCalled,
				"updateResponseCache should %s for %s", map[bool]string{true: "write", false: "skip"}[tt.wantUpdate], tt.name)
		})
	}
}
