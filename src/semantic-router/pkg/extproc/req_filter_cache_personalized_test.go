// Copyright 2025 vLLM Semantic Router Contributors
// SPDX-License-Identifier: Apache-2.0

package extproc

import (
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestDecisionWillPersonalize(t *testing.T) {
	globalMemory := withGlobalPlugins(t, &config.RouterConfig{
		IntelligentRouting: config.IntelligentRouting{Decisions: []config.Decision{{Name: "d"}}},
	}, memoryTestPlugin())
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
						{Type: config.DecisionPluginContext, Configuration: config.MustStructuredPayload(map[string]interface{}{"enabled": true, "sources": []interface{}{map[string]interface{}{"type": "rag", "backend": "vectorstore"}}})},
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
						{Type: config.DecisionPluginContext, Configuration: config.MustStructuredPayload(map[string]interface{}{"enabled": false, "sources": []interface{}{map[string]interface{}{"type": "rag", "backend": "vectorstore"}}})},
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
						{Type: config.DecisionPluginContext, Configuration: config.MustStructuredPayload(map[string]interface{}{"enabled": true, "sources": []interface{}{map[string]interface{}{"type": "memory"}}})},
					},
				},
			},
			cfg:  &config.RouterConfig{},
			want: true,
		},
		{
			name: "global memory enabled",
			ctx:  &RequestContext{VSRSelectedDecision: &globalMemory.Decisions[0]},
			cfg:  globalMemory,
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
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, decisionWillPersonalize(tt.ctx))
		})
	}
}

func TestUpdateResponseCacheSkipsPersonalizedContext(t *testing.T) {
	mockCache := &mockStreamingCache{}
	router := &OpenAIRouter{
		Cache: mockCache,
		Config: withGlobalPlugins(t, &config.RouterConfig{
			SemanticCache: config.SemanticCache{InUse: true},
		}, testPlugin(config.DecisionPluginResponseCache, map[string]interface{}{"enabled": true, "scope": "global"})),
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
			mockCache.addEntryCalled = false
			tt.ctx.RequestModel = "test"
			tt.ctx.RequestQuery = "hello"
			tt.ctx.SemanticRequest = testNeutralRequest("test", "hello")
			tt.ctx.Routing.SelectRecipe(router.Config.DefaultRecipe())
			router.updateResponseCache(tt.ctx, []byte(`{"choices":[]}`))
			assert.Equal(t, tt.wantUpdate, mockCache.addEntryCalled,
				"updateResponseCache should %s for %s", map[bool]string{true: "write", false: "skip"}[tt.wantUpdate], tt.name)
		})
	}
}
