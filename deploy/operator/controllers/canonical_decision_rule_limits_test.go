package controllers

import (
	"context"
	"strings"
	"testing"

	"k8s.io/utils/ptr"

	vllmv1alpha1 "github.com/vllm-project/semantic-router/operator/api/v1alpha1"
)

func TestDecisionRuleLimitsReachRouterConfig(t *testing.T) {
	for _, limits := range []*vllmv1alpha1.DecisionRuleLimitsConfig{
		nil, {MaxDepth: ptr.To(32)}, {MaxNodes: ptr.To(512)}, {MaxDepth: ptr.To(32), MaxNodes: ptr.To(512)},
	} {
		_, cfg := routerConfigOf(t, vllmv1alpha1.ConfigSpec{DecisionRuleLimits: limits})
		depth, nodes, err := cfg.DecisionRuleLimits.Effective()
		wantDepth, wantNodes := 16, 256
		if limits != nil && limits.MaxDepth != nil {
			wantDepth = *limits.MaxDepth
		}
		if limits != nil && limits.MaxNodes != nil {
			wantNodes = *limits.MaxNodes
		}
		if err != nil || depth != wantDepth || nodes != wantNodes {
			t.Fatalf("limits lost in operator/router round trip: %d/%d, %v", depth, nodes, err)
		}
	}
}

func TestOperatorDecisionRuleLimitsRejectOversizedCanonicalRouting(t *testing.T) {
	spec := vllmv1alpha1.ConfigSpec{
		DecisionRuleLimits: &vllmv1alpha1.DecisionRuleLimitsConfig{MaxNodes: ptr.To(1)},
		Routing:            rawCanonicalRoutingJSON(t, `{"decisions":[{"name":"bounded","rules":{"operator":"AND","conditions":[{"type":"keyword","name":"urgent"}]}}]}`),
	}
	r := &SemanticRouterReconciler{}
	_, err := r.buildCanonicalConfig(context.Background(), &vllmv1alpha1.SemanticRouter{Spec: vllmv1alpha1.SemanticRouterSpec{Config: spec}})
	if err == nil || !strings.Contains(err.Error(), "node count 2 exceeds max_nodes=1") {
		t.Fatalf("operator candidate must fail before recursive conversion: %v", err)
	}
}
