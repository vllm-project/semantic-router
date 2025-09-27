package intercluster

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestInterClusterRouting(t *testing.T) {
	// Create test configuration with inter-cluster routing enabled
	cfg := &config.RouterConfig{
		InterClusterRouting: config.InterClusterConfig{
			Enabled: true,
			ClusterDiscovery: config.ClusterDiscoveryConfig{
				Method: "static",
				StaticClusters: []config.ClusterConfig{
					{
						Name:     "test-cluster-1",
						Location: "us-west-2",
						Type:     "vllm",
						Endpoint: "https://cluster1.example.com:8000",
						Models:   []string{"llama-2-70b", "mistral-7b"},
						Performance: config.PerformanceMetrics{
							AvgLatencyMs: 150,
						},
						CostPerToken: 0.001,
					},
					{
						Name:     "test-cluster-2",
						Location: "eu-west-1",
						Type:     "vllm",
						Endpoint: "https://cluster2.example.com:8000",
						Models:   []string{"llama-2-70b", "gpt-4"},
						Performance: config.PerformanceMetrics{
							AvgLatencyMs: 200,
						},
						CostPerToken: 0.002,
						Compliance:   []string{"gdpr"},
					},
				},
			},
			Providers: []config.ProviderConfig{
				{
					Name:     "openai-provider",
					Type:     "openai",
					Endpoint: "https://api.openai.com/v1",
					Models:   []string{"gpt-4", "gpt-3.5-turbo"},
					Performance: config.PerformanceMetrics{
						AvgLatencyMs: 300,
					},
				},
			},
			RoutingStrategies: []config.RoutingStrategy{
				{
					Name:     "latency-optimized",
					Priority: 100,
					Conditions: []config.RoutingCondition{
						{
							Type:         "latency_requirement",
							MaxLatencyMs: 200,
						},
					},
					Actions: []config.RoutingAction{
						{
							Type:   "route_to_cluster",
							Target: "test-cluster-1",
						},
					},
				},
				{
					Name:     "compliance-routing",
					Priority: 200,
					Conditions: []config.RoutingCondition{
						{
							Type:               "compliance_requirement",
							RequiredCompliance: []string{"gdpr"},
						},
					},
					Actions: []config.RoutingAction{
						{
							Type:   "route_to_cluster",
							Target: "test-cluster-2",
						},
					},
				},
			},
		},
	}

	router := NewInterClusterRouter(cfg)

	t.Run("TestLatencyBasedRouting", func(t *testing.T) {
		ctx := &RoutingContext{
			ModelName:          "llama-2-70b",
			Category:           "general",
			UserContent:        "Test query",
			LatencyRequirement: intPtr(200),
		}

		result, err := router.RouteRequest(ctx)
		if err != nil {
			t.Fatalf("Expected successful routing, got error: %v", err)
		}

		if result.TargetName != "test-cluster-1" {
			t.Errorf("Expected cluster 'test-cluster-1', got '%s'", result.TargetName)
		}

		if result.TargetType != "cluster" {
			t.Errorf("Expected target type 'cluster', got '%s'", result.TargetType)
		}

		if result.ReasonCode != "strategy_latency-optimized" {
			t.Errorf("Expected reason code 'strategy_latency-optimized', got '%s'", result.ReasonCode)
		}
	})

	t.Run("TestComplianceBasedRouting", func(t *testing.T) {
		ctx := &RoutingContext{
			ModelName:              "llama-2-70b",
			Category:               "general",
			UserContent:            "Test query",
			ComplianceRequirements: []string{"gdpr"},
		}

		result, err := router.RouteRequest(ctx)
		if err != nil {
			t.Fatalf("Expected successful routing, got error: %v", err)
		}

		if result.TargetName != "test-cluster-2" {
			t.Errorf("Expected cluster 'test-cluster-2', got '%s'", result.TargetName)
		}

		if result.ReasonCode != "strategy_compliance-routing" {
			t.Errorf("Expected reason code 'strategy_compliance-routing', got '%s'", result.ReasonCode)
		}
	})

	t.Run("TestDefaultRouting", func(t *testing.T) {
		ctx := &RoutingContext{
			ModelName:   "mistral-7b",
			Category:    "general",
			UserContent: "Test query",
		}

		result, err := router.RouteRequest(ctx)
		if err != nil {
			t.Fatalf("Expected successful routing, got error: %v", err)
		}

		// Should route to cluster with better latency (test-cluster-1)
		if result.TargetName != "test-cluster-1" {
			t.Errorf("Expected cluster 'test-cluster-1' for default routing, got '%s'", result.TargetName)
		}

		if result.ReasonCode != "default_latency_optimized" {
			t.Errorf("Expected reason code 'default_latency_optimized', got '%s'", result.ReasonCode)
		}
	})

	t.Run("TestProviderRouting", func(t *testing.T) {
		ctx := &RoutingContext{
			ModelName:   "gpt-3.5-turbo",
			Category:    "general",
			UserContent: "Test query",
		}

		result, err := router.RouteRequest(ctx)
		if err != nil {
			t.Fatalf("Expected successful routing, got error: %v", err)
		}

		// Should route to either cluster or provider that supports gpt-3.5-turbo
		// Since only openai-provider supports gpt-3.5-turbo, it should route there
		if result.TargetName != "openai-provider" && result.TargetName != "test-cluster-2" {
			t.Errorf("Expected 'openai-provider' or 'test-cluster-2', got '%s'", result.TargetName)
		}
	})

	t.Run("TestNoSupportingCluster", func(t *testing.T) {
		ctx := &RoutingContext{
			ModelName:   "unsupported-model",
			Category:    "general",
			UserContent: "Test query",
		}

		_, err := router.RouteRequest(ctx)
		if err == nil {
			t.Fatalf("Expected error for unsupported model, got success")
		}
	})

	t.Run("TestDisabledInterClusterRouting", func(t *testing.T) {
		disabledCfg := &config.RouterConfig{
			InterClusterRouting: config.InterClusterConfig{
				Enabled: false,
			},
		}

		disabledRouter := NewInterClusterRouter(disabledCfg)

		ctx := &RoutingContext{
			ModelName:   "llama-2-70b",
			Category:    "general",
			UserContent: "Test query",
		}

		_, err := disabledRouter.RouteRequest(ctx)
		if err == nil {
			t.Fatalf("Expected error when inter-cluster routing is disabled, got success")
		}
	})
}

func TestConditionEvaluation(t *testing.T) {
	cfg := &config.RouterConfig{
		InterClusterRouting: config.InterClusterConfig{
			Enabled: true,
		},
	}

	router := NewInterClusterRouter(cfg)

	t.Run("TestLatencyCondition", func(t *testing.T) {
		condition := config.RoutingCondition{
			Type:         "latency_requirement",
			MaxLatencyMs: 200,
		}

		ctx := &RoutingContext{
			LatencyRequirement: intPtr(150),
		}

		if !router.evaluateCondition(condition, ctx) {
			t.Errorf("Expected latency condition to pass")
		}

		ctx.LatencyRequirement = intPtr(250)
		if router.evaluateCondition(condition, ctx) {
			t.Errorf("Expected latency condition to fail")
		}
	})

	t.Run("TestCostCondition", func(t *testing.T) {
		condition := config.RoutingCondition{
			Type:               "cost_sensitivity",
			MaxCostPer1kTokens: 0.0015,
		}

		ctx := &RoutingContext{
			CostSensitivity: float64Ptr(0.001),
		}

		if !router.evaluateCondition(condition, ctx) {
			t.Errorf("Expected cost condition to pass")
		}

		ctx.CostSensitivity = float64Ptr(0.002)
		if router.evaluateCondition(condition, ctx) {
			t.Errorf("Expected cost condition to fail")
		}
	})

	t.Run("TestComplianceCondition", func(t *testing.T) {
		condition := config.RoutingCondition{
			Type:               "compliance_requirement",
			RequiredCompliance: []string{"gdpr", "sox"},
		}

		ctx := &RoutingContext{
			ComplianceRequirements: []string{"gdpr", "sox", "hipaa"},
		}

		if !router.evaluateCondition(condition, ctx) {
			t.Errorf("Expected compliance condition to pass")
		}

		ctx.ComplianceRequirements = []string{"gdpr"}
		if router.evaluateCondition(condition, ctx) {
			t.Errorf("Expected compliance condition to fail (missing sox)")
		}
	})
}

// Helper functions for test pointers
func intPtr(i int) *int {
	return &i
}

func float64Ptr(f float64) *float64 {
	return &f
}