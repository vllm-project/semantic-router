package intercluster

import (
	"fmt"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability"
)

// InterClusterRouter handles routing decisions across multiple clusters and providers
type InterClusterRouter struct {
	config *config.RouterConfig
}

// NewInterClusterRouter creates a new inter-cluster router
func NewInterClusterRouter(cfg *config.RouterConfig) *InterClusterRouter {
	return &InterClusterRouter{
		config: cfg,
	}
}

// RoutingContext contains information for routing decisions
type RoutingContext struct {
	ModelName           string
	Category            string
	UserContent         string
	LatencyRequirement  *int     // Max latency in milliseconds
	CostSensitivity     *float64 // Max cost per 1k tokens
	ComplianceRequirements []string // Required compliance standards
	DataResidency       string   // Required region
}

// RoutingResult contains the result of a routing decision
type RoutingResult struct {
	TargetType     string // "cluster" or "provider"
	TargetName     string
	TargetEndpoint string
	ReasonCode     string
	Confidence     float64
	EstimatedCost  float64
	EstimatedLatency int
}

// RouteRequest performs inter-cluster routing based on context and strategies
func (r *InterClusterRouter) RouteRequest(ctx *RoutingContext) (*RoutingResult, error) {
	if !r.config.IsInterClusterRoutingEnabled() {
		observability.Infof("Inter-cluster routing disabled, falling back to local routing")
		return nil, fmt.Errorf("inter-cluster routing is not enabled")
	}

	// Get routing strategies by priority
	strategies := r.config.GetRoutingStrategiesByPriority()
	if len(strategies) == 0 {
		observability.Warnf("No routing strategies configured, using default strategy")
		return r.defaultRouting(ctx)
	}

	// Apply strategies in priority order
	for _, strategy := range strategies {
		if r.evaluateConditions(strategy.Conditions, ctx) {
			result, err := r.executeActions(strategy.Actions, ctx)
			if err == nil && result != nil {
				result.ReasonCode = fmt.Sprintf("strategy_%s", strategy.Name)
				observability.Infof("Applied routing strategy '%s' for model '%s'", strategy.Name, ctx.ModelName)
				return result, nil
			}
			observability.Warnf("Strategy '%s' failed: %v", strategy.Name, err)
		}
	}

	// Fall back to default routing
	observability.Infof("No strategies matched, using default routing for model '%s'", ctx.ModelName)
	return r.defaultRouting(ctx)
}

// evaluateConditions checks if all conditions in a strategy are met
func (r *InterClusterRouter) evaluateConditions(conditions []config.RoutingCondition, ctx *RoutingContext) bool {
	if len(conditions) == 0 {
		return true // No conditions means always apply
	}

	for _, condition := range conditions {
		if !r.evaluateCondition(condition, ctx) {
			return false
		}
	}
	return true
}

// evaluateCondition checks if a single condition is met
func (r *InterClusterRouter) evaluateCondition(condition config.RoutingCondition, ctx *RoutingContext) bool {
	switch condition.Type {
	case "latency_requirement":
		if ctx.LatencyRequirement != nil && condition.MaxLatencyMs > 0 {
			return *ctx.LatencyRequirement <= condition.MaxLatencyMs
		}
		return false // If no latency requirement in context, this condition should not match

	case "cost_sensitivity":
		if ctx.CostSensitivity != nil && condition.MaxCostPer1kTokens > 0 {
			return *ctx.CostSensitivity <= condition.MaxCostPer1kTokens
		}
		return false // If no cost sensitivity in context, this condition should not match

	case "data_residency":
		if condition.RequiredRegion != "" && ctx.DataResidency != "" {
			return ctx.DataResidency == condition.RequiredRegion
		}
		return false // If no data residency requirement, this condition should not match

	case "model_requirement":
		if condition.RequiredModel != "" {
			return ctx.ModelName == condition.RequiredModel
		}
		return true

	case "compliance_requirement":
		if len(condition.RequiredCompliance) > 0 && len(ctx.ComplianceRequirements) > 0 {
			return r.hasRequiredCompliance(condition.RequiredCompliance, ctx.ComplianceRequirements)
		}
		return false // If no compliance requirements in context, this condition should not match

	default:
		observability.Warnf("Unknown condition type: %s", condition.Type)
		return false
	}
}

// hasRequiredCompliance checks if all required compliance standards are met
func (r *InterClusterRouter) hasRequiredCompliance(required []string, available []string) bool {
	for _, req := range required {
		found := false
		for _, avail := range available {
			if req == avail {
				found = true
				break
			}
		}
		if !found {
			return false
		}
	}
	return true
}

// executeActions executes the actions for a matched strategy
func (r *InterClusterRouter) executeActions(actions []config.RoutingAction, ctx *RoutingContext) (*RoutingResult, error) {
	for _, action := range actions {
		switch action.Type {
		case "route_to_cluster":
			return r.routeToCluster(action.Target, ctx)
		case "route_to_provider":
			return r.routeToProvider(action.Target, ctx)
		case "load_balance":
			return r.loadBalanceRoute(action, ctx)
		case "failover":
			return r.failoverRoute(action, ctx)
		default:
			observability.Warnf("Unknown action type: %s", action.Type)
		}
	}
	return nil, fmt.Errorf("no executable actions found")
}

// routeToCluster routes to a specific cluster
func (r *InterClusterRouter) routeToCluster(clusterName string, ctx *RoutingContext) (*RoutingResult, error) {
	cluster, found := r.config.GetClusterByName(clusterName)
	if !found {
		return nil, fmt.Errorf("cluster '%s' not found", clusterName)
	}

	// Check if cluster supports the model
	if !r.clusterSupportsModel(cluster, ctx.ModelName) {
		return nil, fmt.Errorf("cluster '%s' does not support model '%s'", clusterName, ctx.ModelName)
	}

	result := &RoutingResult{
		TargetType:       "cluster",
		TargetName:       clusterName,
		TargetEndpoint:   cluster.Endpoint,
		Confidence:       1.0,
		EstimatedCost:    cluster.CostPerToken * 1000, // Convert to per 1k tokens
		EstimatedLatency: cluster.Performance.AvgLatencyMs,
	}

	observability.Infof("Routed to cluster '%s' for model '%s'", clusterName, ctx.ModelName)
	return result, nil
}

// routeToProvider routes to a specific provider
func (r *InterClusterRouter) routeToProvider(providerName string, ctx *RoutingContext) (*RoutingResult, error) {
	provider, found := r.config.GetProviderByName(providerName)
	if !found {
		return nil, fmt.Errorf("provider '%s' not found", providerName)
	}

	// Check if provider supports the model
	if !r.providerSupportsModel(provider, ctx.ModelName) {
		return nil, fmt.Errorf("provider '%s' does not support model '%s'", providerName, ctx.ModelName)
	}

	result := &RoutingResult{
		TargetType:       "provider",
		TargetName:       providerName,
		TargetEndpoint:   provider.Endpoint,
		Confidence:       1.0,
		EstimatedLatency: provider.Performance.AvgLatencyMs,
	}

	observability.Infof("Routed to provider '%s' for model '%s'", providerName, ctx.ModelName)
	return result, nil
}

// loadBalanceRoute implements load balancing across multiple targets
func (r *InterClusterRouter) loadBalanceRoute(action config.RoutingAction, ctx *RoutingContext) (*RoutingResult, error) {
	clusters, providers := r.config.FindClustersForModel(ctx.ModelName)
	
	if len(clusters) == 0 && len(providers) == 0 {
		return nil, fmt.Errorf("no clusters or providers found for model '%s'", ctx.ModelName)
	}

	// Simple round-robin load balancing based on timestamp
	timestamp := time.Now().UnixNano()
	totalTargets := len(clusters) + len(providers)
	selectedIndex := int(timestamp) % totalTargets

	if selectedIndex < len(clusters) {
		// Route to cluster
		cluster := clusters[selectedIndex]
		return r.routeToCluster(cluster.Name, ctx)
	} else {
		// Route to provider
		provider := providers[selectedIndex-len(clusters)]
		return r.routeToProvider(provider.Name, ctx)
	}
}

// failoverRoute implements failover routing
func (r *InterClusterRouter) failoverRoute(action config.RoutingAction, ctx *RoutingContext) (*RoutingResult, error) {
	for _, target := range action.FailoverTargets {
		// Try cluster first
		if result, err := r.routeToCluster(target, ctx); err == nil {
			result.ReasonCode = "failover_cluster"
			return result, nil
		}

		// Try provider
		if result, err := r.routeToProvider(target, ctx); err == nil {
			result.ReasonCode = "failover_provider"
			return result, nil
		}

		observability.Warnf("Failover target '%s' failed for model '%s'", target, ctx.ModelName)
	}

	return nil, fmt.Errorf("all failover targets failed for model '%s'", ctx.ModelName)
}

// defaultRouting implements default routing strategy (latency-optimized)
func (r *InterClusterRouter) defaultRouting(ctx *RoutingContext) (*RoutingResult, error) {
	clusters, providers := r.config.FindClustersForModel(ctx.ModelName)
	
	if len(clusters) == 0 && len(providers) == 0 {
		return nil, fmt.Errorf("no clusters or providers found for model '%s'", ctx.ModelName)
	}

	var bestResult *RoutingResult
	bestLatency := int(^uint(0) >> 1) // Max int

	// Check clusters
	for _, cluster := range clusters {
		if cluster.Performance.AvgLatencyMs > 0 && cluster.Performance.AvgLatencyMs < bestLatency {
			bestLatency = cluster.Performance.AvgLatencyMs
			bestResult = &RoutingResult{
				TargetType:       "cluster",
				TargetName:       cluster.Name,
				TargetEndpoint:   cluster.Endpoint,
				ReasonCode:       "default_latency_optimized",
				Confidence:       0.8,
				EstimatedCost:    cluster.CostPerToken * 1000,
				EstimatedLatency: cluster.Performance.AvgLatencyMs,
			}
		}
	}

	// Check providers
	for _, provider := range providers {
		if provider.Performance.AvgLatencyMs > 0 && provider.Performance.AvgLatencyMs < bestLatency {
			bestLatency = provider.Performance.AvgLatencyMs
			bestResult = &RoutingResult{
				TargetType:       "provider",
				TargetName:       provider.Name,
				TargetEndpoint:   provider.Endpoint,
				ReasonCode:       "default_latency_optimized",
				Confidence:       0.8,
				EstimatedLatency: provider.Performance.AvgLatencyMs,
			}
		}
	}

	if bestResult == nil {
		// Fallback: choose first available
		if len(clusters) > 0 {
			cluster := clusters[0]
			bestResult = &RoutingResult{
				TargetType:       "cluster",
				TargetName:       cluster.Name,
				TargetEndpoint:   cluster.Endpoint,
				ReasonCode:       "default_first_available",
				Confidence:       0.6,
				EstimatedCost:    cluster.CostPerToken * 1000,
				EstimatedLatency: cluster.Performance.AvgLatencyMs,
			}
		} else {
			provider := providers[0]
			bestResult = &RoutingResult{
				TargetType:       "provider",
				TargetName:       provider.Name,
				TargetEndpoint:   provider.Endpoint,
				ReasonCode:       "default_first_available",
				Confidence:       0.6,
				EstimatedLatency: provider.Performance.AvgLatencyMs,
			}
		}
	}

	observability.Infof("Default routing selected '%s' for model '%s' with latency %dms", 
		bestResult.TargetName, ctx.ModelName, bestResult.EstimatedLatency)
	return bestResult, nil
}

// clusterSupportsModel checks if a cluster supports a specific model
func (r *InterClusterRouter) clusterSupportsModel(cluster *config.ClusterConfig, modelName string) bool {
	for _, model := range cluster.Models {
		if model == modelName {
			return true
		}
	}
	return false
}

// providerSupportsModel checks if a provider supports a specific model
func (r *InterClusterRouter) providerSupportsModel(provider *config.ProviderConfig, modelName string) bool {
	for _, model := range provider.Models {
		if model == modelName {
			return true
		}
	}
	return false
}