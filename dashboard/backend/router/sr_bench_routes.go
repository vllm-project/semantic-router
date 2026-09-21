package router

import (
	"log"
	"net/http"
	"os"
	"strings"

	"github.com/vllm-project/semantic-router/dashboard/backend/auth"
	"github.com/vllm-project/semantic-router/dashboard/backend/config"
	"github.com/vllm-project/semantic-router/dashboard/backend/handlers"
)

const maxSRBenchBodyBytes = handlers.SRBenchMaxRequestBytes

func registerSRBenchRoutes(routes *auth.PolicyMux, cfg *config.Config) {
	// Retired /api/evaluation paths are no longer registered; the registry
	// denies them like every other unknown API route.
	notFound := func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Cache-Control", "private, no-store")
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusNotFound)
		_, _ = w.Write([]byte(`{"error":{"message":"Endpoint not found"}}`))
	}

	cfg.SRBenchAvailable = false
	cfg.SRBenchUnavailableReason = "sr-bench service authentication is not configured."
	token := os.Getenv(cfg.SRBenchTokenEnv)
	handler, err := handlers.NewSRBenchHandler(cfg.SRBenchURL, token, cfg.ReadonlyMode)
	if err != nil {
		cfg.SRBenchUnavailableReason = "sr-bench service configuration is invalid."
		log.Printf("sr-bench proxy configuration is invalid")
		routes.HandleGroup(srBenchRouteContracts(), http.HandlerFunc(notFound))
		return
	}
	if strings.TrimSpace(token) != "" {
		cfg.SRBenchAvailable = true
		cfg.SRBenchUnavailableReason = ""
	}
	routes.HandleGroup(srBenchRouteContracts(), handler)
	log.Printf("sr-bench 1.0 API registered; workers are owned by the independent service")
}

// srBenchRouteContracts mirrors the sr-bench 1.0 surface the proxy forwards.
// Run creation and recovery persist a manifest and immediately launch work, so
// both need write and run permissions; cancellation only needs run.
func srBenchRouteContracts() []auth.RouteContract {
	const base = handlers.SRBenchAPIPath
	read := func(pattern string) auth.RouteContract {
		return auth.ProtectedRoute(base+pattern, auth.PermEvalRead, auth.SensitivitySensitive, auth.ResourceOwnerEvaluation, http.MethodGet)
	}
	write := func(pattern, action string) auth.RouteContract {
		return auth.ProtectedMutationRoute(base+pattern, auth.PermEvalWrite, action, auth.SensitivitySensitive, auth.ResourceOwnerEvaluation, maxSRBenchBodyBytes, http.MethodPost)
	}
	contracts := []auth.RouteContract{
		read(""),
		read("/health"), read("/catalog"), read("/datasets"), read("/datasets/selection"),
		read("/targets"), read("/replay-options"), read("/comparison-options"),
		read("/datasets/{id}"), read("/datasets/{id}/cases"),
		// Saved-result comparison is a body-based read.
		auth.ProtectedBoundedRoute(base+"/comparisons", auth.PermEvalRead, auth.SensitivitySensitive, auth.ResourceOwnerEvaluation, maxSRBenchBodyBytes, http.MethodPost),
		write("/plans", "sr_bench.plan.create"),
		write("/replays", "sr_bench.replay.create"),
		write("/datasets/compose", "sr_bench.dataset.compose"),
		auth.Route(base+"/runs",
			auth.ReadPolicy(http.MethodGet, auth.PermEvalRead, auth.SensitivitySensitive, auth.ResourceOwnerEvaluation),
			auth.MutationPolicy(http.MethodPost, auth.PermEvalWrite, "sr_bench.run.create", auth.SensitivitySensitive, auth.ResourceOwnerEvaluation, maxSRBenchBodyBytes).AlsoRequiring(auth.PermEvalRun),
		),
		auth.Route(base+"/experiments",
			auth.ReadPolicy(http.MethodGet, auth.PermEvalRead, auth.SensitivitySensitive, auth.ResourceOwnerEvaluation),
			auth.MutationPolicy(http.MethodPost, auth.PermEvalWrite, "sr_bench.experiment.create", auth.SensitivitySensitive, auth.ResourceOwnerEvaluation, maxSRBenchBodyBytes),
		),
		auth.Route(base+"/experiments/{id}",
			auth.ReadPolicy(http.MethodGet, auth.PermEvalRead, auth.SensitivitySensitive, auth.ResourceOwnerEvaluation),
			auth.MutationPolicy(http.MethodDelete, auth.PermEvalWrite, "sr_bench.experiment.delete", auth.SensitivitySensitive, auth.ResourceOwnerEvaluation, auth.NoBodyLimit),
		),
		auth.Route(base+"/experiments/{id}/runs",
			auth.ReadPolicy(http.MethodGet, auth.PermEvalRead, auth.SensitivitySensitive, auth.ResourceOwnerEvaluation),
			auth.MutationPolicy(http.MethodPost, auth.PermEvalWrite, "sr_bench.experiment.run", auth.SensitivitySensitive, auth.ResourceOwnerEvaluation, maxSRBenchBodyBytes),
		),
		read("/runs/{id}"), read("/runs/{id}/results"), read("/runs/{id}/report"),
		read("/runs/{id}/events"), read("/runs/{id}/calls"), read("/runs/{id}/calls/{call}"),
		auth.ProtectedMutationRoute(base+"/runs/{id}/cancel", auth.PermEvalRun, "sr_bench.run.cancel", auth.SensitivitySensitive, auth.ResourceOwnerEvaluation, maxSRBenchBodyBytes, http.MethodPost),
		write("/runs/{id}/regrade", "sr_bench.run.regrade"),
		write("/runs/{id}/export", "sr_bench.run.export"),
		write("/runs/{id}/recover-plan", "sr_bench.run.recover_plan"),
		auth.Route(base+"/runs/{id}/recover",
			auth.MutationPolicy(http.MethodPost, auth.PermEvalWrite, "sr_bench.run.recover", auth.SensitivitySensitive, auth.ResourceOwnerEvaluation, maxSRBenchBodyBytes).AlsoRequiring(auth.PermEvalRun),
		),
		write("/runs/{id}/reconcile-usage", "sr_bench.run.reconcile_usage"),
		write("/runs/{id}/candidate-plan", "sr_bench.run.candidate_plan"),
	}
	return contracts
}
