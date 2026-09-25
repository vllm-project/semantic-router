package router

import (
	"net/http"

	"github.com/vllm-project/semantic-router/dashboard/backend/auth"
	"github.com/vllm-project/semantic-router/dashboard/backend/handlers"
)

func srBenchRouteContracts() []auth.RouteContract {
	const base = handlers.SRBenchAPIPath
	read := func(path string) auth.RouteContract {
		return auth.ProtectedRoute(base+path, auth.PermEvalRead, auth.SensitivitySensitive, auth.ResourceOwnerEvaluation, http.MethodGet)
	}
	write := func(path, action string) auth.RouteContract {
		return auth.ProtectedMutationRoute(base+path, auth.PermEvalWrite, action, auth.SensitivitySensitive, auth.ResourceOwnerEvaluation, 8<<20, http.MethodPost)
	}
	contracts := []auth.RouteContract{read("")}
	for _, path := range []string{
		"/health", "/catalog", "/datasets", "/datasets/selection", "/targets",
		"/replay-options", "/comparison-options", "/dataset-preparations/options",
		"/datasets/{id}", "/datasets/{id}/cases", "/dataset-preparations/{id}",
		"/runs/{id}", "/runs/{id}/results", "/runs/{id}/report", "/runs/{id}/events",
		"/runs/{id}/calls", "/runs/{id}/calls/{callID}",
	} {
		contracts = append(contracts, read(path))
	}
	for _, path := range []string{"/plans", "/replays", "/datasets/compose"} {
		contracts = append(contracts, write(path, "srbench.write"))
	}
	compare := auth.ReadPolicy(http.MethodPost, auth.PermEvalRead, auth.SensitivitySensitive, auth.ResourceOwnerEvaluation)
	compare.MaxBodyBytes = 8 << 20
	contracts = append(contracts, auth.Route(base+"/comparisons", compare))
	for _, path := range []string{"/runs", "/experiments/{id}/runs"} {
		create := auth.MutationPolicy(http.MethodPost, auth.PermEvalWrite, "srbench.run.create", auth.SensitivitySensitive, auth.ResourceOwnerEvaluation, 8<<20)
		create.AdditionalPermissions = []string{auth.PermEvalRun}
		contracts = append(contracts, auth.Route(base+path,
			auth.ReadPolicy(http.MethodGet, auth.PermEvalRead, auth.SensitivitySensitive, auth.ResourceOwnerEvaluation), create))
	}
	for _, path := range []string{"/experiments", "/dataset-preparations"} {
		contracts = append(contracts, auth.Route(base+path,
			auth.ReadPolicy(http.MethodGet, auth.PermEvalRead, auth.SensitivitySensitive, auth.ResourceOwnerEvaluation),
			auth.MutationPolicy(http.MethodPost, auth.PermEvalWrite, "srbench.create", auth.SensitivitySensitive, auth.ResourceOwnerEvaluation, 8<<20)))
	}
	contracts = append(contracts, auth.Route(base+"/experiments/{id}",
		auth.ReadPolicy(http.MethodGet, auth.PermEvalRead, auth.SensitivitySensitive, auth.ResourceOwnerEvaluation),
		auth.MutationPolicy(http.MethodDelete, auth.PermEvalWrite, "srbench.experiment.delete", auth.SensitivitySensitive, auth.ResourceOwnerEvaluation, 8<<20)))
	for _, action := range []string{"cancel", "regrade", "export", "recover-plan", "recover", "reconcile-usage", "candidate-plan"} {
		policy := auth.MutationPolicy(http.MethodPost, auth.PermEvalWrite, "srbench.run."+action, auth.SensitivitySensitive, auth.ResourceOwnerEvaluation, 8<<20)
		if action == "cancel" {
			policy.Permission = auth.PermEvalRun
		}
		if action == "recover" {
			policy.AdditionalPermissions = []string{auth.PermEvalRun}
		}
		contracts = append(contracts, auth.Route(base+"/runs/{id}/"+action, policy))
	}
	return contracts
}
