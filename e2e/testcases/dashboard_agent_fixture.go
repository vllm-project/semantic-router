package testcases

import (
	"context"
	"fmt"
	"net/http"
	"strconv"
	"time"
)

const (
	dashboardFixtureBackendOrigin = "http://vllm-llama3-8b-instruct.default.svc.cluster.local:8000/v1"
	dashboardFixtureBackendModel  = "base-model"
)

// ensureDashboardAgentFixture creates one complete, public-API-only managed
// topology for the Agent continuity contract. The fixture deliberately uses
// the same Management resources and publication path as a console: no SQL
// seed, mounted routing document, or Dashboard-owned inference authority is
// involved.
func ensureDashboardAgentFixture(
	ctx context.Context,
	client *http.Client,
	baseURL string,
	token string,
	namespaceID string,
	userID string,
	verbose bool,
) (string, string, error) {
	if namespaceID == "" || userID == "" {
		return "", "", fmt.Errorf("Agent E2E fixture requires a namespace and linked User")
	}
	if err := enableDashboardAgentDelegation(ctx, client, baseURL, token, namespaceID, verbose); err != nil {
		return "", "", err
	}

	suffix := strconv.FormatInt(time.Now().UTC().UnixNano(), 10)
	modelID := "dashboard_e2e_model_" + suffix
	recipeID := "dashboard_e2e_recipe_" + suffix
	entrypointID := "dashboard_e2e_entrypoint_" + suffix
	publicID := "dashboard-e2e-" + suffix

	modelReceipt, err := dashboardManagementMutation(
		ctx, client, baseURL, token, namespaceID, http.MethodPost,
		"/routing/models", "dashboard-e2e-model-"+suffix,
		map[string]interface{}{
			"id": modelID, "name": publicID + "-model",
			"aliases":      []string{publicID + "-model"},
			"capabilities": []string{"streaming", "text", "tools"},
			"execution": map[string]interface{}{
				"maxRetries": 1, "requestTimeout": "30s", "streamTimeout": "2m",
			},
			"pricing": map[string]interface{}{
				"inputCostPerMillionTokens": nil, "outputCostPerMillionTokens": nil,
				"cacheReadCostPerMillionTokens": nil, "cacheWriteCostPerMillionTokens": nil,
			},
			"backends": []map[string]interface{}{{
				"providerId": "vllm", "providerModelId": dashboardFixtureBackendModel,
				"baseUrl": dashboardFixtureBackendOrigin, "weight": "1",
			}},
		},
		verbose,
	)
	if err != nil || modelReceipt.Resource == nil || modelReceipt.Resource.ID != modelID {
		return "", "", fmt.Errorf("create Agent E2E Model: %w", fixtureReceiptError(err, modelReceipt.Resource))
	}

	recipeReceipt, err := dashboardManagementMutation(
		ctx, client, baseURL, token, namespaceID, http.MethodPost,
		"/routing/recipes", "dashboard-e2e-recipe-"+suffix,
		map[string]interface{}{
			"id": recipeID, "name": "Dashboard E2E Recipe",
			"description": "Deterministic Agent continuity fixture.",
			"document": map[string]interface{}{
				"signals": map[string]interface{}{}, "projections": map[string]interface{}{},
				"decisions": []map[string]interface{}{{"name": "Default", "rules": map[string]interface{}{}}},
			},
		},
		verbose,
	)
	if err != nil || recipeReceipt.Resource == nil || recipeReceipt.Resource.ID != recipeID {
		return "", "", fmt.Errorf("create Agent E2E Recipe: %w", fixtureReceiptError(err, recipeReceipt.Resource))
	}
	decisionID, err := dashboardFixtureDecisionID(
		ctx, client, baseURL, token, namespaceID, recipeID, verbose,
	)
	if err != nil {
		return "", "", err
	}

	entrypointReceipt, err := dashboardManagementMutation(
		ctx, client, baseURL, token, namespaceID, http.MethodPost,
		"/routing/entrypoints", "dashboard-e2e-entrypoint-"+suffix,
		map[string]interface{}{
			"id": entrypointID, "name": publicID, "aliases": []string{publicID},
			"rules": []map[string]interface{}{{
				"id": "dashboard_e2e_rule_" + suffix, "name": "Default", "recipeId": recipeID,
				"assignments": map[string]interface{}{
					decisionID: map[string]interface{}{
						"models": []map[string]interface{}{{"modelId": modelID, "priority": 0, "weight": "1"}},
					},
				},
			}},
		},
		verbose,
	)
	if err != nil || entrypointReceipt.Resource == nil || entrypointReceipt.Resource.ID != entrypointID {
		return "", "", fmt.Errorf("create Agent E2E Entrypoint: %w", fixtureReceiptError(err, entrypointReceipt.Resource))
	}
	if _, err := dashboardManagementJSONWithHeaders(
		ctx, client, baseURL, token, namespaceID, http.MethodPost,
		"/routing/entrypoints/"+entrypointID+":publish", "dashboard-e2e-publish-"+suffix,
		nil, http.StatusAccepted, nil,
		http.Header{"If-Match": []string{`"ep:1"`}}, verbose,
	); err != nil {
		return "", "", fmt.Errorf("publish Agent E2E Entrypoint: %w", err)
	}

	policyReceipt, err := dashboardManagementMutation(
		ctx, client, baseURL, token, namespaceID, http.MethodPost,
		"/access-policies", "dashboard-e2e-access-policy-"+suffix,
		map[string]interface{}{
			"name": "Dashboard E2E Agent access", "status": "active",
			"grants": []map[string]string{
				{"resourceType": "entrypoint", "resourceId": entrypointID, "permission": "discover", "effect": "allow"},
				{"resourceType": "entrypoint", "resourceId": entrypointID, "permission": "invoke", "effect": "allow"},
				{"resourceType": "model", "resourceId": modelID, "permission": "discover", "effect": "allow"},
				{"resourceType": "model", "resourceId": modelID, "permission": "invoke", "effect": "allow"},
			},
		},
		verbose,
	)
	if err != nil || policyReceipt.Resource == nil || policyReceipt.Resource.ID == "" {
		return "", "", fmt.Errorf("create Agent E2E Access Policy: %w", fixtureReceiptError(err, policyReceipt.Resource))
	}

	var issued struct {
		Data struct {
			KeyID string `json:"keyId"`
		} `json:"data"`
	}
	if err := dashboardManagementJSON(
		ctx, client, baseURL, token, namespaceID, http.MethodPost, "/api-keys",
		"dashboard-e2e-api-key-"+suffix,
		map[string]interface{}{
			"name": "Dashboard E2E Agent", "owner": map[string]string{"type": "user", "id": userID},
			"revealable": false, "accessPolicyIds": []string{policyReceipt.Resource.ID},
		},
		http.StatusCreated, &issued, verbose,
	); err != nil {
		return "", "", fmt.Errorf("create Agent E2E API key: %w", err)
	}
	if issued.Data.KeyID == "" {
		return "", "", fmt.Errorf("create Agent E2E API key returned no key identity")
	}

	if err := waitForDashboardAgentTarget(
		ctx, client, baseURL, token, namespaceID, entrypointID, verbose,
	); err != nil {
		return "", "", err
	}
	return "entrypoint", publicID, nil
}

func enableDashboardAgentDelegation(
	ctx context.Context,
	client *http.Client,
	baseURL string,
	token string,
	namespaceID string,
	verbose bool,
) error {
	path := "/namespaces/" + namespaceID + "/self-service-policy"
	var detail struct {
		Data struct {
			MaxKeysPerUser       int    `json:"maxKeysPerUser"`
			MaxDelegatedSessions int    `json:"maxDelegatedSessions"`
			Revision             uint64 `json:"revision"`
		} `json:"data"`
	}
	headers, err := dashboardManagementJSONWithHeaders(
		ctx, client, baseURL, token, namespaceID, http.MethodGet, path, "", nil,
		http.StatusOK, &detail, nil, verbose,
	)
	if err != nil {
		return fmt.Errorf("read Agent E2E self-service policy: %w", err)
	}
	if detail.Data.MaxKeysPerUser >= 4 && detail.Data.MaxDelegatedSessions >= 4 {
		return nil
	}
	etag := headers.Get("ETag")
	if etag == "" || detail.Data.Revision == 0 {
		return fmt.Errorf("Agent E2E self-service policy omitted its revision")
	}
	_, err = dashboardManagementJSONWithHeaders(
		ctx, client, baseURL, token, namespaceID, http.MethodPatch, path, "",
		map[string]interface{}{
			"maxKeysPerUser": 4, "maxDelegatedSessions": 4,
			"delegatedSessionTtlSeconds": 900,
			"reason":                     "Enable bounded Agent E2E sessions.",
		},
		http.StatusOK, nil, http.Header{"If-Match": []string{etag}}, verbose,
	)
	if err != nil {
		return fmt.Errorf("enable Agent E2E delegation: %w", err)
	}
	return nil
}

func dashboardFixtureDecisionID(
	ctx context.Context,
	client *http.Client,
	baseURL string,
	token string,
	namespaceID string,
	recipeID string,
	verbose bool,
) (string, error) {
	var detail struct {
		Data struct {
			Decisions []struct {
				ID string `json:"id"`
			} `json:"decisions"`
		} `json:"data"`
	}
	if err := dashboardManagementJSON(
		ctx, client, baseURL, token, namespaceID, http.MethodGet,
		"/routing/recipes/"+recipeID, "", nil, http.StatusOK, &detail, verbose,
	); err != nil {
		return "", fmt.Errorf("read Agent E2E Recipe: %w", err)
	}
	if len(detail.Data.Decisions) != 1 || detail.Data.Decisions[0].ID == "" {
		return "", fmt.Errorf("Agent E2E Recipe returned an invalid Decision projection")
	}
	return detail.Data.Decisions[0].ID, nil
}

func waitForDashboardAgentTarget(
	ctx context.Context,
	client *http.Client,
	baseURL string,
	token string,
	namespaceID string,
	entrypointID string,
	verbose bool,
) error {
	deadline := time.Now().Add(90 * time.Second)
	var lastStatus string
	for time.Now().Before(deadline) {
		var detail struct {
			Data struct {
				Status string `json:"status"`
			} `json:"data"`
		}
		err := dashboardManagementJSON(
			ctx, client, baseURL, token, namespaceID, http.MethodGet,
			"/routing/entrypoints/"+entrypointID, "", nil, http.StatusOK, &detail, verbose,
		)
		if err == nil {
			lastStatus = detail.Data.Status
			if lastStatus == "active" {
				return nil
			}
		}
		timer := time.NewTimer(time.Second)
		select {
		case <-ctx.Done():
			timer.Stop()
			return ctx.Err()
		case <-timer.C:
		}
	}
	return fmt.Errorf("Agent E2E Entrypoint did not become active (last status %q)", lastStatus)
}

func fixtureReceiptError(err error, resource interface{}) error {
	if err != nil {
		return err
	}
	if resource == nil {
		return fmt.Errorf("mutation returned no resource")
	}
	return fmt.Errorf("mutation returned the wrong resource")
}
