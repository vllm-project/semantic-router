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
	fixture := newDashboardAgentFixtureClient(ctx, client, baseURL, token, namespaceID, userID, verbose)
	if err := fixture.createModel(); err != nil {
		return "", "", err
	}
	decisionID, err := fixture.createRecipe()
	if err != nil {
		return "", "", err
	}
	if err := fixture.createAndPublishEntrypoint(decisionID); err != nil {
		return "", "", err
	}
	policyID, err := fixture.createAccessPolicy()
	if err != nil {
		return "", "", err
	}
	if err := fixture.issueAPIKey(policyID); err != nil {
		return "", "", err
	}
	if err := waitForDashboardAgentTarget(
		ctx, client, baseURL, token, namespaceID, fixture.entrypointID, verbose,
	); err != nil {
		return "", "", err
	}
	return "entrypoint", fixture.publicID, nil
}

type dashboardAgentFixtureClient struct {
	ctx          context.Context
	client       *http.Client
	baseURL      string
	token        string
	namespaceID  string
	userID       string
	verbose      bool
	suffix       string
	modelID      string
	recipeID     string
	entrypointID string
	publicID     string
}

func newDashboardAgentFixtureClient(
	ctx context.Context,
	client *http.Client,
	baseURL, token, namespaceID, userID string,
	verbose bool,
) dashboardAgentFixtureClient {
	suffix := strconv.FormatInt(time.Now().UTC().UnixNano(), 10)
	return dashboardAgentFixtureClient{
		ctx: ctx, client: client, baseURL: baseURL, token: token,
		namespaceID: namespaceID, userID: userID, verbose: verbose, suffix: suffix,
		modelID: "dashboard_e2e_model_" + suffix, recipeID: "dashboard_e2e_recipe_" + suffix,
		entrypointID: "dashboard_e2e_entrypoint_" + suffix, publicID: "dashboard-e2e-" + suffix,
	}
}

func (fixture dashboardAgentFixtureClient) createModel() error {
	receipt, err := dashboardManagementMutation(
		fixture.ctx, fixture.client, fixture.baseURL, fixture.token, fixture.namespaceID, http.MethodPost,
		"/routing/models", "dashboard-e2e-model-"+fixture.suffix,
		map[string]interface{}{
			"id": fixture.modelID, "name": fixture.publicID + "-model",
			"aliases": []string{fixture.publicID + "-model"}, "capabilities": []string{"streaming", "text", "tools"},
			"control": map[string]interface{}{
				"retry":   map[string]interface{}{"count": 1, "on": []string{"unavailable"}},
				"timeout": map[string]interface{}{"request": "30s", "stream": "2m"},
			},
			"pricing": map[string]interface{}{
				"inputCostPerMillionTokens": nil, "outputCostPerMillionTokens": nil,
				"cacheReadCostPerMillionTokens": nil, "cacheWriteCostPerMillionTokens": nil,
			},
			"backends": []map[string]interface{}{{
				"providerId": "vllm", "providerModelId": dashboardFixtureBackendModel,
				"baseUrl": dashboardFixtureBackendOrigin, "weight": "1",
			}},
		}, fixture.verbose,
	)
	if err != nil || receipt.Resource == nil || receipt.Resource.ID != fixture.modelID {
		return fmt.Errorf("create Agent E2E Model: %w", fixtureReceiptError(err, receipt.Resource))
	}
	return nil
}

func (fixture dashboardAgentFixtureClient) createRecipe() (string, error) {
	receipt, err := dashboardManagementMutation(
		fixture.ctx, fixture.client, fixture.baseURL, fixture.token, fixture.namespaceID, http.MethodPost,
		"/routing/recipes", "dashboard-e2e-recipe-"+fixture.suffix,
		map[string]interface{}{
			"id": fixture.recipeID, "name": "Dashboard E2E Recipe",
			"description": "Deterministic Agent continuity fixture.",
			"document": map[string]interface{}{
				"signals": map[string]interface{}{}, "projections": map[string]interface{}{},
				"decisions": []map[string]interface{}{{"name": "Default", "rules": map[string]interface{}{}}},
			},
		}, fixture.verbose,
	)
	if err != nil || receipt.Resource == nil || receipt.Resource.ID != fixture.recipeID {
		return "", fmt.Errorf("create Agent E2E Recipe: %w", fixtureReceiptError(err, receipt.Resource))
	}
	return dashboardFixtureDecisionID(
		fixture.ctx, fixture.client, fixture.baseURL, fixture.token,
		fixture.namespaceID, fixture.recipeID, fixture.verbose,
	)
}

func (fixture dashboardAgentFixtureClient) createAndPublishEntrypoint(decisionID string) error {
	receipt, err := dashboardManagementMutation(
		fixture.ctx, fixture.client, fixture.baseURL, fixture.token, fixture.namespaceID, http.MethodPost,
		"/routing/entrypoints", "dashboard-e2e-entrypoint-"+fixture.suffix,
		map[string]interface{}{
			"id": fixture.entrypointID, "name": fixture.publicID, "aliases": []string{fixture.publicID},
			"rules": []map[string]interface{}{{
				"id": "dashboard_e2e_rule_" + fixture.suffix, "name": "Default", "recipeId": fixture.recipeID,
				"assignments": map[string]interface{}{decisionID: map[string]interface{}{
					"models": []map[string]interface{}{{"modelId": fixture.modelID, "priority": 0, "weight": "1"}},
				}},
			}},
		}, fixture.verbose,
	)
	if err != nil || receipt.Resource == nil || receipt.Resource.ID != fixture.entrypointID {
		return fmt.Errorf("create Agent E2E Entrypoint: %w", fixtureReceiptError(err, receipt.Resource))
	}
	_, err = dashboardManagementJSONWithHeaders(
		fixture.ctx, fixture.client, fixture.baseURL, fixture.token, fixture.namespaceID, http.MethodPost,
		"/routing/entrypoints/"+fixture.entrypointID+":publish", "dashboard-e2e-publish-"+fixture.suffix,
		nil, http.StatusAccepted, nil, http.Header{"If-Match": []string{`"ep:1"`}}, fixture.verbose,
	)
	if err != nil {
		return fmt.Errorf("publish Agent E2E Entrypoint: %w", err)
	}
	return nil
}

func (fixture dashboardAgentFixtureClient) createAccessPolicy() (string, error) {
	receipt, err := dashboardManagementMutation(
		fixture.ctx, fixture.client, fixture.baseURL, fixture.token, fixture.namespaceID, http.MethodPost,
		"/access-policies", "dashboard-e2e-access-policy-"+fixture.suffix,
		map[string]interface{}{
			"name": "Dashboard E2E Agent access", "status": "active",
			"grants": []map[string]string{
				{"resourceType": "entrypoint", "resourceId": fixture.entrypointID, "permission": "discover", "effect": "allow"},
				{"resourceType": "entrypoint", "resourceId": fixture.entrypointID, "permission": "invoke", "effect": "allow"},
				{"resourceType": "model", "resourceId": fixture.modelID, "permission": "discover", "effect": "allow"},
				{"resourceType": "model", "resourceId": fixture.modelID, "permission": "invoke", "effect": "allow"},
			},
		}, fixture.verbose,
	)
	if err != nil || receipt.Resource == nil || receipt.Resource.ID == "" {
		return "", fmt.Errorf("create Agent E2E Access Policy: %w", fixtureReceiptError(err, receipt.Resource))
	}
	return receipt.Resource.ID, nil
}

func (fixture dashboardAgentFixtureClient) issueAPIKey(policyID string) error {
	var issued struct {
		Data struct {
			KeyID string `json:"keyId"`
		} `json:"data"`
	}
	err := dashboardManagementJSON(
		fixture.ctx, fixture.client, fixture.baseURL, fixture.token, fixture.namespaceID, http.MethodPost,
		"/api-keys", "dashboard-e2e-api-key-"+fixture.suffix,
		map[string]interface{}{
			"name": "Dashboard E2E Agent", "owner": map[string]string{"type": "user", "id": fixture.userID},
			"revealable": false, "accessPolicyIds": []string{policyID},
		}, http.StatusCreated, &issued, fixture.verbose,
	)
	if err != nil {
		return fmt.Errorf("create Agent E2E API key: %w", err)
	}
	if issued.Data.KeyID == "" {
		return fmt.Errorf("create Agent E2E API key returned no key identity")
	}
	return nil
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
