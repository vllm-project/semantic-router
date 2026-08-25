package testcases

import (
	"context"
	"fmt"
	"net/http"
	"net/url"
	"strconv"
	"time"
)

type managedAccessFixtureBuilder struct {
	ctx                    context.Context
	client                 *managedAccessClient
	seed                   string
	namespaceID            string
	userID                 string
	modelID                string
	recipeID               string
	decisionID             string
	authorizedEntrypointID string
	hiddenEntrypointID     string
	authorizedName         string
	hiddenName             string
}

func createManagedAccessFixture(
	ctx context.Context,
	client *managedAccessClient,
) (managedAccessFixture, error) {
	builder := newManagedAccessFixtureBuilder(ctx, client)
	if err := builder.createFixtureNamespaceAccess(); err != nil {
		return managedAccessFixture{}, err
	}
	if err := builder.createFixtureRouting(); err != nil {
		return managedAccessFixture{}, err
	}
	return builder.createFixtureIdentityPolicy()
}

func newManagedAccessFixtureBuilder(
	ctx context.Context,
	client *managedAccessClient,
) *managedAccessFixtureBuilder {
	seed := strconv.FormatInt(time.Now().UTC().UnixNano(), 10)
	return &managedAccessFixtureBuilder{
		ctx:                    ctx,
		client:                 client,
		seed:                   seed,
		modelID:                "managed_access_model_" + seed,
		recipeID:               "managed_access_recipe_" + seed,
		authorizedEntrypointID: "managed_access_authorized_" + seed,
		hiddenEntrypointID:     "managed_access_hidden_" + seed,
		authorizedName:         "managed-access-authorized-" + seed,
		hiddenName:             "managed-access-hidden-" + seed,
	}
}

func (builder *managedAccessFixtureBuilder) createFixtureNamespaceAccess() error {
	principalID, err := builder.readFixturePrincipal()
	if err != nil {
		return err
	}
	namespace, err := builder.client.mutation(
		builder.ctx, "", http.MethodPost, "/namespaces", "managed-access-namespace-"+builder.seed,
		map[string]string{
			"name": "Managed access " + builder.seed, "billingCurrency": "USD",
			"reason": "Create an isolated managed-access E2E namespace.",
		},
	)
	if err != nil {
		return fmt.Errorf("create fixture Namespace: %w", err)
	}
	builder.namespaceID = namespace.ID
	roleID, permissions, err := builder.readPlatformAdministratorRole()
	if err != nil {
		return err
	}
	if _, bindingErr := builder.client.mutation(
		builder.ctx, builder.namespaceID, http.MethodPost, "/role-bindings",
		"managed-access-role-binding-"+builder.seed,
		map[string]interface{}{
			"principalId": principalID, "roleId": roleID,
			"scope":             map[string]string{"kind": "namespace", "namespaceId": builder.namespaceID},
			"delegationCeiling": permissions,
		},
	); bindingErr != nil {
		return fmt.Errorf("bind fixture Namespace administrator: %w", bindingErr)
	}
	user, err := builder.client.mutation(
		builder.ctx, builder.namespaceID, http.MethodPost, "/users", "managed-access-user-"+builder.seed,
		map[string]string{
			"email": "managed-access-" + builder.seed + "@example.test", "displayName": "Managed Access E2E",
		},
	)
	if err != nil {
		return fmt.Errorf("create fixture User: %w", err)
	}
	builder.userID = user.ID
	return nil
}

func (builder *managedAccessFixtureBuilder) readFixturePrincipal() (string, error) {
	var me struct {
		Principal struct {
			PrincipalID string `json:"principalId"`
		} `json:"principal"`
	}
	if _, _, err := builder.client.request(
		builder.ctx, "", http.MethodGet, "/me", "", nil, nil, []int{http.StatusOK}, &me,
	); err != nil {
		return "", fmt.Errorf("read fixture Management identity: %w", err)
	}
	if me.Principal.PrincipalID == "" {
		return "", fmt.Errorf("fixture Management identity has no principal")
	}
	return me.Principal.PrincipalID, nil
}

func (builder *managedAccessFixtureBuilder) readPlatformAdministratorRole() (string, []string, error) {
	var roles struct {
		Data []struct {
			RoleID      string   `json:"roleId"`
			Name        string   `json:"name"`
			Permissions []string `json:"permissions"`
			BuiltIn     bool     `json:"builtIn"`
			Status      string   `json:"status"`
		} `json:"data"`
	}
	rolesPath := "/management-roles?namespaceId=" + url.QueryEscape(builder.namespaceID) + "&pageSize=200"
	if _, _, err := builder.client.request(
		builder.ctx, "", http.MethodGet, rolesPath, "", nil, nil, []int{http.StatusOK}, &roles,
	); err != nil {
		return "", nil, fmt.Errorf("list fixture Management roles: %w", err)
	}
	for _, role := range roles.Data {
		if role.Name == "platform_admin" && role.BuiltIn && role.Status == "active" {
			if role.RoleID != "" && len(role.Permissions) > 0 {
				return role.RoleID, append([]string(nil), role.Permissions...), nil
			}
			break
		}
	}
	return "", nil, fmt.Errorf("fixture Namespace has no active built-in platform administrator role")
}

func (builder *managedAccessFixtureBuilder) createFixtureRouting() error {
	if err := builder.createFixtureModel(); err != nil {
		return err
	}
	if err := builder.createFixtureRecipe(); err != nil {
		return err
	}
	return builder.createFixtureEntrypoints()
}

func (builder *managedAccessFixtureBuilder) createFixtureModel() error {
	model, err := builder.client.mutation(
		builder.ctx, builder.namespaceID, http.MethodPost, "/routing/models", "managed-access-model-"+builder.seed,
		map[string]interface{}{
			"id": builder.modelID, "name": "managed-access-model-" + builder.seed,
			"aliases":      []string{"managed-access-model-" + builder.seed},
			"capabilities": []string{"streaming", "text", "tools"},
			"control": map[string]interface{}{
				"retry":   map[string]interface{}{"count": 1, "on": []string{"unavailable"}},
				"timeout": map[string]interface{}{"request": "30s", "stream": "2m"},
			},
			"pricing": map[string]interface{}{
				"inputCostPerMillionTokens": "1000000", "outputCostPerMillionTokens": "1000000",
				"cacheReadCostPerMillionTokens": "1000000", "cacheWriteCostPerMillionTokens": "1000000",
			},
			"backends": []map[string]interface{}{{
				"providerId": "vllm", "providerModelId": managedAccessFixtureBackendModel,
				"baseUrl": managedAccessFixtureBackendOrigin, "weight": "1",
			}},
		},
	)
	if err != nil || model.ID != builder.modelID {
		return fmt.Errorf("create fixture Model: %w", fixtureReceiptError(err, model.ID))
	}
	return nil
}

func (builder *managedAccessFixtureBuilder) createFixtureRecipe() error {
	recipe, err := builder.client.mutation(
		builder.ctx, builder.namespaceID, http.MethodPost, "/routing/recipes", "managed-access-recipe-"+builder.seed,
		map[string]interface{}{
			"id": builder.recipeID, "name": "Managed Access Recipe",
			"description": "Deterministic managed-access lifecycle fixture.",
			"document": map[string]interface{}{
				"signals": map[string]interface{}{}, "projections": map[string]interface{}{},
				"decisions": []map[string]interface{}{{"name": "Default", "rules": map[string]interface{}{}}},
			},
		},
	)
	if err != nil || recipe.ID != builder.recipeID {
		return fmt.Errorf("create fixture Recipe: %w", fixtureReceiptError(err, recipe.ID))
	}
	var detail struct {
		Data struct {
			Decisions []struct {
				ID string `json:"id"`
			} `json:"decisions"`
		} `json:"data"`
	}
	if _, _, err := builder.client.request(
		builder.ctx, builder.namespaceID, http.MethodGet, "/routing/recipes/"+builder.recipeID,
		"", nil, nil, []int{http.StatusOK}, &detail,
	); err != nil {
		return fmt.Errorf("read fixture Recipe: %w", err)
	}
	if len(detail.Data.Decisions) != 1 || detail.Data.Decisions[0].ID == "" {
		return fmt.Errorf("fixture Recipe has no canonical Decision")
	}
	builder.decisionID = detail.Data.Decisions[0].ID
	return nil
}

func (builder *managedAccessFixtureBuilder) createFixtureEntrypoints() error {
	for _, entrypoint := range []struct {
		id   string
		name string
	}{
		{id: builder.authorizedEntrypointID, name: builder.authorizedName},
		{id: builder.hiddenEntrypointID, name: builder.hiddenName},
	} {
		created, createErr := builder.client.mutation(
			builder.ctx, builder.namespaceID, http.MethodPost, "/routing/entrypoints",
			"managed-access-entrypoint-"+entrypoint.id,
			map[string]interface{}{
				"id": entrypoint.id, "name": entrypoint.name, "aliases": []string{entrypoint.name},
				"rules": []map[string]interface{}{{
					"id": "managed_access_rule_" + entrypoint.id, "name": "Default", "recipeId": builder.recipeID,
					"assignments": map[string]interface{}{
						builder.decisionID: map[string]interface{}{
							"models": []map[string]interface{}{{"modelId": builder.modelID, "priority": 0, "weight": "1"}},
						},
					},
				}},
			},
		)
		if createErr != nil || created.ID != entrypoint.id {
			return fmt.Errorf("create fixture Entrypoint %q: %w", entrypoint.name, fixtureReceiptError(createErr, created.ID))
		}
		if entrypointPublishErr := publishManagedAccessEntrypoint(
			builder.ctx, builder.client, builder.namespaceID, entrypoint.id, created.Revision,
		); entrypointPublishErr != nil {
			return fmt.Errorf("publish fixture Entrypoint %q: %w", entrypoint.name, entrypointPublishErr)
		}
		if waitErr := waitManagedAccessEntrypoint(
			builder.ctx, builder.client, builder.namespaceID, entrypoint.id,
		); waitErr != nil {
			return waitErr
		}
	}
	return nil
}

func (builder *managedAccessFixtureBuilder) createFixtureIdentityPolicy() (managedAccessFixture, error) {
	accessPolicyID, ratePolicyID, err := builder.createFixturePolicies()
	if err != nil {
		return managedAccessFixture{}, err
	}
	return builder.issueFixtureTeamKey(accessPolicyID, ratePolicyID)
}

func (builder *managedAccessFixtureBuilder) createFixturePolicies() (string, string, error) {
	accessPolicy, err := builder.client.mutation(
		builder.ctx, builder.namespaceID, http.MethodPost, "/access-policies", "managed-access-policy-"+builder.seed,
		map[string]interface{}{
			"name": "Managed access E2E", "status": "active",
			"grants": []map[string]string{
				{"resourceType": "entrypoint", "resourceId": builder.authorizedEntrypointID, "permission": "discover", "effect": "allow"},
				{"resourceType": "entrypoint", "resourceId": builder.authorizedEntrypointID, "permission": "invoke", "effect": "allow"},
				{"resourceType": "model", "resourceId": builder.modelID, "permission": "invoke", "effect": "allow"},
			},
		},
	)
	if err != nil {
		return "", "", fmt.Errorf("create fixture Access Policy: %w", err)
	}
	ratePolicy, err := builder.client.mutation(
		builder.ctx, builder.namespaceID, http.MethodPost, "/rate-limit-policies",
		"managed-access-rate-policy-"+builder.seed,
		map[string]interface{}{
			"name": "Managed access E2E limits", "status": "active",
			"rules": []map[string]interface{}{
				{"metric": "requests", "algorithm": "sliding_log", "limit": "12", "window": "PT1M", "accounting": "request", "enforcement": "enforce"},
				{"metric": "served_total_tokens", "algorithm": "calendar_window", "limit": "100000", "period": "day", "timezone": "UTC", "accounting": "response_actual", "enforcement": "enforce"},
				{"metric": "cost", "algorithm": "calendar_window", "limit": "100000", "period": "day", "timezone": "UTC", "accounting": "response_actual", "enforcement": "enforce"},
			},
		},
	)
	if err != nil {
		return "", "", fmt.Errorf("create fixture Rate Limit Policy: %w", err)
	}
	return accessPolicy.ID, ratePolicy.ID, nil
}

func (builder *managedAccessFixtureBuilder) issueFixtureTeamKey(
	accessPolicyID string,
	ratePolicyID string,
) (managedAccessFixture, error) {
	team, err := builder.client.mutation(
		builder.ctx, builder.namespaceID, http.MethodPost, "/teams", "managed-access-team-"+builder.seed,
		map[string]interface{}{
			"name": "Managed Access Team " + builder.seed, "description": "Isolated E2E team.",
			"accessPolicyIds": []string{accessPolicyID}, "rateLimitPolicyId": ratePolicyID,
		},
	)
	if err != nil {
		return managedAccessFixture{}, fmt.Errorf("create fixture Team: %w", err)
	}
	var membership managedAccessMutationReceipt
	if _, _, err := builder.client.request(
		builder.ctx, builder.namespaceID, http.MethodPut, "/teams/"+team.ID+"/members/"+builder.userID,
		"managed-access-membership-"+builder.seed, map[string]string{"role": "member"}, nil,
		[]int{http.StatusOK}, &membership,
	); err != nil {
		return managedAccessFixture{}, fmt.Errorf("create fixture Team membership: %w", err)
	}
	if membership.Resource == nil || membership.Resource.Kind != "team_membership" ||
		membership.Resource.ID != builder.userID || membership.Resource.Revision != 1 {
		return managedAccessFixture{}, fmt.Errorf("fixture Team membership response is incomplete")
	}
	var issued managedAccessIssuedKey
	if _, _, err := builder.client.request(
		builder.ctx, builder.namespaceID, http.MethodPost, "/api-keys", "managed-access-key-"+builder.seed,
		map[string]interface{}{
			"name": "Managed Access E2E", "owner": map[string]string{"type": "team", "id": team.ID},
			"revealable": false,
		}, nil, []int{http.StatusCreated}, &issued,
	); err != nil {
		return managedAccessFixture{}, fmt.Errorf("issue fixture API key: %w", err)
	}
	if issued.Data.KeyID == "" || issued.Data.Revision == 0 || issued.Data.Status != "active" || issued.Secret == "" {
		return managedAccessFixture{}, fmt.Errorf("fixture API key response is incomplete")
	}
	return managedAccessFixture{
		namespaceID:    builder.namespaceID,
		keyID:          issued.Data.KeyID,
		keyRevision:    issued.Data.Revision,
		secret:         issued.Secret,
		authorizedName: builder.authorizedName,
		hiddenName:     builder.hiddenName,
	}, nil
}
