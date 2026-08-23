package agentworkflow

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"time"

	"github.com/google/uuid"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/agentmanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/routingmanagement"
)

type recipeProbeInput struct {
	RecipeID       string `json:"recipeId"`
	EntrypointID   string `json:"entrypointId"`
	TimeoutSeconds int    `json:"timeoutSeconds,omitempty" jsonschema:"minimum=1,maximum=60"`
}

type modelProbeEvidence struct {
	ModelID             string `json:"modelId"`
	ModelName           string `json:"modelName"`
	Available           bool   `json:"available"`
	LatencyMilliseconds int64  `json:"latencyMilliseconds,omitempty"`
	ErrorCode           string `json:"errorCode,omitempty"`
}

type recipeProbeOutput struct {
	ArtifactID         string               `json:"artifactId"`
	RecipeRevision     int64                `json:"recipeRevision"`
	EntrypointRevision int64                `json:"entrypointRevision"`
	Passed             bool                 `json:"passed"`
	Models             []modelProbeEvidence `json:"models"`
}

type recipeEvaluationInput struct {
	RecipeID     string `json:"recipeId"`
	EntrypointID string `json:"entrypointId"`
}

type readinessGate struct {
	Name    string `json:"name"`
	Passed  bool   `json:"passed"`
	Message string `json:"message"`
}

type recipeEvaluationOutput struct {
	ArtifactID         string          `json:"artifactId"`
	RecipeRevision     int64           `json:"recipeRevision"`
	EntrypointRevision int64           `json:"entrypointRevision"`
	Passed             bool            `json:"passed"`
	Gates              []readinessGate `json:"gates"`
}

type workflowPath struct {
	recipe     routingmanagement.Recipe
	entrypoint routingmanagement.Entrypoint
	modelIDs   []string
	coverage   map[string]bool
}

func (provider *Provider) probeRecipe(
	ctx context.Context,
	invocation agentmanagement.ToolInvocationContext,
	raw json.RawMessage,
) (agentmanagement.ToolResult, error) {
	var input recipeProbeInput
	if err := json.Unmarshal(raw, &input); err != nil || input.RecipeID == "" || input.EntrypointID == "" ||
		input.TimeoutSeconds < 0 || input.TimeoutSeconds > 60 {
		return agentmanagement.ToolResult{}, agentmanagement.ErrInvalid
	}
	path, probeRecipeErr := provider.resolveWorkflowPath(ctx, invocation, input.RecipeID, input.EntrypointID)
	if probeRecipeErr != nil {
		return agentmanagement.ToolResult{}, probeRecipeErr
	}
	if err := provider.authorizeResources(
		ctx, invocation, accesscontrol.PermissionEvaluationRun,
		accesscontrol.ScopeResourceRecipe, path.recipe.ID,
	); err != nil {
		return agentmanagement.ToolResult{}, err
	}
	timeout := time.Duration(input.TimeoutSeconds) * time.Second
	if timeout == 0 {
		timeout = 15 * time.Second
	}
	evidence := make([]modelProbeEvidence, 0, len(path.modelIDs))
	passed := true
	for _, modelID := range path.modelIDs {
		model, getErr := provider.routing.GetModel(ctx, invocation.NamespaceID, modelID)
		if getErr != nil {
			return agentmanagement.ToolResult{}, mapRoutingError(getErr)
		}
		result, probeErr := provider.routing.ProbeModel(ctx, invocation.NamespaceID, modelID, timeout)
		item := modelProbeEvidence{ModelID: model.ID, ModelName: model.Name}
		if probeErr == nil {
			item.Available = result.Available
			item.LatencyMilliseconds = result.Latency.Milliseconds()
		} else {
			item.ErrorCode = "probe_failed"
		}
		if !item.Available {
			passed = false
		}
		evidence = append(evidence, item)
	}
	content, probeRecipeErr := json.Marshal(struct {
		InvocationID       string               `json:"invocationId"`
		RecipeID           string               `json:"recipeId"`
		RecipeRevision     int64                `json:"recipeRevision"`
		EntrypointID       string               `json:"entrypointId"`
		EntrypointRevision int64                `json:"entrypointRevision"`
		Passed             bool                 `json:"passed"`
		Models             []modelProbeEvidence `json:"models"`
	}{
		invocation.InvocationID, path.recipe.ID, path.recipe.Current.Revision,
		path.entrypoint.ID, path.entrypoint.Current.Revision, passed, evidence,
	})
	if probeRecipeErr != nil {
		return agentmanagement.ToolResult{}, agentmanagement.ErrInvalid
	}
	preview, _ := json.Marshal(map[string]any{
		"passed": passed, "modelCount": len(evidence),
	})
	artifact, probeRecipeErr := provider.putWorkflowArtifact(
		ctx, invocation, "probe", content, preview,
	)
	if probeRecipeErr != nil {
		return agentmanagement.ToolResult{}, probeRecipeErr
	}
	value, probeRecipeErr := json.Marshal(recipeProbeOutput{
		ArtifactID:         artifact.ID,
		RecipeRevision:     path.recipe.Current.Revision,
		EntrypointRevision: path.entrypoint.Current.Revision,
		Passed:             passed, Models: evidence,
	})
	if probeRecipeErr != nil {
		return agentmanagement.ToolResult{}, agentmanagement.ErrInvalid
	}
	return agentmanagement.ToolResult{Value: value, ArtifactID: artifact.ID}, nil
}

func (provider *Provider) evaluateRecipe(
	ctx context.Context,
	invocation agentmanagement.ToolInvocationContext,
	raw json.RawMessage,
) (agentmanagement.ToolResult, error) {
	var input recipeEvaluationInput
	if err := json.Unmarshal(raw, &input); err != nil || input.RecipeID == "" || input.EntrypointID == "" {
		return agentmanagement.ToolResult{}, agentmanagement.ErrInvalid
	}
	path, evaluateRecipeErr := provider.resolveWorkflowPath(ctx, invocation, input.RecipeID, input.EntrypointID)
	if evaluateRecipeErr != nil {
		return agentmanagement.ToolResult{}, evaluateRecipeErr
	}
	if err := provider.authorizeResources(
		ctx, invocation, accesscontrol.PermissionEvaluationRun,
		accesscontrol.ScopeResourceRecipe, path.recipe.ID,
	); err != nil {
		return agentmanagement.ToolResult{}, err
	}
	gates := evaluateReadiness(path)
	passed := true
	for _, gate := range gates {
		passed = passed && gate.Passed
	}
	content, evaluateRecipeErr := json.Marshal(struct {
		InvocationID       string          `json:"invocationId"`
		RecipeID           string          `json:"recipeId"`
		RecipeRevision     int64           `json:"recipeRevision"`
		EntrypointID       string          `json:"entrypointId"`
		EntrypointRevision int64           `json:"entrypointRevision"`
		Passed             bool            `json:"passed"`
		Gates              []readinessGate `json:"gates"`
	}{
		invocation.InvocationID, path.recipe.ID, path.recipe.Current.Revision,
		path.entrypoint.ID, path.entrypoint.Current.Revision, passed, gates,
	})
	if evaluateRecipeErr != nil {
		return agentmanagement.ToolResult{}, agentmanagement.ErrInvalid
	}
	preview, _ := json.Marshal(map[string]any{
		"passed": passed, "gateCount": len(gates),
	})
	artifact, evaluateRecipeErr := provider.putWorkflowArtifact(
		ctx, invocation, "evaluation", content, preview,
	)
	if evaluateRecipeErr != nil {
		return agentmanagement.ToolResult{}, evaluateRecipeErr
	}
	value, evaluateRecipeErr := json.Marshal(recipeEvaluationOutput{
		ArtifactID:         artifact.ID,
		RecipeRevision:     path.recipe.Current.Revision,
		EntrypointRevision: path.entrypoint.Current.Revision,
		Passed:             passed, Gates: gates,
	})
	if evaluateRecipeErr != nil {
		return agentmanagement.ToolResult{}, agentmanagement.ErrInvalid
	}
	return agentmanagement.ToolResult{Value: value, ArtifactID: artifact.ID}, nil
}

func (provider *Provider) resolveWorkflowPath(
	ctx context.Context,
	invocation agentmanagement.ToolInvocationContext,
	recipeID, entrypointID string,
) (workflowPath, error) {
	recipe, err := provider.routing.GetRecipe(ctx, invocation.NamespaceID, recipeID)
	if err != nil {
		return workflowPath{}, mapRoutingError(err)
	}
	entrypoint, err := provider.routing.GetEntrypoint(ctx, invocation.NamespaceID, entrypointID)
	if err != nil {
		return workflowPath{}, mapRoutingError(err)
	}
	if recipe.Status == routingmanagement.StatusDisabled || recipe.Status == routingmanagement.StatusDeleted ||
		entrypoint.Status == routingmanagement.StatusDisabled || entrypoint.Status == routingmanagement.StatusDeleted {
		return workflowPath{}, agentmanagement.ErrNotFound
	}
	if err := provider.authorizeResources(
		ctx, invocation, accesscontrol.PermissionRoutingRead,
		accesscontrol.ScopeResourceRecipe, recipe.ID,
	); err != nil {
		return workflowPath{}, err
	}
	if err := provider.authorizeResources(
		ctx, invocation, accesscontrol.PermissionRoutingRead,
		accesscontrol.ScopeResourceEntrypoint, entrypoint.ID,
	); err != nil {
		return workflowPath{}, err
	}
	models := make(map[string]struct{})
	coverage := make(map[string]bool, len(recipe.Current.Decisions))
	for _, decision := range recipe.Current.Decisions {
		coverage[decision.ID] = false
	}
	referencesRecipe := false
	for _, rule := range entrypoint.Current.Rules {
		if rule.RecipeID != recipe.ID {
			continue
		}
		referencesRecipe = true
		if rule.RecipeRevision != recipe.Current.Revision {
			return workflowPath{}, agentmanagement.ErrConflict
		}
		for decisionID, assignment := range rule.Assignments {
			if len(assignment.Models) > 0 {
				coverage[decisionID] = true
			}
			for _, model := range assignment.Models {
				models[model.ModelID] = struct{}{}
			}
		}
	}
	if !referencesRecipe || len(models) == 0 {
		return workflowPath{}, agentmanagement.ErrInvalid
	}
	modelIDs := sortedKeys(models)
	for _, modelID := range modelIDs {
		model, err := provider.routing.GetModel(ctx, invocation.NamespaceID, modelID)
		if err != nil || model.Status == routingmanagement.StatusDisabled ||
			model.Status == routingmanagement.StatusDeleted {
			if err != nil {
				return workflowPath{}, mapRoutingError(err)
			}
			return workflowPath{}, agentmanagement.ErrNotFound
		}
	}
	if err := provider.authorizeResources(
		ctx, invocation, accesscontrol.PermissionRoutingRead,
		accesscontrol.ScopeResourceModel, modelIDs...,
	); err != nil {
		return workflowPath{}, err
	}
	return workflowPath{
		recipe: recipe, entrypoint: entrypoint, modelIDs: modelIDs, coverage: coverage,
	}, nil
}

func evaluateReadiness(path workflowPath) []readinessGate {
	coveragePassed := len(path.coverage) > 0
	for _, covered := range path.coverage {
		coveragePassed = coveragePassed && covered
	}
	gates := []readinessGate{
		{
			Name: "recipe_compiles", Passed: len(path.recipe.Current.Decisions) > 0,
			Message: "Recipe has a compiled decision graph.",
		},
		{
			Name: "decision_coverage", Passed: coveragePassed,
			Message: "Every reachable decision has an explicit Model assignment.",
		},
		{
			Name: "models_connected", Passed: len(path.modelIDs) > 0,
			Message: "The Entrypoint resolves to connected Models.",
		},
		{
			Name: "revisions_pinned", Passed: true,
			Message: "Recipe and Entrypoint revisions are pinned.",
		},
	}
	return gates
}

func (provider *Provider) putWorkflowArtifact(
	ctx context.Context,
	invocation agentmanagement.ToolInvocationContext,
	kind string,
	content, preview json.RawMessage,
) (agentmanagement.Artifact, error) {
	artifactID := uuid.NewSHA1(
		uuid.MustParse(invocation.InvocationID), []byte("router-agent-artifact/"+kind),
	).String()
	accessScope, err := json.Marshal(map[string]any{
		"principalId":      invocation.PrincipalID,
		"authorityDigest":  invocation.AuthorityDigest,
		"registryRevision": invocation.RegistryRevision,
	})
	if err != nil {
		return agentmanagement.Artifact{}, agentmanagement.ErrInvalid
	}
	artifact, err := provider.store.PutArtifact(ctx, invocation.NamespaceID, agentmanagement.Artifact{
		ID: artifactID, SessionID: invocation.SessionID, TurnID: invocation.TurnID,
		Kind: kind, MediaType: "application/json", Content: append([]byte(nil), content...),
		SafePreview: append(json.RawMessage(nil), preview...),
		ExpiresAt:   provider.now().UTC().Add(artifactRetention),
	}, accessScope)
	if err == nil {
		return artifact, nil
	}
	if !errors.Is(err, agentmanagement.ErrConflict) {
		return agentmanagement.Artifact{}, err
	}
	existing, getErr := provider.store.GetArtifact(ctx, invocation.NamespaceID, artifactID)
	if getErr != nil || existing.SessionID != invocation.SessionID || existing.TurnID != invocation.TurnID ||
		existing.Kind != kind || !bytes.Equal(existing.Content, content) {
		return agentmanagement.Artifact{}, agentmanagement.ErrConflict
	}
	return existing, nil
}
