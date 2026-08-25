package agentworkflow

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/google/uuid"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/agentmanagement"
)

const publicationPlanLifetime = 30 * time.Minute

type publishPrepareInput struct {
	RecipeID                   string `json:"recipeId"`
	RecipeContentRevision      int64  `json:"recipeContentRevision" jsonschema:"minimum=1"`
	RecipeResourceRevision     int64  `json:"recipeResourceRevision" jsonschema:"minimum=1"`
	EntrypointID               string `json:"entrypointId"`
	EntrypointContentRevision  int64  `json:"entrypointContentRevision" jsonschema:"minimum=1"`
	EntrypointResourceRevision int64  `json:"entrypointResourceRevision" jsonschema:"minimum=1"`
	ProbeArtifactID            string `json:"probeArtifactId"`
	EvaluationArtifactID       string `json:"evaluationArtifactId"`
}

type publishPrepareOutput struct {
	Approval agentmanagement.ApprovalRequestEvent `json:"approval"`
}

type probeArtifactEvidence struct {
	RecipeID           string               `json:"recipeId"`
	RecipeRevision     int64                `json:"recipeRevision"`
	EntrypointID       string               `json:"entrypointId"`
	EntrypointRevision int64                `json:"entrypointRevision"`
	Passed             bool                 `json:"passed"`
	Models             []modelProbeEvidence `json:"models"`
}

type evaluationArtifactEvidence struct {
	RecipeID           string          `json:"recipeId"`
	RecipeRevision     int64           `json:"recipeRevision"`
	EntrypointID       string          `json:"entrypointId"`
	EntrypointRevision int64           `json:"entrypointRevision"`
	Passed             bool            `json:"passed"`
	Gates              []readinessGate `json:"gates"`
}

func (provider *Provider) preparePublication(
	ctx context.Context,
	invocation agentmanagement.ToolInvocationContext,
	raw json.RawMessage,
) (agentmanagement.ToolResult, error) {
	input, err := decodePublishPrepareInput(raw)
	if err != nil {
		return agentmanagement.ToolResult{}, err
	}
	path, preparePublicationErr := provider.resolveWorkflowPath(ctx, invocation, input.RecipeID, input.EntrypointID)
	if preparePublicationErr != nil {
		return agentmanagement.ToolResult{}, preparePublicationErr
	}
	if path.recipe.Current.Revision != input.RecipeContentRevision ||
		path.recipe.Revision != input.RecipeResourceRevision ||
		path.entrypoint.Current.Revision != input.EntrypointContentRevision ||
		path.entrypoint.Revision != input.EntrypointResourceRevision {
		return agentmanagement.ToolResult{}, agentmanagement.ErrConflict
	}
	if err := provider.authorizePublicationPath(ctx, invocation, path); err != nil {
		return agentmanagement.ToolResult{}, err
	}
	probe, evaluation, preparePublicationErr := provider.loadPublicationEvidence(ctx, invocation, input)
	if preparePublicationErr != nil {
		return agentmanagement.ToolResult{}, preparePublicationErr
	}
	serverGates := evaluateReadiness(path)
	for _, gate := range serverGates {
		if !gate.Passed {
			return agentmanagement.ToolResult{}, agentmanagement.ErrApproval
		}
	}
	gateResults := append([]readinessGate(nil), evaluation.Gates...)
	gateResults = append(gateResults, readinessGate{
		Name: "model_probes", Passed: probe.Passed,
		Message: "Every assigned Model passed the pinned connectivity probe.",
	})
	for _, gate := range gateResults {
		if !gate.Passed {
			return agentmanagement.ToolResult{}, agentmanagement.ErrApproval
		}
	}

	topology, assignments, exactDiff, diagnostics, encodedGates, preparePublicationErr := publicationReview(path, input, probe, gateResults)
	if preparePublicationErr != nil {
		return agentmanagement.ToolResult{}, preparePublicationErr
	}
	planID := uuid.NewSHA1(
		uuid.MustParse(invocation.InvocationID),
		[]byte("router-agent-publication-plan/v1"),
	).String()
	planInput := agentmanagement.PublicationPlan{
		ID: planID, SessionID: invocation.SessionID, TurnID: invocation.TurnID,
		RecipeID:                   path.recipe.ID,
		RecipeContentRevision:      path.recipe.Current.Revision,
		RecipeResourceRevision:     path.recipe.Revision,
		EntrypointID:               path.entrypoint.ID,
		EntrypointContentRevision:  path.entrypoint.Current.Revision,
		EntrypointResourceRevision: path.entrypoint.Revision,
		CatalogRevision:            invocation.RegistryRevision,
		ExactDiff:                  exactDiff, Diagnostics: diagnostics, GateResults: encodedGates,
		ExpiresAt: provider.now().UTC().Add(publicationPlanLifetime),
	}
	plan, preparePublicationErr := provider.store.CreatePublicationPlan(
		ctx, invocation.NamespaceID, planInput, agentmanagement.MutationContext{
			PrincipalID: invocation.PrincipalID,
			ActorChain:  []string{invocation.PrincipalID},
			RequestID:   "agent-tool-" + invocation.InvocationID,
			Reason:      "prepare immutable Agent publication review",
		},
	)
	if errors.Is(preparePublicationErr, agentmanagement.ErrConflict) {
		plan, preparePublicationErr = provider.store.GetPublicationPlan(ctx, invocation.NamespaceID, planID)
		if preparePublicationErr == nil && !samePublicationPlan(plan, planInput) {
			preparePublicationErr = agentmanagement.ErrConflict
		}
	}
	if preparePublicationErr != nil {
		return agentmanagement.ToolResult{}, preparePublicationErr
	}
	approval := agentmanagement.ApprovalRequestEvent{
		PlanID: plan.ID, PlanDigest: plan.Digest, PlanRevision: plan.Revision,
		PlanETag: fmt.Sprintf(`"agent:%d"`, plan.Revision), ExpiresAt: plan.ExpiresAt,
		Summary: agentmanagement.PublicationSummary{
			RecipeID: path.recipe.ID, RecipeName: path.recipe.Name,
			EntrypointID: path.entrypoint.ID, EntrypointName: path.entrypoint.Name,
			ChangedResources: []string{"recipe", "entrypoint"},
			Topology:         topology, Assignments: assignments, GateResults: encodedGates,
		},
	}
	value, preparePublicationErr := json.Marshal(publishPrepareOutput{Approval: approval})
	if preparePublicationErr != nil {
		return agentmanagement.ToolResult{}, agentmanagement.ErrInvalid
	}
	return agentmanagement.ToolResult{Value: value}, nil
}

func decodePublishPrepareInput(raw json.RawMessage) (publishPrepareInput, error) {
	var input publishPrepareInput
	if err := json.Unmarshal(raw, &input); err != nil || input.RecipeID == "" ||
		input.EntrypointID == "" || input.RecipeContentRevision < 1 ||
		input.RecipeResourceRevision < 1 || input.EntrypointContentRevision < 1 ||
		input.EntrypointResourceRevision < 1 || uuid.Validate(input.ProbeArtifactID) != nil ||
		uuid.Validate(input.EvaluationArtifactID) != nil {
		return publishPrepareInput{}, agentmanagement.ErrInvalid
	}
	return input, nil
}

func (provider *Provider) authorizePublicationPath(
	ctx context.Context, invocation agentmanagement.ToolInvocationContext, path workflowPath,
) error {
	if err := provider.authorizeResources(
		ctx, invocation, accesscontrol.PermissionRoutingManage,
		accesscontrol.ScopeResourceRecipe, path.recipe.ID,
	); err != nil {
		return err
	}
	return provider.authorizeResources(
		ctx, invocation, accesscontrol.PermissionRoutingManage,
		accesscontrol.ScopeResourceEntrypoint, path.entrypoint.ID,
	)
}

func (provider *Provider) loadPublicationEvidence(
	ctx context.Context,
	invocation agentmanagement.ToolInvocationContext,
	input publishPrepareInput,
) (probeArtifactEvidence, evaluationArtifactEvidence, error) {
	probeArtifact, err := provider.store.GetArtifact(ctx, invocation.NamespaceID, input.ProbeArtifactID)
	if err != nil {
		return probeArtifactEvidence{}, evaluationArtifactEvidence{}, err
	}
	evaluationArtifact, err := provider.store.GetArtifact(
		ctx, invocation.NamespaceID, input.EvaluationArtifactID,
	)
	if err != nil {
		return probeArtifactEvidence{}, evaluationArtifactEvidence{}, err
	}
	for _, artifact := range []agentmanagement.Artifact{probeArtifact, evaluationArtifact} {
		if artifact.SessionID != invocation.SessionID || artifact.TurnID != invocation.TurnID ||
			artifact.Digest == "" || !provider.now().UTC().Before(artifact.ExpiresAt.UTC()) {
			return probeArtifactEvidence{}, evaluationArtifactEvidence{}, agentmanagement.ErrApproval
		}
	}
	if probeArtifact.Kind != "probe" || evaluationArtifact.Kind != "evaluation" {
		return probeArtifactEvidence{}, evaluationArtifactEvidence{}, agentmanagement.ErrApproval
	}
	var probe probeArtifactEvidence
	var evaluation evaluationArtifactEvidence
	if err := json.Unmarshal(probeArtifact.Content, &probe); err != nil {
		return probeArtifactEvidence{}, evaluationArtifactEvidence{}, agentmanagement.ErrConflict
	}
	if err := json.Unmarshal(evaluationArtifact.Content, &evaluation); err != nil {
		return probeArtifactEvidence{}, evaluationArtifactEvidence{}, agentmanagement.ErrConflict
	}
	if !probe.Passed || !evaluation.Passed || len(probe.Models) == 0 || len(evaluation.Gates) == 0 ||
		probe.RecipeID != input.RecipeID || evaluation.RecipeID != input.RecipeID ||
		probe.EntrypointID != input.EntrypointID || evaluation.EntrypointID != input.EntrypointID ||
		probe.RecipeRevision != input.RecipeContentRevision ||
		evaluation.RecipeRevision != input.RecipeContentRevision ||
		probe.EntrypointRevision != input.EntrypointContentRevision ||
		evaluation.EntrypointRevision != input.EntrypointContentRevision {
		return probeArtifactEvidence{}, evaluationArtifactEvidence{}, agentmanagement.ErrApproval
	}
	return probe, evaluation, nil
}

func publicationReview(
	path workflowPath,
	input publishPrepareInput,
	probe probeArtifactEvidence,
	gates []readinessGate,
) (json.RawMessage, json.RawMessage, json.RawMessage, json.RawMessage, json.RawMessage, error) {
	topology, err := json.Marshal(map[string]any{
		"decisions": path.recipe.Current.Decisions,
	})
	if err != nil {
		return nil, nil, nil, nil, nil, agentmanagement.ErrInvalid
	}
	assignments, err := json.Marshal(map[string]any{
		"rules": path.entrypoint.Current.Rules,
	})
	if err != nil {
		return nil, nil, nil, nil, nil, agentmanagement.ErrInvalid
	}
	exactDiff, err := json.Marshal(map[string]any{
		"recipe": map[string]any{
			"id":               path.recipe.ID,
			"contentRevision":  path.recipe.Current.Revision,
			"resourceRevision": path.recipe.Revision,
		},
		"entrypoint": map[string]any{
			"id":               path.entrypoint.ID,
			"contentRevision":  path.entrypoint.Current.Revision,
			"resourceRevision": path.entrypoint.Revision,
		},
		"evidence": map[string]any{
			"probeArtifactId":      input.ProbeArtifactID,
			"evaluationArtifactId": input.EvaluationArtifactID,
			"probedModels":         probe.Models,
		},
		"topology":    json.RawMessage(topology),
		"assignments": json.RawMessage(assignments),
	})
	if err != nil {
		return nil, nil, nil, nil, nil, agentmanagement.ErrInvalid
	}
	diagnostics := json.RawMessage(`[]`)
	gateResults, err := json.Marshal(gates)
	if err != nil {
		return nil, nil, nil, nil, nil, agentmanagement.ErrInvalid
	}
	return topology, assignments, exactDiff, diagnostics, gateResults, nil
}

func samePublicationPlan(current, expected agentmanagement.PublicationPlan) bool {
	return current.ID == expected.ID && current.SessionID == expected.SessionID &&
		current.TurnID == expected.TurnID && current.RecipeID == expected.RecipeID &&
		current.RecipeContentRevision == expected.RecipeContentRevision &&
		current.RecipeResourceRevision == expected.RecipeResourceRevision &&
		current.EntrypointID == expected.EntrypointID &&
		current.EntrypointContentRevision == expected.EntrypointContentRevision &&
		current.EntrypointResourceRevision == expected.EntrypointResourceRevision &&
		current.CatalogRevision == expected.CatalogRevision && current.Digest != "" &&
		(current.Status == agentmanagement.PublicationReady ||
			current.Status == agentmanagement.PublicationPublishing ||
			current.Status == agentmanagement.PublicationCommitted)
}
