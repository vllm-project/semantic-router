package agentruntime

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"time"

	"github.com/google/uuid"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/agentmanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

type publishPrepareOutput struct {
	Approval agentmanagement.ApprovalRequestEvent `json:"approval"`
}

type modelStepPreparation struct {
	credential    []byte
	tools         []llmprotocol.Tool
	projection    executionContext
	request       llmprotocol.Request
	requestDigest []byte
	ordinal       int64
	id            string
}

func (worker *Worker) executeTurn(
	ctx context.Context, lease agentmanagement.TurnLease,
) (agentmanagement.TurnTransition, error) {
	turn, executeTurnErr := worker.store.GetTurn(ctx, lease.NamespaceID, lease.SessionID, lease.TurnID)
	if executeTurnErr != nil || turn.RegistryRevision != lease.RegistryRevision {
		if executeTurnErr != nil {
			return agentmanagement.TurnTransition{}, executeTurnErr
		}
		return agentmanagement.TurnTransition{}, agentmanagement.ErrConflict
	}
	session, executeTurnErr := worker.store.GetSession(ctx, lease.NamespaceID, lease.SessionID)
	if executeTurnErr != nil {
		return agentmanagement.TurnTransition{}, executeTurnErr
	}
	if session.Status != agentmanagement.SessionActive {
		return agentmanagement.TurnTransition{}, agentmanagement.ErrDenied
	}
	currentProfile, executeTurnErr := worker.store.GetProfile(ctx, lease.NamespaceID, session.ProfileID)
	if executeTurnErr != nil || currentProfile.Status != agentmanagement.StatusActive {
		if executeTurnErr != nil {
			return agentmanagement.TurnTransition{}, executeTurnErr
		}
		return agentmanagement.TurnTransition{}, agentmanagement.ErrDenied
	}
	profile, executeTurnErr := worker.store.GetProfileRevision(
		ctx, lease.NamespaceID, session.ProfileID, session.ProfileRevision,
	)
	if executeTurnErr != nil {
		return agentmanagement.TurnTransition{}, executeTurnErr
	}
	if err := worker.authority.Reauthorize(ctx, session, profile.MinimumTargetCapabilities); err != nil {
		return agentmanagement.TurnTransition{}, err
	}
	registry, executeTurnErr := worker.registries.Load(ctx, lease.NamespaceID, lease.RegistryRevision)
	if executeTurnErr != nil {
		return agentmanagement.TurnTransition{}, executeTurnErr
	}
	turnContext, cancel := context.WithTimeout(ctx, time.Duration(profile.MaximumTurnSeconds)*time.Second)
	defer cancel()

	for {
		if err := worker.ensureNotCancelled(turnContext, lease); err != nil {
			return agentmanagement.TurnTransition{}, err
		}
		projection, compacted, err := worker.loadExecutionContext(turnContext, lease, profile)
		if err != nil {
			return agentmanagement.TurnTransition{}, err
		}
		if compacted {
			projection, err = worker.commitExecutionCheckpoint(turnContext, lease, projection)
			if err != nil {
				return agentmanagement.TurnTransition{}, err
			}
		}
		if projection.ToolSteps > profile.MaximumToolSteps {
			return agentmanagement.TurnTransition{}, fmt.Errorf("%w: Agent tool step limit reached", agentmanagement.ErrDenied)
		}
		if len(projection.Pending) == 0 && terminalModelStop(projection.LastModelStopReason) {
			return agentmanagement.TurnTransition{
				Lease: lease, Status: agentmanagement.TurnCompleted, CompletedAt: worker.now().UTC(),
			}, nil
		}
		if len(projection.Pending) > 0 {
			for _, call := range projection.Pending {
				approval, invokeErr := worker.executeTool(
					turnContext, lease, session, profile, registry, call,
				)
				if invokeErr != nil {
					return agentmanagement.TurnTransition{}, invokeErr
				}
				updated, _, loadErr := worker.loadExecutionContext(turnContext, lease, profile)
				if loadErr != nil {
					return agentmanagement.TurnTransition{}, loadErr
				}
				if _, checkpointErr := worker.commitExecutionCheckpoint(turnContext, lease, updated); checkpointErr != nil {
					return agentmanagement.TurnTransition{}, checkpointErr
				}
				if approval != nil {
					return agentmanagement.TurnTransition{
						Lease: lease, Status: agentmanagement.TurnWaitingApproval, Approval: approval,
					}, nil
				}
			}
			continue
		}
		if projection.ToolSteps >= profile.MaximumToolSteps {
			return agentmanagement.TurnTransition{}, fmt.Errorf("%w: Agent tool step limit reached", agentmanagement.ErrDenied)
		}
		err = worker.runModelStep(
			turnContext, lease, session, profile, registry, projection,
		)
		if err != nil {
			return agentmanagement.TurnTransition{}, err
		}
	}
}

func (worker *Worker) runModelStep(
	ctx context.Context,
	lease agentmanagement.TurnLease,
	session agentmanagement.Session,
	profile agentmanagement.Profile,
	registry *agentmanagement.ToolRegistry,
	projection executionContext,
) error {
	prepared, err := worker.prepareModelStep(ctx, lease, session, profile, registry, projection)
	if err != nil {
		return err
	}
	defer clear(prepared.credential)
	step, replayed, err := worker.store.BeginModelStep(ctx, agentmanagement.ModelStep{
		ID: prepared.id, NamespaceID: lease.NamespaceID, SessionID: lease.SessionID,
		TurnID: lease.TurnID, Ordinal: prepared.ordinal, WorkerID: lease.WorkerID, Fence: lease.Fence,
		RegistryRevision: lease.RegistryRevision, RequestDigest: prepared.requestDigest,
	})
	if err != nil {
		return err
	}
	if replayed {
		if step.Status == "completed" {
			return nil
		}
		return fmt.Errorf("%w: public inference outcome is unknown and cannot be repeated", agentmanagement.ErrConflict)
	}
	collector := newModelStepCollector(
		ctx, worker, lease, registry, profile.ToolPolicy, prepared.id, prepared.projection.ToolSteps,
	)
	committedLiveOutput := false
	defer func() {
		if !committedLiveOutput {
			collector.publishLiveTerminal(agentmanagement.LiveModelStepDiscarded)
		}
	}()
	observation, generateErr := worker.inference.Generate(
		ctx, prepared.credential, prepared.request, collector.consume,
	)
	if generateErr != nil {
		return wrapModelStepStageFailure(modelStepStageInferenceStream, generateErr)
	}
	if observeErr := collector.observe(observation); observeErr != nil {
		return wrapModelStepStageFailure(modelStepStageRouterObservation, observeErr)
	}
	output, err := collector.finish()
	if err != nil {
		return wrapModelStepStageFailure(modelStepStageFinish, err)
	}
	if commitErr := worker.commitModelStep(ctx, lease, profile, prepared, output); commitErr != nil {
		return wrapModelStepStageFailure(modelStepStageCommit, commitErr)
	}
	committedLiveOutput = true
	collector.publishLiveTerminal(agentmanagement.LiveModelStepCommitted)
	return nil
}

func (worker *Worker) prepareModelStep(
	ctx context.Context,
	lease agentmanagement.TurnLease,
	session agentmanagement.Session,
	profile agentmanagement.Profile,
	registry *agentmanagement.ToolRegistry,
	projection executionContext,
) (modelStepPreparation, error) {
	if err := worker.authority.Reauthorize(ctx, session, profile.MinimumTargetCapabilities); err != nil {
		return modelStepPreparation{}, err
	}
	if err := worker.authority.RenewDelegation(ctx, session, worker.delegationTTL); err != nil {
		return modelStepPreparation{}, err
	}
	credential, err := worker.authority.ResolveInferenceCredential(ctx, session)
	if err != nil {
		return modelStepPreparation{}, err
	}
	fail := func(err error) (modelStepPreparation, error) {
		clear(credential)
		return modelStepPreparation{}, err
	}
	definitions := registry.Definitions(profile.ToolPolicy)
	tools := make([]llmprotocol.Tool, 0, len(definitions))
	for _, definition := range definitions {
		strict := true
		tools = append(tools, llmprotocol.Tool{
			Name: definition.Name, Description: definition.Description,
			Strict: &strict, InputSchema: append(json.RawMessage(nil), definition.InputSchema...),
		})
	}
	projection, compacted, err := fitExecutionContext(
		projection, profile.ContextTokenBudget, tools,
	)
	if err != nil {
		return fail(err)
	}
	if compacted {
		projection, err = worker.commitExecutionCheckpoint(ctx, lease, projection)
		if err != nil {
			return fail(err)
		}
	}
	parallel := false
	request := llmprotocol.Request{
		Model: session.Target.ID, Instructions: projection.Instructions, Messages: projection.Messages,
		Tools: tools, ToolChoice: llmprotocol.ToolChoice{Mode: llmprotocol.ToolChoiceAuto},
		ParallelToolCalls: &parallel, Stream: true,
	}
	requestDigest, err := canonicalModelStepRequest(request, profile, lease.RegistryRevision)
	if err != nil {
		return fail(err)
	}
	stepOrdinal := int64(projection.ModelSteps + 1)
	stepID := deterministicModelStepID(lease.TurnID, stepOrdinal)
	return modelStepPreparation{
		credential: credential, tools: tools, projection: projection, request: request,
		requestDigest: requestDigest, ordinal: stepOrdinal, id: stepID,
	}, nil
}

func (worker *Worker) commitModelStep(
	ctx context.Context,
	lease agentmanagement.TurnLease,
	profile agentmanagement.Profile,
	prepared modelStepPreparation,
	output modelStepOutput,
) error {
	next, err := projectModelStepOutput(prepared.projection, output)
	if err != nil {
		return err
	}
	next, _, err = fitExecutionContext(next, profile.ContextTokenBudget, prepared.tools)
	if err != nil {
		return err
	}
	checkpoint, err := encodeExecutionCheckpoint(lease, next)
	if err != nil {
		return err
	}
	committed, err := worker.store.CommitModelStep(ctx, agentmanagement.ModelStepCommit{
		Lease: lease,
		Step: agentmanagement.ModelStep{
			ID: prepared.id, Ordinal: prepared.ordinal, Fence: lease.Fence,
			RegistryRevision: lease.RegistryRevision, RequestDigest: prepared.requestDigest,
			StopReason: string(output.StopReason),
		},
		Events: output.Events, Checkpoint: checkpoint,
	})
	if err != nil {
		return err
	}
	for _, event := range committed.Events {
		worker.notifyEvent(ctx, lease, event)
	}
	worker.notifyEvent(ctx, lease, committed.CheckpointEvent)
	return nil
}

func terminalModelStop(reason llmprotocol.StopReason) bool {
	switch reason {
	case llmprotocol.StopEndTurn, llmprotocol.StopMaxTokens, llmprotocol.StopSequence,
		llmprotocol.StopContentFilter:
		return true
	default:
		return false
	}
}

func canonicalModelStepRequest(
	request llmprotocol.Request, profile agentmanagement.Profile, registryRevision string,
) ([]byte, error) {
	encoded, err := json.Marshal(struct {
		Request          llmprotocol.Request `json:"request"`
		ProfileID        string              `json:"profileId"`
		ProfileRevision  int64               `json:"profileRevision"`
		RegistryRevision string              `json:"registryRevision"`
	}{request, profile.ID, profile.ContentRevision, registryRevision})
	if err != nil {
		return nil, fmt.Errorf("encode Agent model step: %w", err)
	}
	digest := sha256.Sum256(encoded)
	return digest[:], nil
}

func projectModelStepOutput(
	projection executionContext, output modelStepOutput,
) (executionContext, error) {
	next := projection
	next.Messages = append([]llmprotocol.Message(nil), projection.Messages...)
	next.Pending = append([]pendingToolCall(nil), projection.Pending...)
	events := make([]agentmanagement.Event, 0, len(output.Events))
	sequence := projection.ThroughSequence
	for _, appendRequest := range output.Events {
		normalized, err := agentmanagement.NormalizeEventAppend(appendRequest)
		if err != nil {
			return executionContext{}, err
		}
		sequence++
		events = append(events, agentmanagement.Event{
			SessionID: appendRequest.SessionID, TurnID: appendRequest.TurnID,
			Sequence: sequence, Type: normalized.Type, Payload: normalized.Payload,
		})
	}
	if err := projectEvents(&next, events); err != nil {
		return executionContext{}, err
	}
	next.ModelSteps++
	next.LastModelStopReason = output.StopReason
	return next, nil
}

func (worker *Worker) executeTool(
	ctx context.Context,
	lease agentmanagement.TurnLease,
	session agentmanagement.Session,
	profile agentmanagement.Profile,
	registry *agentmanagement.ToolRegistry,
	call pendingToolCall,
) (*agentmanagement.ApprovalRequestEvent, error) {
	if err := worker.ensureNotCancelled(ctx, lease); err != nil {
		return nil, err
	}
	currentProfile, executeToolErr := worker.store.GetProfile(ctx, lease.NamespaceID, profile.ID)
	if executeToolErr != nil || currentProfile.Status != agentmanagement.StatusActive {
		if executeToolErr != nil {
			return nil, executeToolErr
		}
		return nil, agentmanagement.ErrDenied
	}
	if err := worker.authority.Reauthorize(ctx, session, profile.MinimumTargetCapabilities); err != nil {
		return nil, err
	}
	definition, origin, found := registry.Definition(call.Name, profile.ToolPolicy)
	if !found {
		return nil, agentmanagement.ErrToolUnavailable
	}
	invocationContext := agentmanagement.ToolInvocationContext{
		NamespaceID: lease.NamespaceID, PrincipalID: session.OwnerPrincipalID,
		SessionID: lease.SessionID, TurnID: lease.TurnID, InvocationID: call.InvocationID,
		Target: session.Target,
	}
	cleanInput, executeToolErr := registry.ScrubInvocationInput(
		ctx, lease.RegistryRevision, profile.ToolPolicy, call.Name, call.Arguments,
	)
	if executeToolErr != nil {
		return nil, executeToolErr
	}
	digest := sha256.Sum256(cleanInput)
	record := agentmanagement.InvocationRecord{
		ID: call.InvocationID, NamespaceID: lease.NamespaceID, SessionID: lease.SessionID,
		TurnID: lease.TurnID, Fence: lease.Fence, RegistryRevision: lease.RegistryRevision,
		ToolName: call.Name, CredentialVersionID: origin.CredentialVersionID,
		InputDigest: digest[:], Input: cleanInput,
		Idempotency: definition.Idempotency, Class: definition.Class,
	}
	existing, replayed, executeToolErr := worker.store.BeginInvocation(ctx, record)
	if executeToolErr != nil {
		return nil, executeToolErr
	}
	if replayed {
		switch existing.Status {
		case "completed", "failed":
			return approvalFromStoredInvocation(call.Name, existing)
		case "unknown":
			return nil, fmt.Errorf("%w: non-idempotent tool outcome is unknown", agentmanagement.ErrConflict)
		default:
			return nil, agentmanagement.ErrConflict
		}
	}
	result, invokeErr := registry.Invoke(ctx, lease.RegistryRevision, profile.ToolPolicy,
		invocationContext, call.Name, existing.Input)
	if invokeErr != nil {
		record.Status, record.ErrorCode = "failed", "tool_failed"
		event, finishErr := worker.store.FinishInvocation(ctx, record)
		if finishErr != nil {
			return nil, finishErr
		}
		worker.notifyEvent(ctx, lease, event)
		return nil, nil
	}
	record.Status, record.Result, record.ArtifactID = "completed", result.Value, result.ArtifactID
	event, executeToolErr := worker.store.FinishInvocation(ctx, record)
	if executeToolErr != nil {
		return nil, executeToolErr
	}
	worker.notifyEvent(ctx, lease, event)
	if call.Name != "router.publish.prepare" {
		return nil, nil
	}
	var output publishPrepareOutput
	if err := json.Unmarshal(result.Value, &output); err != nil {
		return nil, agentmanagement.ErrConflict
	}
	return &output.Approval, nil
}

func approvalFromStoredInvocation(
	toolName string, invocation agentmanagement.InvocationRecord,
) (*agentmanagement.ApprovalRequestEvent, error) {
	if invocation.Status != "completed" || toolName != "router.publish.prepare" {
		return nil, nil
	}
	var output publishPrepareOutput
	if err := json.Unmarshal(invocation.Result, &output); err != nil {
		return nil, agentmanagement.ErrConflict
	}
	return &output.Approval, nil
}

func (worker *Worker) ensureNotCancelled(ctx context.Context, lease agentmanagement.TurnLease) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	cancelled, err := worker.store.CancellationRequested(ctx, lease)
	if err != nil {
		return err
	}
	if cancelled {
		return agentmanagement.ErrCancelled
	}
	return nil
}

func (worker *Worker) commitExecutionCheckpoint(
	ctx context.Context, lease agentmanagement.TurnLease, projection executionContext,
) (executionContext, error) {
	checkpoint, err := encodeExecutionCheckpoint(lease, projection)
	if err != nil {
		return executionContext{}, err
	}
	_, event, err := worker.store.CommitCheckpoint(ctx, lease, checkpoint)
	if err == nil {
		projection.ThroughSequence = event.Sequence
		worker.notifyEvent(ctx, lease, event)
	}
	return projection, err
}

func deterministicInvocationID(turnID string, step, ordinal int) string {
	namespace := uuid.MustParse(turnID)
	return uuid.NewSHA1(namespace, []byte(fmt.Sprintf("agent-tool/%d/%d", step, ordinal))).String()
}

func deterministicModelStepID(turnID string, ordinal int64) string {
	namespace := uuid.MustParse(turnID)
	return uuid.NewSHA1(namespace, []byte(fmt.Sprintf("agent-model-step/%d", ordinal))).String()
}

func uuidForCheckpoint(turnID string, throughSequence int64) string {
	namespace := uuid.MustParse(turnID)
	return uuid.NewSHA1(namespace, []byte(fmt.Sprintf("agent-checkpoint/%d", throughSequence))).String()
}
