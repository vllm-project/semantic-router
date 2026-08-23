// Package agentpublication coordinates the explicit human approval boundary
// with the existing Router publication application service.
package agentpublication

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/agentmanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/routingmanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementserver"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingsnapshot"
)

const publicationCommandTTL = 24 * time.Hour

var publicationETag = regexp.MustCompile(`^"agent:([1-9][0-9]*)"$`)

type Store interface {
	Ready(context.Context) error
	ReservePublicationCommit(
		context.Context, string, string, string, int64, agentmanagement.MutationContext,
	) (agentmanagement.PublicationCommitReservation, error)
	FinalizePublicationCommit(
		context.Context, string, string, string, int64, time.Time,
	) (agentmanagement.PublicationCommitResult, error)
	FailPublicationCommit(
		context.Context, string, string, string, time.Time,
	) (agentmanagement.PublicationCommitResult, error)
}

type Publisher interface {
	PublishEntrypoint(
		context.Context, string, string, int64, routingmanagement.MutationContext,
	) (*routingsnapshot.Snapshot, routingmanagement.RevisionReceipt, error)
}

type Options struct {
	Store     Store
	Publisher Publisher
	Commands  *managementcommand.Codec
	Notifier  agentmanagement.TurnNotifier
	Now       func() time.Time
}

// Committer is an idempotent two-transaction saga. The Agent transaction
// first fences cancellation by moving the immutable plan to publishing. The
// existing Routing transaction then publishes using the exact caller command.
// A crash after Routing commits is recovered by retrying the same command,
// which replays its durable operation receipt before the Agent transaction
// atomically writes approval_result, terminal, and the committed plan.
type Committer struct {
	store     Store
	publisher Publisher
	commands  *managementcommand.Codec
	notifier  agentmanagement.TurnNotifier
	now       func() time.Time
}

func New(options Options) (*Committer, error) {
	if options.Store == nil || options.Publisher == nil || options.Commands == nil {
		return nil, errors.New("agent publication dependencies are incomplete")
	}
	now := options.Now
	if now == nil {
		now = time.Now
	}
	return &Committer{
		store: options.Store, publisher: options.Publisher, commands: options.Commands,
		notifier: options.Notifier, now: now,
	}, nil
}

func (committer *Committer) Ready(ctx context.Context) error {
	if committer == nil || committer.store == nil || committer.publisher == nil || committer.commands == nil {
		return errors.New("agent publication is unavailable")
	}
	return committer.store.Ready(ctx)
}

func (committer *Committer) Commit(
	ctx context.Context, request managementserver.AgentPublicationCommitRequest,
) (managementserver.AgentPublicationCommitResult, error) {
	if committer == nil || request.PlanDigest == "" || request.IdempotencyKey == "" {
		return managementserver.AgentPublicationCommitResult{}, agentmanagement.ErrInvalid
	}
	expected, err := parseRevision(request.ExpectedETag)
	if err != nil {
		return managementserver.AgentPublicationCommitResult{}, err
	}
	reservation, err := committer.store.ReservePublicationCommit(
		ctx, request.NamespaceID, request.PlanID, request.PlanDigest, expected, request.Mutation,
	)
	if err != nil {
		return managementserver.AgentPublicationCommitResult{}, err
	}
	if reservation.OperationID != "" && reservation.DesiredRevision > 0 {
		return managementserver.AgentPublicationCommitResult{
			OperationID: reservation.OperationID,
			Revision:    reservation.DesiredRevision,
			Replayed:    true,
		}, nil
	}
	plan := reservation.Plan
	canonical, err := json.Marshal(struct {
		PlanID     string `json:"planId"`
		PlanDigest string `json:"planDigest"`
		PlanETag   string `json:"planEtag"`
	}{request.PlanID, request.PlanDigest, request.ExpectedETag})
	if err != nil {
		return managementserver.AgentPublicationCommitResult{}, agentmanagement.ErrInvalid
	}
	now := committer.now().UTC()
	endpoint := "/management/v1/publication-plans/" + request.PlanID + ":commit"
	command, err := committer.commands.Bind(
		managementcommand.NamespaceCommandScope(request.NamespaceID),
		request.Mutation.PrincipalID, endpoint, request.IdempotencyKey,
		canonical, now, now.Add(publicationCommandTTL),
	)
	if err != nil {
		if errors.Is(err, managementcommand.ErrConflict) {
			return managementserver.AgentPublicationCommitResult{}, agentmanagement.ErrConflict
		}
		return managementserver.AgentPublicationCommitResult{}, agentmanagement.ErrInvalid
	}
	_, receipt, err := committer.publisher.PublishEntrypoint(
		ctx, request.NamespaceID, plan.EntrypointID, plan.EntrypointResourceRevision,
		routingmanagement.MutationContext{
			PrincipalID: request.Mutation.PrincipalID,
			ActorChain:  append([]string(nil), request.Mutation.ActorChain...),
			RequestID:   request.Mutation.RequestID,
			Reason:      "approve Agent publication plan",
			Command:     &command,
		},
	)
	if err != nil {
		mapped := mapRoutingError(err)
		if deterministicPublicationFailure(err) {
			failed, failErr := committer.store.FailPublicationCommit(
				ctx, request.NamespaceID, request.PlanID, "publication_rejected", committer.now().UTC(),
			)
			if failErr != nil {
				return managementserver.AgentPublicationCommitResult{}, errors.Join(mapped, failErr)
			}
			committer.notify(ctx, request.NamespaceID, failed)
		}
		return managementserver.AgentPublicationCommitResult{}, mapped
	}
	if receipt.OperationID == "" || receipt.DesiredRevision < 1 {
		failed, failErr := committer.store.FailPublicationCommit(
			ctx, request.NamespaceID, request.PlanID, "publication_receipt_invalid", committer.now().UTC(),
		)
		if failErr == nil {
			committer.notify(ctx, request.NamespaceID, failed)
		} else {
			return managementserver.AgentPublicationCommitResult{}, errors.Join(agentmanagement.ErrConflict, failErr)
		}
		return managementserver.AgentPublicationCommitResult{}, agentmanagement.ErrConflict
	}
	finalized, err := committer.store.FinalizePublicationCommit(
		ctx, request.NamespaceID, request.PlanID, receipt.OperationID,
		receipt.DesiredRevision, committer.now().UTC(),
	)
	if err != nil {
		return managementserver.AgentPublicationCommitResult{}, err
	}
	committer.notify(ctx, request.NamespaceID, finalized)
	return managementserver.AgentPublicationCommitResult{
		OperationID: receipt.OperationID,
		Revision:    receipt.DesiredRevision,
		Replayed:    reservation.Replayed || receipt.Replayed || finalized.Replayed,
	}, nil
}

func (committer *Committer) notify(
	ctx context.Context, namespaceID string, result agentmanagement.PublicationCommitResult,
) {
	if committer.notifier == nil {
		return
	}
	for _, event := range []agentmanagement.Event{result.ApprovalEvent, result.TerminalEvent} {
		if event.Sequence > 0 {
			_ = committer.notifier.NotifyEvents(ctx, namespaceID, event.SessionID, event.Sequence)
		}
	}
}

func parseRevision(value string) (int64, error) {
	matches := publicationETag.FindStringSubmatch(strings.TrimSpace(value))
	if len(matches) != 2 {
		return 0, agentmanagement.ErrConflict
	}
	revision, err := strconv.ParseInt(matches[1], 10, 64)
	if err != nil || revision < 1 {
		return 0, agentmanagement.ErrConflict
	}
	return revision, nil
}

func mapRoutingError(err error) error {
	switch {
	case errors.Is(err, managementcommand.ErrConflict), errors.Is(err, routingmanagement.ErrConflict):
		return agentmanagement.ErrConflict
	case errors.Is(err, routingmanagement.ErrNotFound):
		return agentmanagement.ErrNotFound
	case errors.Is(err, routingmanagement.ErrInvalid), errors.Is(err, routingmanagement.ErrPublication):
		return agentmanagement.ErrApproval
	default:
		return fmt.Errorf("publish approved Agent plan: %w", err)
	}
}

func deterministicPublicationFailure(err error) bool {
	return errors.Is(err, managementcommand.ErrConflict) ||
		errors.Is(err, routingmanagement.ErrConflict) ||
		errors.Is(err, routingmanagement.ErrNotFound) ||
		errors.Is(err, routingmanagement.ErrInvalid) ||
		errors.Is(err, routingmanagement.ErrPublication)
}

var _ managementserver.AgentPublicationCommitter = (*Committer)(nil)
