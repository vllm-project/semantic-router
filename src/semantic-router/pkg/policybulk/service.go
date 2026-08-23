package policybulk

import (
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/policymanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/securitykeyring"
)

const (
	accessBulkEndpoint       = "/management/v1/access-policy-bindings:bulk-apply"
	rateBulkEndpoint         = "/management/v1/rate-limit-bindings:bulk-apply"
	operationCancelEndpoint  = "/management/v1/operations/{operationId}:cancel"
	defaultOperationPageSize = 50
	maximumOperationPageSize = 200
)

type Options struct {
	Repository        Repository
	Policies          BindingService
	Authorization     ExecutionAuthorizer
	CommandCodec      *managementcommand.Codec
	CursorKeyring     securitykeyring.Symmetric
	IdempotencyTTL    time.Duration
	WorkerID          string
	WorkerConcurrency int
	PollInterval      time.Duration
	ClaimLease        time.Duration
	MaximumAttempts   int
	Now               func() time.Time
	NewID             func() string
}

type Service struct {
	repository        Repository
	policies          BindingService
	authorization     ExecutionAuthorizer
	commands          *managementcommand.Codec
	cursors           operationCursorCodec
	idempotencyTTL    time.Duration
	workerID          string
	workerConcurrency int
	pollInterval      time.Duration
	claimLease        time.Duration
	maximumAttempts   int
	now               func() time.Time
	newID             func() string
}

func NewService(options Options) (*Service, error) {
	if options.Repository == nil || options.Policies == nil || options.Authorization == nil ||
		options.CommandCodec == nil || options.IdempotencyTTL < time.Minute ||
		options.IdempotencyTTL > 7*24*time.Hour || strings.TrimSpace(options.WorkerID) == "" ||
		len(options.WorkerID) > 128 || options.WorkerConcurrency < 1 || options.WorkerConcurrency > 64 ||
		options.PollInterval < 10*time.Millisecond || options.PollInterval > time.Minute ||
		options.ClaimLease < time.Second || options.ClaimLease > 10*time.Minute ||
		options.MaximumAttempts < 1 || options.MaximumAttempts > 32 {
		return nil, ErrUnavailable
	}
	cursors, err := newOperationCursorCodec(options.CursorKeyring)
	if err != nil {
		return nil, fmt.Errorf("%w: %w", ErrUnavailable, err)
	}
	now := options.Now
	if now == nil {
		now = time.Now
	}
	newID := options.NewID
	if newID == nil {
		newID = uuid.NewString
	}
	return &Service{
		repository: options.Repository, policies: options.Policies,
		authorization: options.Authorization, commands: options.CommandCodec,
		cursors:        cursors,
		idempotencyTTL: options.IdempotencyTTL, workerID: options.WorkerID,
		workerConcurrency: options.WorkerConcurrency, pollInterval: options.PollInterval,
		claimLease: options.ClaimLease, maximumAttempts: options.MaximumAttempts,
		now: now, newID: newID,
	}, nil
}

func (service *Service) Close() {
	if service != nil {
		service.cursors.close()
	}
}

func (service *Service) Ready(ctx context.Context) error {
	if service == nil || service.repository == nil || service.commands == nil {
		return ErrUnavailable
	}
	if err := service.repository.Ready(ctx, service.commands); err != nil {
		return fmt.Errorf("%w: %w", ErrUnavailable, err)
	}
	return nil
}

func (service *Service) EnqueueAccessBindings(ctx context.Context, request EnqueueAccessRequest) (EnqueueResult, error) {
	if service == nil || !validActor(request.NamespaceID, request.Actor) || ValidateAccessItems(request.Items) != nil {
		return EnqueueResult{}, ErrInvalidRequest
	}
	canonical := struct {
		Items []AccessBindingItem `json:"items"`
	}{Items: request.Items}
	command, err := service.bindCommand(request.NamespaceID, request.Actor.PrincipalID,
		accessBulkEndpoint, request.IdempotencyKey, canonical)
	if err != nil {
		return EnqueueResult{}, err
	}
	operation, operationContext, err := service.newOperation(request.NamespaceID,
		AccessBindingOperationKind, request.Actor, accessTargets(request.Items))
	if err != nil {
		return EnqueueResult{}, err
	}
	return service.repository.EnqueueAccess(ctx, command, operation, operationContext, request.Items)
}

func (service *Service) EnqueueRateBindings(ctx context.Context, request EnqueueRateRequest) (EnqueueResult, error) {
	if service == nil || !validActor(request.NamespaceID, request.Actor) || ValidateRateItems(request.Items) != nil {
		return EnqueueResult{}, ErrInvalidRequest
	}
	canonical := struct {
		Items []RateBindingItem `json:"items"`
	}{Items: request.Items}
	command, err := service.bindCommand(request.NamespaceID, request.Actor.PrincipalID,
		rateBulkEndpoint, request.IdempotencyKey, canonical)
	if err != nil {
		return EnqueueResult{}, err
	}
	operation, operationContext, err := service.newOperation(request.NamespaceID,
		RateBindingOperationKind, request.Actor, rateTargets(request.Items))
	if err != nil {
		return EnqueueResult{}, err
	}
	return service.repository.EnqueueRate(ctx, command, operation, operationContext, request.Items)
}

func (service *Service) Get(ctx context.Context, namespaceID, operationID string) (Operation, error) {
	if service == nil || !canonicalUUID(namespaceID) || !canonicalUUID(operationID) {
		return Operation{}, ErrInvalidRequest
	}
	return service.repository.Get(ctx, namespaceID, operationID)
}

func (service *Service) List(ctx context.Context, request ListRequest) (Page, error) {
	if service == nil || !canonicalUUID(request.NamespaceID) ||
		(request.OriginPrincipalID != "" && !canonicalUUID(request.OriginPrincipalID)) ||
		(request.Kind != "" && !validOperationKind(request.Kind)) ||
		(request.State != "" && !request.State.Valid()) {
		return Page{}, ErrInvalidRequest
	}
	pageSize := request.PageSize
	if pageSize == 0 {
		pageSize = defaultOperationPageSize
	}
	if pageSize < 1 || pageSize > maximumOperationPageSize {
		return Page{}, ErrInvalidRequest
	}
	visibilityDigest, listErr := operationVisibilityDigest(request.NamespaceID, request.Visibility)
	if listErr != nil {
		return Page{}, ErrInvalidRequest
	}
	query := OperationQuery{
		NamespaceID:       request.NamespaceID,
		OriginPrincipalID: request.OriginPrincipalID, Kind: request.Kind, State: request.State,
		Limit: pageSize, Visibility: request.Visibility,
	}
	if request.Cursor != "" {
		cursor, err := service.cursors.decode(request.Cursor)
		if err != nil || cursor.NamespaceID != request.NamespaceID ||
			cursor.OriginPrincipalID != request.OriginPrincipalID || cursor.Kind != request.Kind ||
			cursor.State != request.State || cursor.VisibilityDigest != visibilityDigest ||
			!canonicalUUID(cursor.ID) || cursor.CreatedAt.IsZero() {
			return Page{}, ErrInvalidRequest
		}
		query.After = &Cursor{CreatedAt: cursor.CreatedAt, ID: cursor.ID}
	}
	accessEmpty := policyResultScopeEmpty(request.Visibility.Access, accesscontrol.ScopeResourceAccessPolicy)
	rateEmpty := policyResultScopeEmpty(request.Visibility.Rate, accesscontrol.ScopeResourceRateLimitPolicy)
	if (request.Kind == AccessBindingOperationKind && accessEmpty) ||
		(request.Kind == RateBindingOperationKind && rateEmpty) || (request.Kind == "" && accessEmpty && rateEmpty) {
		return Page{Items: []Operation{}, PageSize: pageSize}, nil
	}
	page, listErr := service.repository.List(ctx, query)
	if listErr != nil {
		return Page{}, listErr
	}
	result := Page{Items: page.Items, HasMore: page.HasMore, PageSize: pageSize}
	if !page.HasMore {
		return result, nil
	}
	if len(page.Items) == 0 {
		return Page{}, ErrUnavailable
	}
	last := page.Items[len(page.Items)-1]
	result.NextCursor, listErr = service.cursors.encode(operationCursorPayload{
		NamespaceID: request.NamespaceID, OriginPrincipalID: request.OriginPrincipalID,
		Kind: request.Kind, State: request.State, CreatedAt: last.CreatedAt, ID: last.ID,
		VisibilityDigest: visibilityDigest,
	})
	if listErr != nil || result.NextCursor == "" {
		return Page{}, ErrUnavailable
	}
	return result, nil
}

func operationVisibilityDigest(namespaceID string, visibility OperationVisibility) (string, error) {
	if !canonicalUUID(visibility.PrincipalID) {
		return "", ErrInvalidRequest
	}
	digests := make([]string, 0, 3)
	for _, scope := range []accesscontrol.ResultScope{visibility.Operation, visibility.Access, visibility.Rate} {
		if string(scope.NamespaceID) != namespaceID {
			return "", ErrInvalidRequest
		}
		digest, err := scope.Digest()
		if err != nil {
			return "", ErrInvalidRequest
		}
		digests = append(digests, digest)
	}
	hash := sha256.New()
	_, _ = hash.Write([]byte("vllm-sr/policy-bulk-visibility/v1\x00"))
	_, _ = hash.Write([]byte(visibility.PrincipalID))
	for _, digest := range digests {
		_, _ = hash.Write([]byte{0})
		_, _ = hash.Write([]byte(digest))
	}
	return base64.RawURLEncoding.EncodeToString(hash.Sum(nil)), nil
}

func policyResultScopeEmpty(scope accesscontrol.ResultScope, resourceType accesscontrol.ScopeResourceType) bool {
	return !scope.All && len(scope.IDs(resourceType)) == 0
}

func (service *Service) Cancel(ctx context.Context, request CancelRequest) (CancelResult, error) {
	if service == nil || !canonicalUUID(request.NamespaceID) || !canonicalUUID(request.OperationID) ||
		request.ExpectedVersion == 0 || !validActor(request.NamespaceID, request.Actor) {
		return CancelResult{}, ErrInvalidRequest
	}
	canonical := struct {
		OperationID     string `json:"operationId"`
		ExpectedVersion uint64 `json:"expectedVersion"`
	}{OperationID: request.OperationID, ExpectedVersion: request.ExpectedVersion}
	command, err := service.bindCommand(request.NamespaceID, request.Actor.PrincipalID,
		operationCancelEndpoint, request.IdempotencyKey, canonical)
	if err != nil {
		return CancelResult{}, err
	}
	return service.repository.Cancel(ctx, command, request)
}

// Run owns only worker goroutines.  Cancelling ctx stops polling; in-flight
// domain transactions finish or observe cancellation, and an unfinished lease
// is reclaimed by another replica after ClaimLease.
func (service *Service) Run(ctx context.Context) error {
	if service == nil {
		return ErrUnavailable
	}
	var workers sync.WaitGroup
	workers.Add(service.workerConcurrency)
	for index := 0; index < service.workerConcurrency; index++ {
		go func() {
			defer workers.Done()
			service.runWorker(ctx)
		}()
	}
	workers.Wait()
	return ctx.Err()
}

func (service *Service) runWorker(ctx context.Context) {
	ticker := time.NewTicker(service.pollInterval)
	defer ticker.Stop()
	for {
		processed, err := service.ProcessOne(ctx)
		if ctx.Err() != nil {
			return
		}
		if err == nil && processed {
			continue
		}
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
		}
	}
}

// ProcessOne is exported for deterministic worker integration tests and for
// runtimes that already own a cooperative work scheduler.
func (service *Service) ProcessOne(ctx context.Context) (bool, error) {
	if service == nil {
		return false, ErrUnavailable
	}
	now := service.timeNow()
	claim, found, processOneErr := service.repository.Claim(ctx, service.workerID, now, service.claimLease, service.maximumAttempts)
	if processOneErr != nil || !found {
		return found, processOneErr
	}
	request := AuthorizationRequest{
		NamespaceID: claim.NamespaceID, PrincipalID: claim.OriginPrincipalID,
		ActorChain: append([]string(nil), claim.ActorChain...), Kind: claim.ItemKind,
		ItemID: claimItemID(claim),
	}
	switch claim.ItemKind {
	case ItemKindAccessBinding:
		request.PolicyID, request.Subject = claim.Access.PolicyID, claim.Access.Subject
	case ItemKindRateBinding:
		request.PolicyID, request.Subject = claim.Rate.PolicyID, claim.Rate.Subject
		request.InlinePolicy = claim.Rate.InlinePolicy != nil
	default:
		return true, service.finishFailure(ctx, claim,
			ItemFailure{ItemID: claimItemID(claim), Code: "invalid_item", Reason: "Stored item is invalid."}, false, nil)
	}
	if err := service.authorization.AuthorizePolicyBulkItem(ctx, request); err != nil {
		if !errors.Is(err, ErrExecutionDenied) {
			return true, service.finishFailure(ctx, claim,
				ItemFailure{ItemID: claimItemID(claim), Code: "authorization_unavailable", Reason: "Authorization could not be checked."}, true, err)
		}
		return true, service.finishFailure(ctx, claim,
			ItemFailure{ItemID: claimItemID(claim), Code: "authorization_revoked", Reason: "Authorization is no longer valid."}, false, err)
	}
	result, processOneErr := service.executeClaim(ctx, claim)
	if processOneErr == nil {
		_, processOneErr = service.repository.Complete(ctx, claim, result, service.timeNow())
		return true, processOneErr
	}
	if ctx.Err() != nil {
		return true, ctx.Err()
	}
	retry, failure := classifyExecutionError(claimItemID(claim), processOneErr)
	return true, service.finishFailure(ctx, claim, failure, retry, processOneErr)
}

func (service *Service) executeClaim(ctx context.Context, claim ItemClaim) (ItemResult, error) {
	actor := policymanagement.Actor{
		PrincipalID: claim.OriginPrincipalID, ActorChain: append([]string(nil), claim.ActorChain...),
		RequestID: claim.Context.RequestID, SourceIP: claim.Context.SourceIP,
	}
	idempotencyKey := "policy-bulk/" + claim.OperationID + "/" + claimItemID(claim)
	switch claim.ItemKind {
	case ItemKindAccessBinding:
		result, err := service.policies.CreateAccessBinding(ctx, policymanagement.CreateAccessBindingRequest{
			NamespaceID: claim.NamespaceID, PolicyID: claim.Access.PolicyID,
			Subject: claim.Access.Subject, IdempotencyKey: idempotencyKey, Actor: actor,
		})
		return ItemResult{BindingID: result.ID}, err
	case ItemKindRateBinding:
		if claim.Rate.InlinePolicy == nil {
			result, err := service.policies.CreateRateBinding(ctx, policymanagement.CreateRateBindingRequest{
				NamespaceID: claim.NamespaceID, PolicyID: claim.Rate.PolicyID,
				Subject: claim.Rate.Subject, Mode: claim.Rate.Mode,
				IdempotencyKey: idempotencyKey, Actor: actor,
			})
			return ItemResult{BindingID: result.ID}, err
		}
		inline := claim.Rate.InlinePolicy
		result, err := service.policies.CreateInlineRateBinding(ctx, policymanagement.CreateInlineRateBindingRequest{
			NamespaceID: claim.NamespaceID, Name: inline.Name, Description: inline.Description,
			Rules: inline.Rules, Subject: claim.Rate.Subject, Mode: claim.Rate.Mode,
			IdempotencyKey: idempotencyKey, Actor: actor,
		})
		return ItemResult{BindingID: result.Binding.ID, PolicyID: result.Policy.ID}, err
	default:
		return ItemResult{}, ErrInvalidRequest
	}
}

func (service *Service) finishFailure(ctx context.Context, claim ItemClaim, failure ItemFailure, retry bool, cause error) error {
	now := service.timeNow()
	retryAt := now.Add(retryDelay(claim.Attempt))
	_, err := service.repository.Fail(ctx, claim, failure, retry, retryAt, now, service.maximumAttempts)
	if err != nil {
		return err
	}
	if retry {
		return cause
	}
	return nil
}

func (service *Service) bindCommand(namespaceID, principalID, endpoint, key string, body any) (managementcommand.Command, error) {
	canonical, err := json.Marshal(body)
	if err != nil {
		return managementcommand.Command{}, ErrInvalidRequest
	}
	now := service.timeNow()
	command, err := service.commands.Bind(managementcommand.NamespaceCommandScope(namespaceID),
		principalID, endpoint, key, canonical, now, now.Add(service.idempotencyTTL))
	if err != nil {
		return managementcommand.Command{}, ErrInvalidRequest
	}
	return command, nil
}

func (service *Service) newOperation(namespaceID, kind string, actor policymanagement.Actor, targets []OperationTarget) (Operation, OperationContext, error) {
	operationID := service.newID()
	if !canonicalUUID(operationID) {
		return Operation{}, OperationContext{}, ErrUnavailable
	}
	now := service.timeNow()
	return Operation{
			ID: operationID, NamespaceID: namespaceID, Kind: kind,
			OriginPrincipalID: actor.PrincipalID, ActorChain: append([]string(nil), actor.ActorChain...),
			Version: 1, State: OperationPending,
			Total: uint64(len(targets)), TargetIDs: operationTargetIDs(targets),
			Targets:   append([]OperationTarget(nil), targets...),
			CreatedAt: now, UpdatedAt: now,
		}, OperationContext{
			RequestID: actor.RequestID, SourceIP: actor.SourceIP,
			ExpiresAt: now.Add(service.idempotencyTTL),
		}, nil
}

func (service *Service) timeNow() time.Time { return service.now().UTC().Truncate(time.Microsecond) }

func accessTargets(items []AccessBindingItem) []OperationTarget {
	targets := make([]OperationTarget, len(items))
	for index := range items {
		targets[index] = OperationTarget{
			ItemID: items[index].ItemID,
			Kind:   ItemKindAccessBinding, PolicyID: items[index].PolicyID, Subject: items[index].Subject,
		}
	}
	return targets
}

func rateTargets(items []RateBindingItem) []OperationTarget {
	targets := make([]OperationTarget, len(items))
	for index := range items {
		targets[index] = OperationTarget{
			ItemID: items[index].ItemID,
			Kind:   ItemKindRateBinding, PolicyID: items[index].PolicyID,
			InlinePolicy: items[index].InlinePolicy != nil, Subject: items[index].Subject, Mode: items[index].Mode,
		}
	}
	return targets
}

func operationTargetIDs(targets []OperationTarget) []string {
	ids := make([]string, len(targets))
	for index := range targets {
		ids[index] = targets[index].ItemID
	}
	return ids
}

func claimItemID(claim ItemClaim) string {
	if claim.Access != nil {
		return claim.Access.ItemID
	}
	if claim.Rate != nil {
		return claim.Rate.ItemID
	}
	return ""
}

func retryDelay(attempt int) time.Duration {
	if attempt < 1 {
		attempt = 1
	}
	if attempt > 9 {
		attempt = 9
	}
	return time.Duration(1<<(attempt-1)) * 100 * time.Millisecond
}

func classifyExecutionError(itemID string, err error) (bool, ItemFailure) {
	failure := ItemFailure{ItemID: itemID, Code: "internal_error", Reason: "The item could not be applied."}
	switch {
	case errors.Is(err, policymanagement.ErrInvalidRequest):
		failure.Code, failure.Reason = "invalid_item", "The item is no longer valid."
	case errors.Is(err, policymanagement.ErrNotFound):
		failure.Code, failure.Reason = "not_found", "A referenced resource no longer exists."
	case errors.Is(err, policymanagement.ErrAlreadyExists):
		failure.Code, failure.Reason = "already_exists", "The binding already exists."
	case errors.Is(err, policymanagement.ErrAllocationConflict):
		failure.Code, failure.Reason = "allocation_conflict", "The subject already has an active allocation."
	case errors.Is(err, policymanagement.ErrCounterSemantics):
		failure.Code, failure.Reason = "counter_semantics", "The rate rule would change counter semantics."
	case errors.Is(err, policymanagement.ErrResourceInUse),
		errors.Is(err, policymanagement.ErrUnknownUsageFence),
		errors.Is(err, policymanagement.ErrRevisionConflict),
		errors.Is(err, managementcommand.ErrConflict):
		failure.Code, failure.Reason = "conflict", "The item conflicts with current state."
	case errors.Is(err, policymanagement.ErrUnavailable):
		return true, failure
	default:
		return true, failure
	}
	return false, failure
}
