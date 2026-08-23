package quotareconciliation

import (
	"context"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/quotaruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/securitykeyring"
)

const (
	reconcileEndpoint = "/management/v1/unknown-usage-fences/{fenceId}:reconcile"
	defaultPageSize   = 50
	maximumPageSize   = 200
)

type Options struct {
	Repository        Repository
	Runtime           Runtime
	WaiveAuth         WaiveAuthenticator
	CommandCodec      *managementcommand.Codec
	CursorKeyring     securitykeyring.Symmetric
	IdempotencyTTL    time.Duration
	WorkerID          string
	WorkerConcurrency int
	PollInterval      time.Duration
	ClaimLease        time.Duration
	Now               func() time.Time
	NewID             func() string
}

type Service struct {
	repository        Repository
	runtime           Runtime
	waiveAuth         WaiveAuthenticator
	commands          *managementcommand.Codec
	cursors           *cursorCodec
	idempotencyTTL    time.Duration
	workerID          string
	workerConcurrency int
	pollInterval      time.Duration
	claimLease        time.Duration
	now               func() time.Time
	newID             func() string
}

func NewService(options Options) (*Service, error) {
	if options.Repository == nil || options.Runtime == nil || options.WaiveAuth == nil ||
		options.CommandCodec == nil || options.IdempotencyTTL < time.Minute || options.IdempotencyTTL > 7*24*time.Hour ||
		strings.TrimSpace(options.WorkerID) == "" || len(options.WorkerID) > 128 ||
		options.WorkerConcurrency < 1 || options.WorkerConcurrency > 32 ||
		options.PollInterval < 10*time.Millisecond || options.PollInterval > time.Minute ||
		options.ClaimLease < time.Second || options.ClaimLease > 10*time.Minute {
		return nil, ErrUnavailable
	}
	cursors, err := newCursorCodec(options.CursorKeyring)
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
		repository: options.Repository, runtime: options.Runtime, waiveAuth: options.WaiveAuth,
		commands: options.CommandCodec, cursors: cursors, idempotencyTTL: options.IdempotencyTTL,
		workerID: options.WorkerID, workerConcurrency: options.WorkerConcurrency,
		pollInterval: options.PollInterval, claimLease: options.ClaimLease, now: now, newID: newID,
	}, nil
}

func (service *Service) Close() {
	if service != nil && service.cursors != nil {
		service.cursors.Close()
	}
}

func (service *Service) Ready(ctx context.Context) error {
	if service == nil || service.repository == nil || service.commands == nil {
		return ErrUnavailable
	}
	if err := service.repository.ReadyQuotaReconciliation(ctx, service.commands); err != nil {
		return fmt.Errorf("%w: %w", ErrUnavailable, err)
	}
	return nil
}

func (service *Service) Get(ctx context.Context, namespaceID, fenceID string) (Fence, error) {
	if service == nil || !canonicalUUID(namespaceID) || !canonicalUUID(fenceID) {
		return Fence{}, ErrInvalidRequest
	}
	return service.repository.Get(ctx, namespaceID, fenceID)
}

// GetOperation reads one durable reconciliation operation in its namespace.
// Authorization remains the Management API's responsibility because it must
// re-evaluate both generic operation visibility and the original fence scope.
func (service *Service) GetOperation(ctx context.Context, namespaceID, operationID string) (Operation, error) {
	if service == nil || !canonicalUUID(namespaceID) || !canonicalUUID(operationID) {
		return Operation{}, ErrInvalidRequest
	}
	return service.repository.GetOperation(ctx, namespaceID, operationID)
}

func (service *Service) List(ctx context.Context, request ListRequest) (Page, error) {
	if service == nil || !canonicalUUID(request.NamespaceID) ||
		(request.State != "" && request.State != FenceOpen && request.State != FenceReconciling && request.State != FenceResolved) {
		return Page{}, ErrInvalidRequest
	}
	scope, listErr := request.Scope.Canonical()
	if listErr != nil || string(scope.NamespaceID) != request.NamespaceID {
		return Page{}, ErrInvalidRequest
	}
	pageSize := request.PageSize
	if pageSize == 0 {
		pageSize = defaultPageSize
	}
	if pageSize < 1 || pageSize > maximumPageSize {
		return Page{}, ErrInvalidRequest
	}
	scopeDigest, listErr := scope.Digest()
	if listErr != nil {
		return Page{}, ErrInvalidRequest
	}
	query := FenceQuery{NamespaceID: request.NamespaceID, State: request.State, Scope: scope, Limit: pageSize}
	if request.Cursor != "" {
		cursor, err := service.cursors.Decode(request.Cursor)
		if err != nil || cursor.NamespaceID != request.NamespaceID || cursor.State != request.State ||
			cursor.ScopeDigest != scopeDigest || !canonicalUUID(cursor.ID) || cursor.CreatedAt.IsZero() {
			return Page{}, ErrInvalidRequest
		}
		query.After = &Cursor{CreatedAt: cursor.CreatedAt, ID: cursor.ID}
	}
	if scope.Empty() {
		return Page{Items: []Fence{}, PageSize: pageSize}, nil
	}
	page, listErr := service.repository.List(ctx, query)
	if listErr != nil {
		return Page{}, listErr
	}
	result := Page{Items: page.Items, HasMore: page.HasMore, PageSize: pageSize}
	if page.HasMore {
		if len(page.Items) == 0 {
			return Page{}, ErrUnavailable
		}
		last := page.Items[len(page.Items)-1]
		result.NextCursor, listErr = service.cursors.Encode(cursorPayload{
			NamespaceID: request.NamespaceID,
			State:       request.State, ScopeDigest: scopeDigest, CreatedAt: last.CreatedAt, ID: last.ID,
		})
		if listErr != nil {
			return Page{}, ErrUnavailable
		}
	}
	return result, nil
}

func (service *Service) Reconcile(ctx context.Context, request ReconcileRequest) (EnqueueResult, error) {
	if service == nil || !validReconcileRequest(request) {
		return EnqueueResult{}, ErrInvalidRequest
	}
	now := service.now().UTC()
	if request.Strategy == StrategyWaive {
		if err := service.waiveAuth.AuthorizeWaive(ctx, request.NamespaceID, request.Session, now); err != nil {
			if errors.Is(err, ErrWaiveDenied) {
				return EnqueueResult{}, err
			}
			return EnqueueResult{}, ErrUnavailable
		}
	}
	canonical := struct {
		FenceID            string       `json:"fenceId"`
		ExpectedRevision   uint64       `json:"expectedRevision"`
		Strategy           Strategy     `json:"strategy"`
		Actual             *ActualUsage `json:"actual,omitempty"`
		EvidenceReferences []string     `json:"evidenceReferences,omitempty"`
		Reason             string       `json:"reason"`
	}{
		request.FenceID, request.ExpectedRevision, request.Strategy, request.Actual,
		append([]string(nil), request.EvidenceReferences...), request.Reason,
	}
	payload, err := json.Marshal(canonical)
	if err != nil {
		return EnqueueResult{}, ErrInvalidRequest
	}
	command, err := service.commands.Bind(managementcommand.NamespaceCommandScope(request.NamespaceID),
		request.Actor.PrincipalID, reconcileEndpoint, request.IdempotencyKey, payload, now, now.Add(service.idempotencyTTL))
	if err != nil {
		if errors.Is(err, managementcommand.ErrConflict) {
			return EnqueueResult{}, err
		}
		return EnqueueResult{}, ErrInvalidRequest
	}
	return service.repository.Prepare(ctx, command, request, service.newID(), now)
}

func validReconcileRequest(request ReconcileRequest) bool {
	if !canonicalUUID(request.NamespaceID) || !canonicalUUID(request.FenceID) || request.ExpectedRevision == 0 ||
		!request.Strategy.Valid() || !canonicalUUID(request.Actor.PrincipalID) ||
		len(request.Actor.ActorChain) == 0 || len(request.Actor.ActorChain) > 32 ||
		request.Actor.RequestID == "" || strings.TrimSpace(request.Reason) != request.Reason ||
		len(request.Reason) < 1 || len(request.Reason) > 512 {
		return false
	}
	seenActors := make(map[string]struct{}, len(request.Actor.ActorChain))
	for _, actor := range request.Actor.ActorChain {
		if !canonicalUUID(actor) {
			return false
		}
		if _, duplicate := seenActors[actor]; duplicate {
			return false
		}
		seenActors[actor] = struct{}{}
	}
	if request.Actor.SourceIP.IsValid() && request.Actor.SourceIP != request.Actor.SourceIP.Unmap() {
		return false
	}
	if len(request.EvidenceReferences) < 1 || len(request.EvidenceReferences) > 64 {
		return false
	}
	for _, reference := range request.EvidenceReferences {
		if len(reference) < 1 || len(reference) > 512 || strings.TrimSpace(reference) != reference || strings.ContainsAny(reference, "\x00\r\n") {
			return false
		}
	}
	switch request.Strategy {
	case StrategyActual:
		return request.Actual != nil && len(request.Actual.Dispatches) > 0 && len(request.Actual.Dispatches) <= 4096
	case StrategyConservativeDebit, StrategyWaive:
		return request.Actual == nil
	default:
		return false
	}
}

func (service *Service) Run(ctx context.Context) error {
	if service == nil {
		return ErrUnavailable
	}
	var workers sync.WaitGroup
	workers.Add(service.workerConcurrency)
	for worker := 0; worker < service.workerConcurrency; worker++ {
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
		processed, _ := service.ProcessOne(ctx)
		if ctx.Err() != nil {
			return
		}
		if processed {
			continue
		}
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
		}
	}
}

func (service *Service) ProcessOne(ctx context.Context) (bool, error) {
	now := service.now().UTC()
	claim, found, err := service.repository.Claim(ctx, service.workerID, now, service.claimLease)
	if err != nil || !found {
		return found, err
	}
	fail := func(cause error) (bool, error) {
		if releaseErr := service.repository.Release(ctx, claim, service.now().UTC(), cause); releaseErr != nil {
			return true, errors.Join(cause, releaseErr)
		}
		return true, cause
	}
	event, err := json.Marshal(struct {
		Schema           string               `json:"schema"`
		ReconciliationID string               `json:"reconciliationId"`
		FenceID          string               `json:"fenceId"`
		AdmissionID      string               `json:"admissionId"`
		Strategy         Strategy             `json:"strategy"`
		Dispatches       []CorrectionDispatch `json:"dispatches"`
	}{
		"usage.reconciliation.v1", claim.Plan.ReconciliationID, claim.Plan.FenceID,
		claim.Plan.AdmissionID, claim.Plan.Strategy, claim.Plan.Dispatches,
	})
	if err != nil {
		return fail(err)
	}
	if claim.Phase == PhaseRuntimePending {
		result, err := service.runtime.ApplyReconciliation(ctx, quotaruntime.ReconciliationRequest{
			Partition: claim.Plan.Partition, FenceID: claim.Plan.FenceID,
			AdmissionID: claim.Plan.AdmissionID, ReconciliationID: claim.Plan.ReconciliationID,
			PlanDigest: claim.PlanDigest, Event: string(event), Corrections: claim.Plan.Corrections,
		})
		if err != nil {
			return fail(err)
		}
		if err := service.repository.MarkRuntimeApplied(ctx, claim, result.StreamID, service.now().UTC()); err != nil {
			return fail(err)
		}
		claim.Phase, claim.RuntimeStreamID = PhaseRuntimeApplied, result.StreamID
	}
	if claim.Phase == PhaseRuntimeApplied {
		if err := service.repository.SettleLedger(ctx, claim, service.now().UTC()); err != nil {
			return fail(err)
		}
		claim.Phase = PhaseLedgerApplied
	}
	if claim.Phase == PhaseLedgerApplied {
		bindingIDs := make([]string, 0, len(claim.Plan.Corrections))
		for _, correction := range claim.Plan.Corrections {
			if correction.Enforcement == "enforce" {
				bindingIDs = append(bindingIDs, correction.BindingID)
			}
		}
		if _, err := service.runtime.RemoveReconciledFence(ctx, quotaruntime.FenceRemovalRequest{
			Partition: claim.Plan.Partition, FenceID: claim.Plan.FenceID,
			ReconciliationID: claim.Plan.ReconciliationID, PlanDigest: claim.PlanDigest,
			BindingIDs: bindingIDs,
		}); err != nil {
			return fail(err)
		}
		if _, err := service.repository.Complete(ctx, claim, service.now().UTC()); err != nil {
			return fail(err)
		}
	}
	return true, nil
}

type cursorPayload struct {
	NamespaceID string     `json:"namespaceId"`
	State       FenceState `json:"state,omitempty"`
	ScopeDigest string     `json:"scopeDigest"`
	CreatedAt   time.Time  `json:"createdAt"`
	ID          string     `json:"id"`
}

type cursorCodec struct{ key []byte }

func newCursorCodec(keyring securitykeyring.Symmetric) (*cursorCodec, error) {
	key, found := keyring.Keys[keyring.ActiveVersion]
	if !found || len(key) < sha256.Size {
		return nil, errors.New("active cursor key is unavailable")
	}
	mac := hmac.New(sha256.New, key)
	_, _ = mac.Write([]byte("vllm-sr/unknown-usage-fence-cursor/v1\x00" + keyring.ActiveVersion))
	return &cursorCodec{key: mac.Sum(nil)}, nil
}

func (codec *cursorCodec) Encode(payload cursorPayload) (string, error) {
	encoded, err := json.Marshal(payload)
	if err != nil {
		return "", err
	}
	mac := hmac.New(sha256.New, codec.key)
	_, _ = mac.Write(encoded)
	signature := mac.Sum(nil)
	return base64.RawURLEncoding.EncodeToString(encoded) + "." + base64.RawURLEncoding.EncodeToString(signature), nil
}

func (codec *cursorCodec) Decode(value string) (cursorPayload, error) {
	parts := strings.Split(value, ".")
	if len(parts) != 2 {
		return cursorPayload{}, ErrInvalidRequest
	}
	payload, err := base64.RawURLEncoding.DecodeString(parts[0])
	if err != nil || base64.RawURLEncoding.EncodeToString(payload) != parts[0] {
		return cursorPayload{}, ErrInvalidRequest
	}
	signature, err := base64.RawURLEncoding.DecodeString(parts[1])
	if err != nil || len(signature) != sha256.Size ||
		base64.RawURLEncoding.EncodeToString(signature) != parts[1] {
		return cursorPayload{}, ErrInvalidRequest
	}
	mac := hmac.New(sha256.New, codec.key)
	_, _ = mac.Write(payload)
	if !hmac.Equal(signature, mac.Sum(nil)) {
		return cursorPayload{}, ErrInvalidRequest
	}
	var decoded cursorPayload
	decoder := json.NewDecoder(strings.NewReader(string(payload)))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&decoded); err != nil {
		return cursorPayload{}, ErrInvalidRequest
	}
	return decoded, nil
}

func (codec *cursorCodec) Close() {
	if codec == nil {
		return
	}
	for index := range codec.key {
		codec.key[index] = 0
	}
	codec.key = nil
}

func canonicalUUID(value string) bool {
	parsed, err := uuid.Parse(value)
	return err == nil && parsed.String() == value
}

func DigestPlan(plan Plan) (string, []byte, error) {
	payload, err := json.Marshal(plan)
	if err != nil {
		return "", nil, err
	}
	digest := sha256.Sum256(payload)
	return hex.EncodeToString(digest[:]), payload, nil
}
