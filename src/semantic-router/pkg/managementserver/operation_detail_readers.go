package managementserver

import (
	"context"
	"errors"
	"fmt"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementapi"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauth"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauthorization"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/policybulk"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/quotareconciliation"
)

var (
	errOperationDetailNotFound    = errors.New("management operation detail not found")
	errOperationDetailUnavailable = errors.New("management operation detail unavailable")
)

// OperationDetailReadRequest carries the already authenticated namespace and
// session to a registered operation-domain reader. A reader must recheck the
// original domain authority before returning a result.
type OperationDetailReadRequest struct {
	NamespaceID string
	OperationID string
	Session     managementauth.AuthenticatedSession
}

type OperationDetailResult struct {
	Operation managementapi.Operation
	Version   uint64
}

// OperationDetailReader is the narrow registration seam for one durable
// operation kind. There is intentionally no generic task submission seam.
type OperationDetailReader interface {
	OperationKinds() []string
	ReadOperation(context.Context, OperationDetailReadRequest) (OperationDetailResult, error)
}

type operationDetailRegistry struct {
	readers []OperationDetailReader
}

func newOperationDetailRegistry(readers []OperationDetailReader) (*operationDetailRegistry, error) {
	if len(readers) == 0 {
		return nil, errors.New("management Operation detail requires at least one registered reader")
	}
	seen := make(map[string]struct{}, len(readers))
	registered := make([]OperationDetailReader, 0, len(readers))
	for _, reader := range readers {
		if reader == nil {
			return nil, errors.New("management Operation detail reader is required")
		}
		kinds := reader.OperationKinds()
		if len(kinds) == 0 {
			return nil, errors.New("management Operation detail reader kind is required")
		}
		for _, kind := range kinds {
			canonical := strings.TrimSpace(kind)
			if canonical == "" || canonical != kind {
				return nil, errors.New("management Operation detail reader kind is invalid")
			}
			if _, duplicate := seen[kind]; duplicate {
				return nil, fmt.Errorf("management Operation detail reader kind %q is registered twice", kind)
			}
			seen[kind] = struct{}{}
		}
		registered = append(registered, reader)
	}
	return &operationDetailRegistry{readers: registered}, nil
}

func (registry *operationDetailRegistry) Read(
	ctx context.Context,
	request OperationDetailReadRequest,
) (OperationDetailResult, error) {
	if registry == nil || !canonicalUUID(request.NamespaceID) || !canonicalUUID(request.OperationID) ||
		request.Session.NamespaceID != request.NamespaceID {
		return OperationDetailResult{}, errOperationDetailUnavailable
	}
	for _, reader := range registry.readers {
		result, err := reader.ReadOperation(ctx, request)
		if errors.Is(err, errOperationDetailNotFound) {
			continue
		}
		if err != nil {
			return OperationDetailResult{}, err
		}
		if result.Version == 0 || result.Operation.OperationID != request.OperationID ||
			!readerOwnsOperationKind(reader, result.Operation.Kind) {
			return OperationDetailResult{}, errOperationDetailUnavailable
		}
		return result, nil
	}
	return OperationDetailResult{}, errOperationDetailNotFound
}

func readerOwnsOperationKind(reader OperationDetailReader, kind string) bool {
	for _, registered := range reader.OperationKinds() {
		if registered == kind {
			return true
		}
	}
	return false
}

type policyBulkOperationDetailReader struct {
	service       OperationService
	authorization Authorizer
	contracts     operationRouteContracts
}

func (*policyBulkOperationDetailReader) OperationKinds() []string {
	return []string{policybulk.AccessBindingOperationKind, policybulk.RateBindingOperationKind}
}

func (reader *policyBulkOperationDetailReader) ReadOperation(
	ctx context.Context,
	request OperationDetailReadRequest,
) (OperationDetailResult, error) {
	operation, err := reader.service.Get(ctx, request.NamespaceID, request.OperationID)
	if errors.Is(err, policybulk.ErrNotFound) {
		return OperationDetailResult{}, errOperationDetailNotFound
	}
	if err != nil {
		return OperationDetailResult{}, errOperationDetailUnavailable
	}
	if err := authorizePolicyBulkStoredOperation(ctx, reader.authorization, reader.contracts,
		request.Session, operation, operationAuthorizationRead); err != nil {
		if operationDenied(err) {
			return OperationDetailResult{}, errOperationDetailNotFound
		}
		return OperationDetailResult{}, errOperationDetailUnavailable
	}
	return OperationDetailResult{Operation: newPolicyBulkOperation(operation), Version: operation.Version}, nil
}

type unknownUsageOperationDetailReader struct {
	service       UnknownUsageService
	authorization Authorizer
	fenceRead     managementapi.OperationContract
	operationRead managementapi.OperationContract
}

// NewUnknownUsageOperationDetailReader registers reconciliation Operations
// with the generic Operation detail resource without coupling the generic
// route to the quota domain.
func NewUnknownUsageOperationDetailReader(
	service UnknownUsageService,
	authorization Authorizer,
) (OperationDetailReader, error) {
	if service == nil || authorization == nil {
		return nil, errors.New("unknown-usage Operation reader requires service and authorization")
	}
	fenceRead, found := managementapi.LookupOperation(managementapi.MethodGET,
		unknownUsageFencesPath+"/{fenceId}")
	if !found {
		return nil, errors.New("unknown-usage detail authorization contract is unavailable")
	}
	operationRead, found := managementapi.LookupOperation(managementapi.MethodGET,
		operationsPath+"/{operationId}")
	if !found {
		return nil, errors.New("operation detail authorization contract is unavailable")
	}
	return &unknownUsageOperationDetailReader{
		service: service, authorization: authorization,
		fenceRead: fenceRead, operationRead: operationRead,
	}, nil
}

func (*unknownUsageOperationDetailReader) OperationKinds() []string {
	return []string{quotareconciliation.OperationKind}
}

func (reader *unknownUsageOperationDetailReader) ReadOperation(
	ctx context.Context,
	request OperationDetailReadRequest,
) (OperationDetailResult, error) {
	operation, err := reader.service.GetOperation(ctx, request.NamespaceID, request.OperationID)
	if errors.Is(err, quotareconciliation.ErrNotFound) {
		return OperationDetailResult{}, errOperationDetailNotFound
	}
	if err != nil || !validUnknownUsageOperation(operation, request.NamespaceID) {
		return OperationDetailResult{}, errOperationDetailUnavailable
	}
	fence, err := reader.service.Get(ctx, request.NamespaceID, operation.FenceID)
	if err != nil || fence.ID != operation.FenceID || fence.NamespaceID != operation.NamespaceID {
		return OperationDetailResult{}, errOperationDetailUnavailable
	}
	targets := unknownUsageTargets(fence)
	if _, err := reader.authorization.Authorize(ctx, AuthorizationRequest{
		Operation: reader.fenceRead, Session: request.Session, NamespaceID: request.NamespaceID,
		Targets: targets,
	}); err != nil {
		if errors.Is(err, managementauthorization.ErrDenied) {
			return OperationDetailResult{}, errOperationDetailNotFound
		}
		return OperationDetailResult{}, errOperationDetailUnavailable
	}
	operationTargets := append([]accesscontrol.ScopedTarget(nil), targets["all_affected_bindings"]...)
	originator := operation.OriginPrincipalID == request.Session.Session.PrincipalID
	if _, err := reader.authorization.Authorize(ctx, AuthorizationRequest{
		Operation: reader.operationRead, Session: request.Session, NamespaceID: request.NamespaceID,
		Targets: map[string][]accesscontrol.ScopedTarget{"operation_targets": operationTargets},
		Conditions: map[string]bool{
			"operation_originator":  originator,
			"cross_actor_operation": !originator,
		},
		Recorded: map[string]bool{"original_domain_read": true},
	}); err != nil {
		if errors.Is(err, managementauthorization.ErrDenied) {
			return OperationDetailResult{}, errOperationDetailNotFound
		}
		return OperationDetailResult{}, errOperationDetailUnavailable
	}
	return OperationDetailResult{Operation: unknownUsageOperation(operation), Version: operation.Version}, nil
}

func validUnknownUsageOperation(operation quotareconciliation.Operation, namespaceID string) bool {
	if !canonicalUUID(operation.ID) || operation.NamespaceID != namespaceID || !canonicalUUID(operation.NamespaceID) ||
		!canonicalUUID(operation.FenceID) || operation.Kind != quotareconciliation.OperationKind ||
		!canonicalUUID(operation.OriginPrincipalID) || operation.Version == 0 || operation.Total != 1 ||
		operation.Completed > operation.Total || operation.CreatedAt.IsZero() || operation.UpdatedAt.IsZero() ||
		len(operation.ActorChain) == 0 || len(operation.ActorChain) > 32 {
		return false
	}
	for _, actor := range operation.ActorChain {
		if !canonicalUUID(actor) {
			return false
		}
	}
	switch operation.State {
	case quotareconciliation.OperationPending, quotareconciliation.OperationRunning:
		return operation.CompletedAt == nil
	case quotareconciliation.OperationSucceeded:
		return operation.Completed == operation.Total && operation.CompletedAt != nil
	default:
		return false
	}
}
