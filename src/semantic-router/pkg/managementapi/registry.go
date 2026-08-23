package managementapi

import (
	"sort"
)

type HTTPMethod string

const (
	MethodGET    HTTPMethod = "GET"
	MethodPOST   HTTPMethod = "POST"
	MethodPUT    HTTPMethod = "PUT"
	MethodPATCH  HTTPMethod = "PATCH"
	MethodDELETE HTTPMethod = "DELETE"
)

type OperationScope string

const (
	ScopeAuthentication OperationScope = "authentication"
	ScopeIntrinsicSelf  OperationScope = "intrinsic_self"
	ScopeCluster        OperationScope = "cluster"
	ScopeNamespace      OperationScope = "namespace"
	ScopeResource       OperationScope = "resource"
	ScopeSubject        OperationScope = "subject"
	ScopeCompound       OperationScope = "compound"
	ScopeResultSet      OperationScope = "result_set"
	ScopeOperation      OperationScope = "operation_origin_and_targets"
)

type AsyncMode string

const (
	AsyncSynchronous AsyncMode = "synchronous"
	AsyncOperation   AsyncMode = "operation"
	AsyncConditional AsyncMode = "conditional_operation"
)

type (
	SecretInputMode  string
	SecretOutputMode string
)

const (
	SecretInputNone          SecretInputMode = "none"
	SecretInputAuthorization SecretInputMode = "authorization_header"
	SecretInputBody          SecretInputMode = "request_body"

	SecretOutputNone        SecretOutputMode = "none"
	SecretOutputOneTime     SecretOutputMode = "one_time_secret"
	SecretOutputAccessToken SecretOutputMode = "management_access_token"
)

type SecretMetadata struct {
	Input         SecretInputMode  `json:"input"`
	Output        SecretOutputMode `json:"output"`
	NoStore       bool             `json:"noStore"`
	Authenticated bool             `json:"varyAuthorization"`
}

type PaginationMode string

const (
	PaginationNone     PaginationMode = "none"
	PaginationKeyset   PaginationMode = "keyset"
	PaginationProvider PaginationMode = "provider_cursor"
)

type IdempotencyMode string

const (
	IdempotencyNone     IdempotencyMode = "none"
	IdempotencyRequired IdempotencyMode = "required"
)

type RevisionMode string

const (
	RevisionNone    RevisionMode = "none"
	RevisionReturns RevisionMode = "returns_etag"
	RevisionCAS     RevisionMode = "if_match_and_returns_etag"
)

// OperationContract is transport metadata only. It does not contain a handler,
// repository, or business resource type.
type OperationContract struct {
	Method      HTTPMethod           `json:"method"`
	Path        string               `json:"path"`
	OperationID string               `json:"operationId"`
	Tag         string               `json:"tag"`
	Permission  PermissionExpression `json:"permission"`
	Scope       OperationScope       `json:"scope"`
	Async       AsyncMode            `json:"async"`
	Secret      SecretMetadata       `json:"secret"`
	Pagination  PaginationMode       `json:"pagination"`
	Idempotency IdempotencyMode      `json:"idempotency"`
	Revision    RevisionMode         `json:"revision"`
}

type operationOption func(*OperationContract)

func paginated() operationOption {
	return func(operation *OperationContract) { operation.Pagination = PaginationKeyset }
}

func providerPaginated() operationOption {
	return func(operation *OperationContract) { operation.Pagination = PaginationProvider }
}

func casRevision() operationOption {
	return func(operation *OperationContract) { operation.Revision = RevisionCAS }
}

func asynchronous() operationOption {
	return func(operation *OperationContract) { operation.Async = AsyncOperation }
}

func secret(input SecretInputMode, output SecretOutputMode, authenticated bool) operationOption {
	return func(operation *OperationContract) {
		operation.Secret = SecretMetadata{
			Input:         input,
			Output:        output,
			NoStore:       input != SecretInputNone || output != SecretOutputNone,
			Authenticated: authenticated,
		}
	}
}

func sensitiveNoStore(authenticated bool) operationOption {
	return func(operation *OperationContract) {
		operation.Secret.NoStore = true
		operation.Secret.Authenticated = authenticated
	}
}

func noIdempotency() operationOption {
	return func(operation *OperationContract) { operation.Idempotency = IdempotencyNone }
}

func noRevision() operationOption {
	return func(operation *OperationContract) { operation.Revision = RevisionNone }
}

func operation(method HTTPMethod, path, tag string, scope OperationScope, permission PermissionExpression, options ...operationOption) OperationContract {
	contract := OperationContract{
		Method:      method,
		Path:        path,
		OperationID: deriveOperationID(method, path),
		Tag:         tag,
		Permission:  permission,
		Scope:       scope,
		Async:       AsyncSynchronous,
		Secret: SecretMetadata{
			Input:  SecretInputNone,
			Output: SecretOutputNone,
		},
		Pagination:  PaginationNone,
		Idempotency: IdempotencyNone,
		Revision:    RevisionNone,
	}
	if method == MethodPOST || method == MethodPUT {
		contract.Idempotency = IdempotencyRequired
		contract.Revision = RevisionReturns
	}
	if method == MethodPATCH {
		contract.Revision = RevisionCAS
	}
	if method == MethodDELETE {
		contract.Revision = RevisionReturns
	}
	for _, apply := range options {
		apply(&contract)
	}
	return contract
}

func resourceCRUD(tag, basePath, idParameter string, read, manage PermissionExpression, scope OperationScope) []OperationContract {
	detail := basePath + "/{" + idParameter + "}"
	return []OperationContract{
		operation(MethodGET, basePath, tag, scope, read, paginated()),
		operation(MethodPOST, basePath, tag, scope, manage),
		operation(MethodGET, detail, tag, scope, read),
		operation(MethodPATCH, detail, tag, scope, manage),
		operation(MethodDELETE, detail, tag, scope, manage),
	}
}

var registry = buildRegistry()

func Operations() []OperationContract {
	result := make([]OperationContract, len(registry))
	for i := range registry {
		result[i] = cloneOperationContract(registry[i])
	}
	sort.Slice(result, func(i, j int) bool {
		if result[i].Path == result[j].Path {
			return result[i].Method < result[j].Method
		}
		return result[i].Path < result[j].Path
	})
	return result
}

func LookupOperation(method HTTPMethod, path string) (OperationContract, bool) {
	for _, candidate := range registry {
		if candidate.Method == method && candidate.Path == path {
			return cloneOperationContract(candidate), true
		}
	}
	return OperationContract{}, false
}

func cloneOperationContract(source OperationContract) OperationContract {
	cloned := source
	cloned.Permission = clonePermissionExpression(source.Permission)
	return cloned
}

func clonePermissionExpression(source PermissionExpression) PermissionExpression {
	cloned := source
	if source.Operands != nil {
		cloned.Operands = make([]PermissionExpression, len(source.Operands))
		for i := range source.Operands {
			cloned.Operands[i] = clonePermissionExpression(source.Operands[i])
		}
	}
	return cloned
}

func buildRegistry() []OperationContract {
	var operations []OperationContract
	operations = append(operations, identityOperations()...)
	operations = append(operations, subjectOperations()...)
	operations = append(operations, accessOperations()...)
	operations = append(operations, routingOperations()...)
	operations = append(operations, agentOperations()...)
	operations = append(operations, observabilityOperations()...)
	if err := ValidateRegistry(operations); err != nil {
		panic(err)
	}
	return operations
}
