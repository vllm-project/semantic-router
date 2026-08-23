package managementapi

import (
	"fmt"
	"regexp"
	"strings"
	"unicode"
)

func ValidateRegistry(operations []OperationContract) error {
	seenRoutes := make(map[string]bool, len(operations))
	seenIDs := make(map[string]bool, len(operations))
	for i := range operations {
		candidate := operations[i]
		if !knownMethods[candidate.Method] {
			return fmt.Errorf("operation %d has unsupported method %q", i, candidate.Method)
		}
		if !validManagementPath(candidate.Path) {
			return fmt.Errorf("operation %d has invalid path %q", i, candidate.Path)
		}
		if !operationIDPattern.MatchString(candidate.OperationID) {
			return fmt.Errorf("operation %s %s has invalid operation ID %q", candidate.Method, candidate.Path, candidate.OperationID)
		}
		routeKey := string(candidate.Method) + " " + candidate.Path
		if seenRoutes[routeKey] {
			return fmt.Errorf("duplicate operation route %s", routeKey)
		}
		seenRoutes[routeKey] = true
		if seenIDs[candidate.OperationID] {
			return fmt.Errorf("duplicate operation ID %q", candidate.OperationID)
		}
		seenIDs[candidate.OperationID] = true
		if candidate.Tag == "" {
			return fmt.Errorf("operation %s has no tag", routeKey)
		}
		if !knownOperationScopes[candidate.Scope] {
			return fmt.Errorf("operation %s has invalid scope %q", routeKey, candidate.Scope)
		}
		if err := candidate.Permission.Validate(); err != nil {
			return fmt.Errorf("operation %s permission: %w", routeKey, err)
		}
		if !knownAsyncModes[candidate.Async] || !knownPaginationModes[candidate.Pagination] ||
			!knownIdempotencyModes[candidate.Idempotency] || !knownRevisionModes[candidate.Revision] {
			return fmt.Errorf("operation %s has invalid protocol metadata", routeKey)
		}
		if !knownSecretInputs[candidate.Secret.Input] || !knownSecretOutputs[candidate.Secret.Output] {
			return fmt.Errorf("operation %s has invalid secret metadata", routeKey)
		}
		if (candidate.Secret.Input != SecretInputNone || candidate.Secret.Output != SecretOutputNone) && !candidate.Secret.NoStore {
			return fmt.Errorf("operation %s carries a secret without no-store", routeKey)
		}
		if candidate.Pagination != PaginationNone && candidate.Method != MethodGET && candidate.Pagination != PaginationProvider {
			return fmt.Errorf("operation %s uses keyset pagination on a non-GET route", routeKey)
		}
	}
	return nil
}

var (
	operationIDPattern   = regexp.MustCompile(`^[a-z][A-Za-z0-9]*$`)
	pathParameterPattern = regexp.MustCompile(`^\{[A-Za-z][A-Za-z0-9]*\}(?::[a-z][a-z0-9-]*)?$`)
)

func validManagementPath(path string) bool {
	if !strings.HasPrefix(path, BasePath+"/") || strings.ContainsAny(path, "?# ") {
		return false
	}
	for _, segment := range strings.Split(strings.TrimPrefix(path, BasePath+"/"), "/") {
		if segment == "" {
			return false
		}
		if strings.ContainsAny(segment, "{}") && !pathParameterPattern.MatchString(segment) {
			return false
		}
	}
	return true
}

func deriveOperationID(method HTTPMethod, path string) string {
	tokens := []string{strings.ToLower(string(method))}
	for _, segment := range strings.Split(strings.TrimPrefix(path, BasePath+"/"), "/") {
		parts := strings.SplitN(segment, ":", 2)
		base := parts[0]
		if strings.HasPrefix(base, "{") {
			tokens = append(tokens, "by", strings.TrimSuffix(strings.TrimPrefix(base, "{"), "}"))
		} else {
			tokens = append(tokens, strings.FieldsFunc(base, func(r rune) bool { return r == '-' || r == '_' })...)
		}
		if len(parts) == 2 {
			tokens = append(tokens, strings.FieldsFunc(parts[1], func(r rune) bool { return r == '-' || r == '_' })...)
		}
	}
	var result strings.Builder
	result.WriteString(tokens[0])
	for _, token := range tokens[1:] {
		if token == "" {
			continue
		}
		runes := []rune(token)
		runes[0] = unicode.ToUpper(runes[0])
		result.WriteString(string(runes))
	}
	return result.String()
}

var (
	knownMethods          = map[HTTPMethod]bool{MethodGET: true, MethodPOST: true, MethodPUT: true, MethodPATCH: true, MethodDELETE: true}
	knownOperationScopes  = map[OperationScope]bool{ScopeAuthentication: true, ScopeIntrinsicSelf: true, ScopeCluster: true, ScopeNamespace: true, ScopeResource: true, ScopeSubject: true, ScopeCompound: true, ScopeResultSet: true, ScopeOperation: true}
	knownAsyncModes       = map[AsyncMode]bool{AsyncSynchronous: true, AsyncOperation: true, AsyncConditional: true}
	knownPaginationModes  = map[PaginationMode]bool{PaginationNone: true, PaginationKeyset: true, PaginationProvider: true}
	knownIdempotencyModes = map[IdempotencyMode]bool{IdempotencyNone: true, IdempotencyRequired: true}
	knownRevisionModes    = map[RevisionMode]bool{RevisionNone: true, RevisionReturns: true, RevisionCAS: true}
	knownSecretInputs     = map[SecretInputMode]bool{SecretInputNone: true, SecretInputAuthorization: true, SecretInputBody: true}
	knownSecretOutputs    = map[SecretOutputMode]bool{SecretOutputNone: true, SecretOutputOneTime: true, SecretOutputAccessToken: true}
)
