package handlers

import (
	"errors"
	"fmt"
	"net/http"
	"sort"
	"strings"

	"gopkg.in/yaml.v3"
)

// ErrDatabaseOwnedConfigMutation marks fields whose lifecycle belongs to the
// auth database rather than config.yaml. Generic config update, compiler,
// deploy, activation, and rollback bridges must fail before writing them.
var ErrDatabaseOwnedConfigMutation = errors.New("database-owned resource cannot be mutated through router config")

var databaseOwnedConfigKeys = map[string]string{
	"tenantgrants": "tenant_grants",
	"tenantquotas": "tenant_quotas",
	"quotapolicy":  "tenant_quotas",
	"budgetpolicy": "tenant_quotas",
	"virtualkeys":  "virtual_keys",
	"auditpolicy":  "audit_policy",
	"breakglass":   "breakglass",
}

func validateConfigMutationOwnership(data []byte) error {
	if len(strings.TrimSpace(string(data))) == 0 {
		return nil
	}

	var document yaml.Node
	if err := yaml.Unmarshal(data, &document); err != nil {
		return nil
	}

	found := map[string][]string{}
	collectDatabaseOwnedConfigPaths(&document, nil, found)
	if len(found) == 0 {
		return nil
	}

	owners := make([]string, 0, len(found))
	for owner := range found {
		owners = append(owners, owner)
	}
	sort.Strings(owners)
	return fmt.Errorf("%w: %s", ErrDatabaseOwnedConfigMutation, strings.Join(owners, ", "))
}

func rejectDatabaseOwnedConfigMutation(w http.ResponseWriter, data []byte) bool {
	err := validateConfigMutationOwnership(data)
	if err == nil {
		return false
	}
	if errors.Is(err, ErrDatabaseOwnedConfigMutation) {
		http.Error(w, err.Error(), http.StatusForbidden)
		return true
	}
	http.Error(w, fmt.Sprintf("Invalid configuration payload: %v", err), http.StatusBadRequest)
	return true
}

func collectDatabaseOwnedConfigPaths(node *yaml.Node, path []string, found map[string][]string) {
	if node == nil {
		return
	}
	if node.Kind == yaml.DocumentNode {
		for _, child := range node.Content {
			collectDatabaseOwnedConfigPaths(child, path, found)
		}
		return
	}
	if node.Kind == yaml.MappingNode {
		for index := 0; index+1 < len(node.Content); index += 2 {
			key := node.Content[index]
			value := node.Content[index+1]
			keyPath := appendPath(path, key.Value)
			if owner, blocked := databaseOwnedConfigKeys[normalizeOwnershipKey(key.Value)]; blocked {
				found[owner] = append(found[owner], strings.Join(keyPath, "."))
			}
			collectDatabaseOwnedConfigPaths(value, keyPath, found)
		}
		return
	}
	for _, child := range node.Content {
		collectDatabaseOwnedConfigPaths(child, path, found)
	}
}

func appendPath(path []string, value string) []string {
	next := make([]string, len(path), len(path)+1)
	copy(next, path)
	return append(next, value)
}

func normalizeOwnershipKey(value string) string {
	value = strings.ToLower(strings.TrimSpace(value))
	value = strings.ReplaceAll(value, "_", "")
	return strings.ReplaceAll(value, "-", "")
}
