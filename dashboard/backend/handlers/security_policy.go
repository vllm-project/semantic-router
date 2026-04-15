package handlers

import (
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"

	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

var (
	storedPolicy   *SecurityPolicyConfig
	storedPolicyMu sync.RWMutex
)

// SecurityPolicyConfig represents the dashboard-managed security policy
// that maps RBAC roles to router role_bindings and rate-limit rules.
type SecurityPolicyConfig struct {
	RoleMappings []RoleMapping   `json:"role_mappings"`
	RateTiers    []RateLimitTier `json:"rate_tiers"`
	UpdatedAt    string          `json:"updated_at"`
}

// RoleMapping maps a dashboard role or group to a router authz signal role
// and the set of models that role is allowed to access.
type RoleMapping struct {
	Name      string    `json:"name" yaml:"name"`
	Subjects  []Subject `json:"subjects" yaml:"subjects"`
	Role      string    `json:"role" yaml:"role"`
	ModelRefs []string  `json:"model_refs" yaml:"model_refs"`
	Priority  int       `json:"priority" yaml:"priority"`
}

// Subject identifies a user or group in a role binding.
type Subject struct {
	Kind string `json:"kind" yaml:"kind"` // "User" or "Group"
	Name string `json:"name" yaml:"name"`
}

// RateLimitTier maps a role/group to rate limit rules.
type RateLimitTier struct {
	Name           string `json:"name" yaml:"name"`
	Group          string `json:"group,omitempty" yaml:"group,omitempty"`
	User           string `json:"user,omitempty" yaml:"user,omitempty"`
	RequestsPerMin int    `json:"rpm" yaml:"rpm"`
	TokensPerMin   int    `json:"tpm,omitempty" yaml:"tpm,omitempty"`
}

// GeneratedRouterFragment contains the router config YAML fragments
// generated from the security policy.
type GeneratedRouterFragment struct {
	RoleBindings []RoleBindingFragment `json:"role_bindings" yaml:"role_bindings"`
	Decisions    []DecisionFragment    `json:"decisions" yaml:"decisions"`
	RateLimit    RateLimitFragment     `json:"ratelimit" yaml:"ratelimit"`
}

// RoleBindingFragment is a router-config-compatible role_binding entry.
type RoleBindingFragment struct {
	Subjects []Subject `json:"subjects" yaml:"subjects"`
	Role     string    `json:"role" yaml:"role"`
}

// DecisionFragment is a router-config-compatible decision entry.
type DecisionFragment struct {
	Name      string             `json:"name" yaml:"name"`
	Priority  int                `json:"priority" yaml:"priority"`
	Rules     RuleFragment       `json:"rules" yaml:"rules"`
	ModelRefs []ModelRefFragment `json:"modelRefs" yaml:"modelRefs"`
}

// RuleFragment is a minimal decision rule referencing an authz signal.
type RuleFragment struct {
	Type string `json:"type" yaml:"type"`
	Name string `json:"name" yaml:"name"`
}

// ModelRefFragment identifies a model in a decision.
type ModelRefFragment struct {
	Model string `json:"model" yaml:"model"`
}

// RateLimitFragment is the rate-limit config section.
type RateLimitFragment struct {
	Providers []RateLimitProviderFragment `json:"providers" yaml:"providers"`
}

// RateLimitProviderFragment is a rate-limit provider config entry.
type RateLimitProviderFragment struct {
	Type  string                  `json:"type" yaml:"type"`
	Rules []RateLimitRuleFragment `json:"rules" yaml:"rules"`
}

// RateLimitRuleFragment is a rate-limit rule config entry.
type RateLimitRuleFragment struct {
	Name  string                 `json:"name" yaml:"name"`
	Match RateLimitMatchFragment `json:"match" yaml:"match"`
	RPU   int                    `json:"requests_per_unit" yaml:"requests_per_unit"`
	TPU   int                    `json:"tokens_per_unit,omitempty" yaml:"tokens_per_unit,omitempty"`
	Unit  string                 `json:"unit" yaml:"unit"`
}

// RateLimitMatchFragment matches users/groups for rate limiting.
type RateLimitMatchFragment struct {
	User  string `json:"user,omitempty" yaml:"user,omitempty"`
	Group string `json:"group,omitempty" yaml:"group,omitempty"`
}

// GenerateRouterFragment converts a SecurityPolicyConfig into router config
// fragments that can be merged into the running router configuration.
func GenerateRouterFragment(policy *SecurityPolicyConfig) *GeneratedRouterFragment {
	fragment := &GeneratedRouterFragment{}

	for _, mapping := range policy.RoleMappings {
		fragment.RoleBindings = append(fragment.RoleBindings, RoleBindingFragment{
			Subjects: mapping.Subjects,
			Role:     mapping.Role,
		})

		modelRefs := make([]ModelRefFragment, 0, len(mapping.ModelRefs))
		for _, m := range mapping.ModelRefs {
			modelRefs = append(modelRefs, ModelRefFragment{Model: m})
		}

		fragment.Decisions = append(fragment.Decisions, DecisionFragment{
			Name:     fmt.Sprintf("rbac-%s", mapping.Name),
			Priority: mapping.Priority,
			Rules: RuleFragment{
				Type: "authz",
				Name: mapping.Role,
			},
			ModelRefs: modelRefs,
		})
	}

	if len(policy.RateTiers) > 0 {
		rules := make([]RateLimitRuleFragment, 0, len(policy.RateTiers))
		for _, tier := range policy.RateTiers {
			rules = append(rules, RateLimitRuleFragment{
				Name: tier.Name,
				Match: RateLimitMatchFragment{
					User:  tier.User,
					Group: tier.Group,
				},
				RPU:  tier.RequestsPerMin,
				TPU:  tier.TokensPerMin,
				Unit: "minute",
			})
		}
		fragment.RateLimit = RateLimitFragment{
			Providers: []RateLimitProviderFragment{
				{
					Type:  "local-limiter",
					Rules: rules,
				},
			},
		}
	}

	return fragment
}

var (
	securityPolicyConfigPath string
	securityPolicyConfigDir  string
)

// SetSecurityPolicyConfigPaths sets the router config file path and config
// directory used by HandleUpdateSecurityPolicy to apply fragments at save time.
func SetSecurityPolicyConfigPaths(configPath, configDir string) {
	securityPolicyConfigPath = configPath
	securityPolicyConfigDir = configDir
}

// toCanonicalYAML converts a GeneratedRouterFragment into canonical YAML bytes
// that can be consumed by mergeDeployPayload. It maps:
//   - fragment.RoleBindings -> routing.signals.role_bindings
//   - fragment.Decisions    -> routing.decisions
//   - fragment.RateLimit    -> global.services.ratelimit
func toCanonicalYAML(fragment *GeneratedRouterFragment) ([]byte, error) {
	doc := routingFragmentDocument{}

	for _, rb := range fragment.RoleBindings {
		subjects := make([]routerconfig.Subject, len(rb.Subjects))
		for i, s := range rb.Subjects {
			subjects[i] = routerconfig.Subject{Kind: s.Kind, Name: s.Name}
		}
		doc.Routing.Signals.RoleBindings = append(doc.Routing.Signals.RoleBindings,
			routerconfig.RoleBinding{Subjects: subjects, Role: rb.Role})
	}

	for _, d := range fragment.Decisions {
		modelRefs := make([]routerconfig.ModelRef, len(d.ModelRefs))
		for i, mr := range d.ModelRefs {
			modelRefs[i] = routerconfig.ModelRef{Model: mr.Model}
		}
		doc.Routing.Decisions = append(doc.Routing.Decisions, routerconfig.Decision{
			Name:      d.Name,
			Priority:  d.Priority,
			Rules:     routerconfig.RuleNode{Type: d.Rules.Type, Name: d.Rules.Name},
			ModelRefs: modelRefs,
		})
	}

	if len(fragment.RateLimit.Providers) > 0 {
		rlCfg := &routerconfig.RateLimitConfig{}
		for _, p := range fragment.RateLimit.Providers {
			provider := routerconfig.RateLimitProviderConfig{Type: p.Type}
			for _, r := range p.Rules {
				provider.Rules = append(provider.Rules, routerconfig.RateLimitRule{
					Name:            r.Name,
					Match:           routerconfig.RateLimitMatch{User: r.Match.User, Group: r.Match.Group},
					RequestsPerUnit: r.RPU,
					TokensPerUnit:   r.TPU,
					Unit:            r.Unit,
				})
			}
			rlCfg.Providers = append(rlCfg.Providers, provider)
		}
		doc.Global = &globalFragment{
			Services: &globalServicesFragment{RateLimit: rlCfg},
		}
	}

	return marshalYAMLBytes(doc)
}

// HandleGetSecurityPolicy returns the current security policy configuration.
func HandleGetSecurityPolicy(w http.ResponseWriter, r *http.Request) {
	policy := loadSecurityPolicy()
	writeJSON(w, http.StatusOK, policy)
}

// HandleUpdateSecurityPolicy updates the security policy, regenerates the
// router config fragment, and applies it to the running router config.
func HandleUpdateSecurityPolicy(w http.ResponseWriter, r *http.Request) {
	var policy SecurityPolicyConfig
	if err := json.NewDecoder(r.Body).Decode(&policy); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{
			"error": "invalid request body: " + err.Error(),
		})
		return
	}

	if err := validateSecurityPolicy(&policy); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{
			"error": err.Error(),
		})
		return
	}

	policy.UpdatedAt = time.Now().UTC().Format(time.RFC3339)
	fragment := GenerateRouterFragment(&policy)
	saveSecurityPolicy(&policy)

	applied := applySecurityFragment(fragment)

	msg := "Security policy updated and applied to router config."
	if !applied {
		msg = "Security policy updated. Router config path not configured; fragment was not auto-applied."
	}

	writeJSON(w, http.StatusOK, map[string]interface{}{
		"policy":   policy,
		"fragment": fragment,
		"applied":  applied,
		"message":  msg,
	})
}

// applySecurityFragment merges the generated fragment into config.yaml and
// triggers a runtime hot-reload. Returns true if the apply succeeded.
func applySecurityFragment(fragment *GeneratedRouterFragment) bool {
	if securityPolicyConfigPath == "" {
		return false
	}

	deployMu.Lock()
	defer deployMu.Unlock()

	yamlBytes, err := toCanonicalYAML(fragment)
	if err != nil {
		log.Printf("[SecurityPolicy] failed to marshal canonical YAML: %v", err)
		return false
	}

	existingData, err := os.ReadFile(securityPolicyConfigPath)
	if err != nil {
		if !errors.Is(err, os.ErrNotExist) {
			log.Printf("[SecurityPolicy] failed to read config file %s: %v", securityPolicyConfigPath, err)
			return false
		}
		existingData = nil
	}

	merged, err := mergeDeployPayload(existingData, DeployRequest{YAML: string(yamlBytes)})
	if err != nil {
		log.Printf("[SecurityPolicy] failed to merge fragment into config: %v", err)
		return false
	}

	// TODO: validate merged YAML parses into a valid router config before writing (comment 10).
	if err := writeConfigAtomically(securityPolicyConfigPath, merged); err != nil {
		log.Printf("[SecurityPolicy] failed to write config: %v", err)
		return false
	}

	log.Printf("[SecurityPolicy] Config written to %s (%d bytes)", securityPolicyConfigPath, len(merged))

	if err := applyWrittenConfig(securityPolicyConfigPath, securityPolicyConfigDir, existingData, true); err != nil {
		log.Printf("[SecurityPolicy] failed to apply config to runtime: %v", err)
		return false
	}

	return true
}

// HandlePreviewSecurityFragment generates a router config fragment from the
// given security policy without applying it.
func HandlePreviewSecurityFragment(w http.ResponseWriter, r *http.Request) {
	var policy SecurityPolicyConfig
	if err := json.NewDecoder(r.Body).Decode(&policy); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{
			"error": "invalid request body: " + err.Error(),
		})
		return
	}

	if err := validateSecurityPolicy(&policy); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{
			"error": err.Error(),
		})
		return
	}

	fragment := GenerateRouterFragment(&policy)
	writeJSON(w, http.StatusOK, fragment)
}

func validateSecurityPolicy(policy *SecurityPolicyConfig) error {
	seen := make(map[string]bool)
	for i, m := range policy.RoleMappings {
		if m.Name == "" {
			return fmt.Errorf("role_mappings[%d]: name is required", i)
		}
		if seen[m.Name] {
			return fmt.Errorf("role_mappings[%d]: duplicate name %q", i, m.Name)
		}
		seen[m.Name] = true
		if m.Role == "" {
			return fmt.Errorf("role_mappings[%d]: role is required", i)
		}
		if len(m.Subjects) == 0 {
			return fmt.Errorf("role_mappings[%d]: at least one subject is required", i)
		}
		for j, s := range m.Subjects {
			kind := strings.ToLower(s.Kind)
			if kind != "user" && kind != "group" {
				return fmt.Errorf("role_mappings[%d].subjects[%d]: kind must be User or Group", i, j)
			}
			if s.Name == "" {
				return fmt.Errorf("role_mappings[%d].subjects[%d]: name is required", i, j)
			}
		}
	}
	for i, t := range policy.RateTiers {
		if t.Name == "" {
			return fmt.Errorf("rate_tiers[%d]: name is required", i)
		}
		if t.RequestsPerMin == 0 && t.TokensPerMin == 0 {
			return fmt.Errorf("rate_tiers[%d]: at least one of rpm or tpm must be set", i)
		}
	}
	return nil
}

func loadSecurityPolicy() *SecurityPolicyConfig {
	storedPolicyMu.RLock()
	defer storedPolicyMu.RUnlock()
	if storedPolicy != nil {
		return storedPolicy
	}
	return &SecurityPolicyConfig{
		RoleMappings: []RoleMapping{},
		RateTiers:    []RateLimitTier{},
	}
}

func saveSecurityPolicy(policy *SecurityPolicyConfig) {
	storedPolicyMu.Lock()
	defer storedPolicyMu.Unlock()
	storedPolicy = policy
}

func writeJSON(w http.ResponseWriter, status int, data interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(data)
}
