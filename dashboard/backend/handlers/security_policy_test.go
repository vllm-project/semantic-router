package handlers

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestGenerateRouterFragmentRoleMappings(t *testing.T) {
	t.Parallel()

	policy := &SecurityPolicyConfig{
		RoleMappings: []RoleMapping{
			{
				Name:      "premium",
				Subjects:  []Subject{{Kind: "Group", Name: "paying-users"}},
				Role:      "premium_tier",
				ModelRefs: []string{"gpt-4", "claude-3"},
				Priority:  10,
			},
			{
				Name:      "internal",
				Subjects:  []Subject{{Kind: "User", Name: "admin@co.com"}, {Kind: "Group", Name: "eng"}},
				Role:      "internal_tier",
				ModelRefs: []string{"gpt-4o"},
				Priority:  20,
			},
		},
	}

	fragment := GenerateRouterFragment(policy)

	if len(fragment.RoleBindings) != 2 {
		t.Fatalf("expected 2 role bindings, got %d", len(fragment.RoleBindings))
	}
	if fragment.RoleBindings[0].Role != "premium_tier" {
		t.Fatalf("expected role premium_tier, got %q", fragment.RoleBindings[0].Role)
	}
	if len(fragment.RoleBindings[1].Subjects) != 2 {
		t.Fatalf("expected 2 subjects for internal binding, got %d", len(fragment.RoleBindings[1].Subjects))
	}

	if len(fragment.Decisions) != 2 {
		t.Fatalf("expected 2 decisions, got %d", len(fragment.Decisions))
	}
	if fragment.Decisions[0].Name != "rbac-premium" {
		t.Fatalf("expected decision name rbac-premium, got %q", fragment.Decisions[0].Name)
	}
	if fragment.Decisions[0].Priority != 10 {
		t.Fatalf("expected priority 10, got %d", fragment.Decisions[0].Priority)
	}
	if fragment.Decisions[0].Rules.Type != "authz" {
		t.Fatalf("expected rule type authz, got %q", fragment.Decisions[0].Rules.Type)
	}
	if len(fragment.Decisions[0].ModelRefs) != 2 {
		t.Fatalf("expected 2 model refs for premium, got %d", len(fragment.Decisions[0].ModelRefs))
	}
	if fragment.Decisions[0].ModelRefs[0].Model != "gpt-4" {
		t.Fatalf("expected first model gpt-4, got %q", fragment.Decisions[0].ModelRefs[0].Model)
	}
}

func TestGenerateRouterFragmentRateTiers(t *testing.T) {
	t.Parallel()

	policy := &SecurityPolicyConfig{
		RateTiers: []RateLimitTier{
			{Name: "free-tier", Group: "free-users", RequestsPerMin: 10, TokensPerMin: 5000},
			{Name: "admin-unlimited", User: "admin@co.com", RequestsPerMin: 1000},
		},
	}

	fragment := GenerateRouterFragment(policy)

	if len(fragment.RateLimit.Providers) != 1 {
		t.Fatalf("expected 1 rate limit provider, got %d", len(fragment.RateLimit.Providers))
	}
	provider := fragment.RateLimit.Providers[0]
	if provider.Type != "local-limiter" {
		t.Fatalf("expected provider type local-limiter, got %q", provider.Type)
	}
	if len(provider.Rules) != 2 {
		t.Fatalf("expected 2 rate rules, got %d", len(provider.Rules))
	}
	if provider.Rules[0].Name != "free-tier" {
		t.Fatalf("expected rule name free-tier, got %q", provider.Rules[0].Name)
	}
	if provider.Rules[0].Match.Group != "free-users" {
		t.Fatalf("expected match group free-users, got %q", provider.Rules[0].Match.Group)
	}
	if provider.Rules[0].RPU != 10 {
		t.Fatalf("expected RPU 10, got %d", provider.Rules[0].RPU)
	}
	if provider.Rules[0].TPU != 5000 {
		t.Fatalf("expected TPU 5000, got %d", provider.Rules[0].TPU)
	}
	if provider.Rules[0].Unit != "minute" {
		t.Fatalf("expected unit minute, got %q", provider.Rules[0].Unit)
	}

	if provider.Rules[1].Match.User != "admin@co.com" {
		t.Fatalf("expected match user admin@co.com, got %q", provider.Rules[1].Match.User)
	}
}

func TestGenerateRouterFragmentEmpty(t *testing.T) {
	t.Parallel()

	policy := &SecurityPolicyConfig{}
	fragment := GenerateRouterFragment(policy)

	if len(fragment.RoleBindings) != 0 {
		t.Fatalf("expected 0 role bindings, got %d", len(fragment.RoleBindings))
	}
	if len(fragment.Decisions) != 0 {
		t.Fatalf("expected 0 decisions, got %d", len(fragment.Decisions))
	}
	if len(fragment.RateLimit.Providers) != 0 {
		t.Fatalf("expected 0 rate limit providers, got %d", len(fragment.RateLimit.Providers))
	}
}

func TestValidateSecurityPolicyRejectsEmptyName(t *testing.T) {
	t.Parallel()

	policy := &SecurityPolicyConfig{
		RoleMappings: []RoleMapping{
			{Name: "", Role: "tier", Subjects: []Subject{{Kind: "Group", Name: "g"}}},
		},
	}
	if err := validateSecurityPolicy(policy); err == nil {
		t.Fatal("expected validation error for empty name")
	}
}

func TestValidateSecurityPolicyRejectsDuplicateNames(t *testing.T) {
	t.Parallel()

	policy := &SecurityPolicyConfig{
		RoleMappings: []RoleMapping{
			{Name: "dup", Role: "a", Subjects: []Subject{{Kind: "Group", Name: "g1"}}},
			{Name: "dup", Role: "b", Subjects: []Subject{{Kind: "User", Name: "u1"}}},
		},
	}
	if err := validateSecurityPolicy(policy); err == nil {
		t.Fatal("expected validation error for duplicate name")
	}
}

func TestValidateSecurityPolicyRejectsEmptyRole(t *testing.T) {
	t.Parallel()

	policy := &SecurityPolicyConfig{
		RoleMappings: []RoleMapping{
			{Name: "ok", Role: "", Subjects: []Subject{{Kind: "Group", Name: "g"}}},
		},
	}
	if err := validateSecurityPolicy(policy); err == nil {
		t.Fatal("expected validation error for empty role")
	}
}

func TestValidateSecurityPolicyRejectsEmptySubjects(t *testing.T) {
	t.Parallel()

	policy := &SecurityPolicyConfig{
		RoleMappings: []RoleMapping{
			{Name: "ok", Role: "tier", Subjects: []Subject{}},
		},
	}
	if err := validateSecurityPolicy(policy); err == nil {
		t.Fatal("expected validation error for empty subjects")
	}
}

func TestValidateSecurityPolicyRejectsInvalidSubjectKind(t *testing.T) {
	t.Parallel()

	policy := &SecurityPolicyConfig{
		RoleMappings: []RoleMapping{
			{Name: "ok", Role: "tier", Subjects: []Subject{{Kind: "Robot", Name: "bot"}}},
		},
	}
	if err := validateSecurityPolicy(policy); err == nil {
		t.Fatal("expected validation error for invalid subject kind")
	}
}

func TestValidateSecurityPolicyRejectsEmptySubjectName(t *testing.T) {
	t.Parallel()

	policy := &SecurityPolicyConfig{
		RoleMappings: []RoleMapping{
			{Name: "ok", Role: "tier", Subjects: []Subject{{Kind: "User", Name: ""}}},
		},
	}
	if err := validateSecurityPolicy(policy); err == nil {
		t.Fatal("expected validation error for empty subject name")
	}
}

func TestValidateSecurityPolicyRejectsRateTierNoLimits(t *testing.T) {
	t.Parallel()

	policy := &SecurityPolicyConfig{
		RateTiers: []RateLimitTier{
			{Name: "no-limits", Group: "g", RequestsPerMin: 0, TokensPerMin: 0},
		},
	}
	if err := validateSecurityPolicy(policy); err == nil {
		t.Fatal("expected validation error for rate tier with no limits")
	}
}

func TestValidateSecurityPolicyRejectsRateTierEmptyName(t *testing.T) {
	t.Parallel()

	policy := &SecurityPolicyConfig{
		RateTiers: []RateLimitTier{
			{Name: "", Group: "g", RequestsPerMin: 10},
		},
	}
	if err := validateSecurityPolicy(policy); err == nil {
		t.Fatal("expected validation error for rate tier with empty name")
	}
}

func TestValidateSecurityPolicyAcceptsValidConfig(t *testing.T) {
	t.Parallel()

	policy := &SecurityPolicyConfig{
		RoleMappings: []RoleMapping{
			{
				Name:      "premium",
				Role:      "premium_tier",
				Subjects:  []Subject{{Kind: "Group", Name: "paying"}},
				ModelRefs: []string{"gpt-4"},
			},
		},
		RateTiers: []RateLimitTier{
			{Name: "default", Group: "all", RequestsPerMin: 60},
		},
	}
	if err := validateSecurityPolicy(policy); err != nil {
		t.Fatalf("expected no validation error, got %v", err)
	}
}

func TestHandleGetSecurityPolicyReturnsEmptyDefault(t *testing.T) {
	saveSecurityPolicy(nil)

	req := httptest.NewRequest(http.MethodGet, "/api/security/policy", nil)
	rr := httptest.NewRecorder()
	HandleGetSecurityPolicy(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d", rr.Code)
	}

	var policy SecurityPolicyConfig
	if err := json.Unmarshal(rr.Body.Bytes(), &policy); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}
	if len(policy.RoleMappings) != 0 {
		t.Fatalf("expected empty role mappings, got %d", len(policy.RoleMappings))
	}
	if len(policy.RateTiers) != 0 {
		t.Fatalf("expected empty rate tiers, got %d", len(policy.RateTiers))
	}
}

func TestHandleUpdateSecurityPolicyRejectsInvalidBody(t *testing.T) {
	t.Parallel()

	req := httptest.NewRequest(http.MethodPut, "/api/security/policy", bytes.NewBufferString("not-json"))
	rr := httptest.NewRecorder()
	HandleUpdateSecurityPolicy(rr, req)

	if rr.Code != http.StatusBadRequest {
		t.Fatalf("expected status 400, got %d", rr.Code)
	}
}

func TestHandleUpdateSecurityPolicyRejectsInvalidPolicy(t *testing.T) {
	t.Parallel()

	policy := SecurityPolicyConfig{
		RoleMappings: []RoleMapping{
			{Name: "", Role: "t", Subjects: []Subject{{Kind: "Group", Name: "g"}}},
		},
	}
	body, _ := json.Marshal(policy)
	req := httptest.NewRequest(http.MethodPut, "/api/security/policy", bytes.NewBuffer(body))
	rr := httptest.NewRecorder()
	HandleUpdateSecurityPolicy(rr, req)

	if rr.Code != http.StatusBadRequest {
		t.Fatalf("expected status 400, got %d", rr.Code)
	}
}

func TestHandleUpdateSecurityPolicySucceeds(t *testing.T) {
	policy := SecurityPolicyConfig{
		RoleMappings: []RoleMapping{
			{
				Name:      "admin-access",
				Role:      "admin_tier",
				Subjects:  []Subject{{Kind: "Group", Name: "admins"}},
				ModelRefs: []string{"gpt-4", "claude-3"},
				Priority:  10,
			},
		},
		RateTiers: []RateLimitTier{
			{Name: "admin-rate", Group: "admins", RequestsPerMin: 1000},
		},
	}
	body, _ := json.Marshal(policy)
	req := httptest.NewRequest(http.MethodPut, "/api/security/policy", bytes.NewBuffer(body))
	rr := httptest.NewRecorder()
	HandleUpdateSecurityPolicy(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d; body: %s", rr.Code, rr.Body.String())
	}

	var resp map[string]interface{}
	if err := json.Unmarshal(rr.Body.Bytes(), &resp); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}
	if resp["message"] == nil {
		t.Fatal("expected message field in response")
	}
	if resp["policy"] == nil {
		t.Fatal("expected policy field in response")
	}
	if resp["fragment"] == nil {
		t.Fatal("expected fragment field in response")
	}

	retPolicy := resp["policy"].(map[string]interface{})
	if retPolicy["updated_at"] == nil || retPolicy["updated_at"] == "" {
		t.Fatal("expected non-empty updated_at in returned policy")
	}
}

func TestHandlePreviewSecurityFragmentReturnsFragment(t *testing.T) {
	t.Parallel()

	policy := SecurityPolicyConfig{
		RoleMappings: []RoleMapping{
			{
				Name:      "viewer",
				Role:      "viewer_tier",
				Subjects:  []Subject{{Kind: "Group", Name: "viewers"}},
				ModelRefs: []string{"gpt-3.5"},
				Priority:  5,
			},
		},
	}
	body, _ := json.Marshal(policy)
	req := httptest.NewRequest(http.MethodPost, "/api/security/policy/preview", bytes.NewBuffer(body))
	rr := httptest.NewRecorder()
	HandlePreviewSecurityFragment(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d; body: %s", rr.Code, rr.Body.String())
	}

	var fragment GeneratedRouterFragment
	if err := json.Unmarshal(rr.Body.Bytes(), &fragment); err != nil {
		t.Fatalf("failed to decode fragment: %v", err)
	}
	if len(fragment.RoleBindings) != 1 {
		t.Fatalf("expected 1 role binding, got %d", len(fragment.RoleBindings))
	}
	if len(fragment.Decisions) != 1 {
		t.Fatalf("expected 1 decision, got %d", len(fragment.Decisions))
	}
	if fragment.Decisions[0].Name != "rbac-viewer" {
		t.Fatalf("expected decision name rbac-viewer, got %q", fragment.Decisions[0].Name)
	}
}

func TestHandlePreviewSecurityFragmentRejectsInvalidBody(t *testing.T) {
	t.Parallel()

	req := httptest.NewRequest(http.MethodPost, "/api/security/policy/preview", bytes.NewBufferString("{bad"))
	rr := httptest.NewRecorder()
	HandlePreviewSecurityFragment(rr, req)

	if rr.Code != http.StatusBadRequest {
		t.Fatalf("expected status 400, got %d", rr.Code)
	}
}

func TestGenerateRouterFragmentCombinedRoleMappingsAndRateTiers(t *testing.T) {
	t.Parallel()

	policy := &SecurityPolicyConfig{
		RoleMappings: []RoleMapping{
			{
				Name:      "premium",
				Subjects:  []Subject{{Kind: "Group", Name: "paying"}},
				Role:      "premium_tier",
				ModelRefs: []string{"gpt-4", "claude-3"},
				Priority:  10,
			},
			{
				Name:      "free",
				Subjects:  []Subject{{Kind: "Group", Name: "free-users"}},
				Role:      "free_tier",
				ModelRefs: []string{"gpt-3.5"},
				Priority:  20,
			},
		},
		RateTiers: []RateLimitTier{
			{Name: "premium-rate", Group: "paying", RequestsPerMin: 1000, TokensPerMin: 100000},
			{Name: "free-rate", Group: "free-users", RequestsPerMin: 10, TokensPerMin: 5000},
		},
	}

	fragment := GenerateRouterFragment(policy)

	if len(fragment.RoleBindings) != 2 {
		t.Fatalf("expected 2 role bindings, got %d", len(fragment.RoleBindings))
	}
	if len(fragment.Decisions) != 2 {
		t.Fatalf("expected 2 decisions, got %d", len(fragment.Decisions))
	}
	if len(fragment.RateLimit.Providers) != 1 {
		t.Fatalf("expected 1 rate limit provider, got %d", len(fragment.RateLimit.Providers))
	}
	if len(fragment.RateLimit.Providers[0].Rules) != 2 {
		t.Fatalf("expected 2 rate rules, got %d", len(fragment.RateLimit.Providers[0].Rules))
	}

	if fragment.Decisions[0].Rules.Name != "premium_tier" {
		t.Fatalf("expected decision[0] rule name premium_tier, got %q", fragment.Decisions[0].Rules.Name)
	}
	if fragment.Decisions[1].Rules.Name != "free_tier" {
		t.Fatalf("expected decision[1] rule name free_tier, got %q", fragment.Decisions[1].Rules.Name)
	}
}

// --- Edge-case tests for validation ---

func TestValidateSecurityPolicyAcceptsTPMOnly(t *testing.T) {
	t.Parallel()

	policy := &SecurityPolicyConfig{
		RateTiers: []RateLimitTier{
			{Name: "tpm-only", Group: "g", RequestsPerMin: 0, TokensPerMin: 5000},
		},
	}
	if err := validateSecurityPolicy(policy); err != nil {
		t.Fatalf("expected no error for TPM-only tier, got %v", err)
	}
}

func TestValidateSecurityPolicyAcceptsRPMOnly(t *testing.T) {
	t.Parallel()

	policy := &SecurityPolicyConfig{
		RateTiers: []RateLimitTier{
			{Name: "rpm-only", Group: "g", RequestsPerMin: 100, TokensPerMin: 0},
		},
	}
	if err := validateSecurityPolicy(policy); err != nil {
		t.Fatalf("expected no error for RPM-only tier, got %v", err)
	}
}

func TestValidateSecurityPolicyAcceptsCaseInsensitiveSubjectKind(t *testing.T) {
	t.Parallel()

	for _, kind := range []string{"user", "User", "USER", "group", "Group", "GROUP"} {
		policy := &SecurityPolicyConfig{
			RoleMappings: []RoleMapping{
				{Name: "test-" + kind, Role: "r", Subjects: []Subject{{Kind: kind, Name: "n"}}},
			},
		}
		if err := validateSecurityPolicy(policy); err != nil {
			t.Fatalf("expected no error for subject kind %q, got %v", kind, err)
		}
	}
}

func TestValidateSecurityPolicyAcceptsEmptyRoleMappingsWithRateTiersOnly(t *testing.T) {
	t.Parallel()

	policy := &SecurityPolicyConfig{
		RateTiers: []RateLimitTier{
			{Name: "global-rate", Group: "all", RequestsPerMin: 60},
		},
	}
	if err := validateSecurityPolicy(policy); err != nil {
		t.Fatalf("expected no error with only rate tiers, got %v", err)
	}
}

func TestValidateSecurityPolicyAcceptsEmptyPolicy(t *testing.T) {
	t.Parallel()

	policy := &SecurityPolicyConfig{}
	if err := validateSecurityPolicy(policy); err != nil {
		t.Fatalf("expected no error for empty policy, got %v", err)
	}
}

func TestValidateSecurityPolicyAcceptsMixedSubjectKinds(t *testing.T) {
	t.Parallel()

	policy := &SecurityPolicyConfig{
		RoleMappings: []RoleMapping{
			{
				Name:     "mixed",
				Role:     "tier",
				Subjects: []Subject{{Kind: "User", Name: "alice"}, {Kind: "Group", Name: "eng"}},
			},
		},
	}
	if err := validateSecurityPolicy(policy); err != nil {
		t.Fatalf("expected no error for mixed subject kinds, got %v", err)
	}
}

// --- Edge-case tests for fragment generation ---

func TestGenerateRouterFragmentNoModelRefs(t *testing.T) {
	t.Parallel()

	policy := &SecurityPolicyConfig{
		RoleMappings: []RoleMapping{
			{
				Name:      "no-models",
				Subjects:  []Subject{{Kind: "Group", Name: "g"}},
				Role:      "tier",
				ModelRefs: []string{},
				Priority:  1,
			},
		},
	}

	fragment := GenerateRouterFragment(policy)

	if len(fragment.Decisions[0].ModelRefs) != 0 {
		t.Fatalf("expected 0 model refs, got %d", len(fragment.Decisions[0].ModelRefs))
	}
}

func TestGenerateRouterFragmentZeroPriority(t *testing.T) {
	t.Parallel()

	policy := &SecurityPolicyConfig{
		RoleMappings: []RoleMapping{
			{
				Name:     "zero-pri",
				Subjects: []Subject{{Kind: "User", Name: "u"}},
				Role:     "r",
				Priority: 0,
			},
		},
	}

	fragment := GenerateRouterFragment(policy)
	if fragment.Decisions[0].Priority != 0 {
		t.Fatalf("expected priority 0, got %d", fragment.Decisions[0].Priority)
	}
	if fragment.Decisions[0].Name != "rbac-zero-pri" {
		t.Fatalf("expected decision name rbac-zero-pri, got %q", fragment.Decisions[0].Name)
	}
}

func TestGenerateRouterFragmentUserOnlyRateTier(t *testing.T) {
	t.Parallel()

	policy := &SecurityPolicyConfig{
		RateTiers: []RateLimitTier{
			{Name: "user-limit", User: "admin@co.com", RequestsPerMin: 999},
		},
	}

	fragment := GenerateRouterFragment(policy)
	rule := fragment.RateLimit.Providers[0].Rules[0]
	if rule.Match.User != "admin@co.com" {
		t.Fatalf("expected user match admin@co.com, got %q", rule.Match.User)
	}
	if rule.Match.Group != "" {
		t.Fatalf("expected empty group match, got %q", rule.Match.Group)
	}
}

func TestGenerateRouterFragmentBothUserAndGroupRateTier(t *testing.T) {
	t.Parallel()

	policy := &SecurityPolicyConfig{
		RateTiers: []RateLimitTier{
			{Name: "both", User: "admin", Group: "admins", RequestsPerMin: 100},
		},
	}

	fragment := GenerateRouterFragment(policy)
	rule := fragment.RateLimit.Providers[0].Rules[0]
	if rule.Match.User != "admin" {
		t.Fatalf("expected user match admin, got %q", rule.Match.User)
	}
	if rule.Match.Group != "admins" {
		t.Fatalf("expected group match admins, got %q", rule.Match.Group)
	}
}

func TestGenerateRouterFragmentTPMOnlyRateTier(t *testing.T) {
	t.Parallel()

	policy := &SecurityPolicyConfig{
		RateTiers: []RateLimitTier{
			{Name: "tpm-only", Group: "g", TokensPerMin: 50000},
		},
	}

	fragment := GenerateRouterFragment(policy)
	rule := fragment.RateLimit.Providers[0].Rules[0]
	if rule.RPU != 0 {
		t.Fatalf("expected RPU 0 for TPM-only tier, got %d", rule.RPU)
	}
	if rule.TPU != 50000 {
		t.Fatalf("expected TPU 50000, got %d", rule.TPU)
	}
}

func TestGenerateRouterFragmentSubjectKindPassthrough(t *testing.T) {
	t.Parallel()

	policy := &SecurityPolicyConfig{
		RoleMappings: []RoleMapping{
			{
				Name:     "mixed-subjects",
				Subjects: []Subject{{Kind: "User", Name: "alice"}, {Kind: "Group", Name: "eng"}},
				Role:     "test_tier",
			},
		},
	}

	fragment := GenerateRouterFragment(policy)
	if fragment.RoleBindings[0].Subjects[0].Kind != "User" {
		t.Fatalf("expected subject kind User, got %q", fragment.RoleBindings[0].Subjects[0].Kind)
	}
	if fragment.RoleBindings[0].Subjects[1].Kind != "Group" {
		t.Fatalf("expected subject kind Group, got %q", fragment.RoleBindings[0].Subjects[1].Kind)
	}
}

// --- Edge-case tests for handlers ---

func TestHandlePreviewWithRateTiers(t *testing.T) {
	t.Parallel()

	policy := SecurityPolicyConfig{
		RoleMappings: []RoleMapping{
			{
				Name:      "vip",
				Subjects:  []Subject{{Kind: "Group", Name: "vips"}},
				Role:      "vip_tier",
				ModelRefs: []string{"gpt-4"},
				Priority:  5,
			},
		},
		RateTiers: []RateLimitTier{
			{Name: "vip-rate", Group: "vips", RequestsPerMin: 500, TokensPerMin: 100000},
		},
	}
	body, _ := json.Marshal(policy)
	req := httptest.NewRequest(http.MethodPost, "/api/security/policy/preview", bytes.NewBuffer(body))
	rr := httptest.NewRecorder()
	HandlePreviewSecurityFragment(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d; body: %s", rr.Code, rr.Body.String())
	}

	var fragment GeneratedRouterFragment
	if err := json.Unmarshal(rr.Body.Bytes(), &fragment); err != nil {
		t.Fatalf("decode error: %v", err)
	}
	if len(fragment.RateLimit.Providers) != 1 {
		t.Fatalf("expected 1 ratelimit provider, got %d", len(fragment.RateLimit.Providers))
	}
	if len(fragment.RateLimit.Providers[0].Rules) != 1 {
		t.Fatalf("expected 1 rate rule, got %d", len(fragment.RateLimit.Providers[0].Rules))
	}
	if fragment.RateLimit.Providers[0].Rules[0].RPU != 500 {
		t.Fatalf("expected RPU 500, got %d", fragment.RateLimit.Providers[0].Rules[0].RPU)
	}
}

func TestHandleUpdateResponseContainsAppliedField(t *testing.T) {
	policy := SecurityPolicyConfig{
		RoleMappings: []RoleMapping{
			{Name: "t", Role: "r", Subjects: []Subject{{Kind: "Group", Name: "g"}}},
		},
	}
	body, _ := json.Marshal(policy)
	req := httptest.NewRequest(http.MethodPut, "/api/security/policy", bytes.NewBuffer(body))
	rr := httptest.NewRecorder()
	HandleUpdateSecurityPolicy(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d; body: %s", rr.Code, rr.Body.String())
	}

	var resp map[string]interface{}
	if err := json.Unmarshal(rr.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode error: %v", err)
	}
	if _, ok := resp["applied"]; !ok {
		t.Fatal("expected 'applied' field in response")
	}
}

func TestSaveAndLoadPolicyPersistence(t *testing.T) {
	policy := &SecurityPolicyConfig{
		RoleMappings: []RoleMapping{
			{Name: "persist-test", Role: "r", Subjects: []Subject{{Kind: "User", Name: "u"}}},
		},
		RateTiers: []RateLimitTier{{Name: "rate-persist", Group: "g", RequestsPerMin: 42}},
		UpdatedAt: "2026-01-01T00:00:00Z",
	}

	saveSecurityPolicy(policy)
	loaded := loadSecurityPolicy()

	if len(loaded.RoleMappings) != 1 || loaded.RoleMappings[0].Name != "persist-test" {
		t.Fatalf("expected persisted role mapping, got %+v", loaded.RoleMappings)
	}
	if len(loaded.RateTiers) != 1 || loaded.RateTiers[0].Name != "rate-persist" {
		t.Fatalf("expected persisted rate tier, got %+v", loaded.RateTiers)
	}
	if loaded.UpdatedAt != "2026-01-01T00:00:00Z" {
		t.Fatalf("expected persisted timestamp, got %q", loaded.UpdatedAt)
	}
}
