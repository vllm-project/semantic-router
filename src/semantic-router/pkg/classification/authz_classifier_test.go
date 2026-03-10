package classification

import (
	"testing"

	. "github.com/onsi/gomega"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestAuthzClassifierCreation(t *testing.T) {
	RegisterTestingT(t)

	t.Run("empty bindings is valid", func(t *testing.T) {
		RegisterTestingT(t)
		c, err := NewAuthzClassifier(nil)
		Expect(err).NotTo(HaveOccurred())
		Expect(c).NotTo(BeNil())
	})

	t.Run("rejects binding with empty name", func(t *testing.T) {
		RegisterTestingT(t)
		_, err := NewAuthzClassifier([]config.RoleBinding{
			{Name: "", Role: "admin", Subjects: []config.Subject{{Kind: "User", Name: "root"}}},
		})
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("empty name"))
	})

	t.Run("rejects duplicate binding names", func(t *testing.T) {
		RegisterTestingT(t)
		_, err := NewAuthzClassifier([]config.RoleBinding{
			{Name: "same-name", Role: "admin", Subjects: []config.Subject{{Kind: "User", Name: "root"}}},
			{Name: "same-name", Role: "viewer", Subjects: []config.Subject{{Kind: "User", Name: "alice"}}},
		})
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("duplicate binding name"))
		Expect(err.Error()).To(ContainSubstring("same-name"))
	})

	t.Run("rejects binding with empty role", func(t *testing.T) {
		RegisterTestingT(t)
		_, err := NewAuthzClassifier([]config.RoleBinding{
			{Name: "test", Role: "", Subjects: []config.Subject{{Kind: "User", Name: "root"}}},
		})
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("empty role"))
	})

	t.Run("rejects binding with no subjects", func(t *testing.T) {
		RegisterTestingT(t)
		_, err := NewAuthzClassifier([]config.RoleBinding{
			{Name: "test", Role: "admin", Subjects: nil},
		})
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("no subjects"))
	})

	t.Run("rejects subject with invalid kind", func(t *testing.T) {
		RegisterTestingT(t)
		_, err := NewAuthzClassifier([]config.RoleBinding{
			{Name: "test", Role: "admin", Subjects: []config.Subject{
				{Kind: "ServiceAccount", Name: "default"},
			}},
		})
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("invalid kind"))
		Expect(err.Error()).To(ContainSubstring("ServiceAccount"))
	})

	t.Run("rejects subject with empty name", func(t *testing.T) {
		RegisterTestingT(t)
		_, err := NewAuthzClassifier([]config.RoleBinding{
			{Name: "test", Role: "admin", Subjects: []config.Subject{
				{Kind: "User", Name: ""},
			}},
		})
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("empty name"))
	})

	t.Run("rejects subject with whitespace-only name", func(t *testing.T) {
		RegisterTestingT(t)
		_, err := NewAuthzClassifier([]config.RoleBinding{
			{Name: "test", Role: "admin", Subjects: []config.Subject{
				{Kind: "User", Name: "   "},
			}},
		})
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("empty name"))
	})

	t.Run("accepts kind case-insensitively", func(t *testing.T) {
		RegisterTestingT(t)
		c, err := NewAuthzClassifier([]config.RoleBinding{
			{Name: "test", Role: "admin", Subjects: []config.Subject{
				{Kind: "user", Name: "admin"},
				{Kind: "GROUP", Name: "ops"},
			}},
		})
		Expect(err).NotTo(HaveOccurred())
		Expect(c).NotTo(BeNil())
	})

	t.Run("valid binding with User subject", func(t *testing.T) {
		RegisterTestingT(t)
		c, err := NewAuthzClassifier([]config.RoleBinding{
			{Name: "admin-binding", Role: "admin", Subjects: []config.Subject{
				{Kind: "User", Name: "admin"},
			}},
		})
		Expect(err).NotTo(HaveOccurred())
		Expect(c).NotTo(BeNil())
	})

	t.Run("valid binding with Group subject", func(t *testing.T) {
		RegisterTestingT(t)
		c, err := NewAuthzClassifier([]config.RoleBinding{
			{Name: "premium-binding", Role: "premium_tier", Subjects: []config.Subject{
				{Kind: "Group", Name: "premium"},
			}},
		})
		Expect(err).NotTo(HaveOccurred())
		Expect(c).NotTo(BeNil())
	})

	t.Run("normalizes subject name whitespace at creation", func(t *testing.T) {
		RegisterTestingT(t)
		// Subject names with leading/trailing whitespace are trimmed at creation time
		// so they match header values correctly at runtime.
		c, err := NewAuthzClassifier([]config.RoleBinding{
			{Name: "ws-test", Role: "trimmed_role", Subjects: []config.Subject{
				{Kind: "User", Name: "  admin  "},
			}},
		})
		Expect(err).NotTo(HaveOccurred())
		Expect(c).NotTo(BeNil())

		// The trimmed name should match "admin" at runtime
		result, err := c.Classify("admin", nil)
		Expect(err).NotTo(HaveOccurred())
		Expect(result.MatchedRules).To(ConsistOf("trimmed_role"))

		// The original untrimmed value should NOT match
		result, err = c.Classify("  admin  ", nil)
		Expect(err).NotTo(HaveOccurred())
		Expect(result.MatchedRules).To(BeEmpty())
	})

	t.Run("normalizes subject kind at creation for matching", func(t *testing.T) {
		RegisterTestingT(t)
		// Kind: "  User  " is normalized to "user" at creation time
		c, err := NewAuthzClassifier([]config.RoleBinding{
			{Name: "kind-norm-test", Role: "some_role", Subjects: []config.Subject{
				{Kind: "  Group  ", Name: "ops"},
			}},
		})
		Expect(err).NotTo(HaveOccurred())
		result, err := c.Classify("anyone", []string{"ops"})
		Expect(err).NotTo(HaveOccurred())
		Expect(result.MatchedRules).To(ConsistOf("some_role"))
	})
}

func TestAuthzClassifierClassify(t *testing.T) {
	RegisterTestingT(t)

	// K8s RBAC-style role bindings
	bindings := []config.RoleBinding{
		{
			Name:        "premium-users",
			Description: "Premium and enterprise users",
			Role:        "premium_tier",
			Subjects: []config.Subject{
				{Kind: "Group", Name: "premium"},
				{Kind: "Group", Name: "enterprise"},
			},
		},
		{
			Name:        "free-users",
			Description: "Free and trial users",
			Role:        "free_tier",
			Subjects: []config.Subject{
				{Kind: "Group", Name: "free"},
				{Kind: "Group", Name: "trial"},
			},
		},
		{
			Name:        "admin-override",
			Description: "Admin users — unrestricted",
			Role:        "admin",
			Subjects: []config.Subject{
				{Kind: "User", Name: "admin"},
				{Kind: "User", Name: "root"},
			},
		},
	}

	c, err := NewAuthzClassifier(bindings)
	Expect(err).NotTo(HaveOccurred())

	t.Run("User subject match returns role name", func(t *testing.T) {
		RegisterTestingT(t)
		result, err := c.Classify("admin", nil)
		Expect(err).NotTo(HaveOccurred())
		Expect(result.MatchedRules).To(ConsistOf("admin"))
	})

	t.Run("Group subject match returns role name", func(t *testing.T) {
		RegisterTestingT(t)
		result, err := c.Classify("user123", []string{"premium"})
		Expect(err).NotTo(HaveOccurred())
		Expect(result.MatchedRules).To(ConsistOf("premium_tier"))
	})

	t.Run("multi-group match returns multiple roles", func(t *testing.T) {
		RegisterTestingT(t)
		result, err := c.Classify("user456", []string{"free", "enterprise"})
		Expect(err).NotTo(HaveOccurred())
		Expect(result.MatchedRules).To(ConsistOf("premium_tier", "free_tier"))
	})

	t.Run("User + Group match returns both roles", func(t *testing.T) {
		RegisterTestingT(t)
		result, err := c.Classify("admin", []string{"premium"})
		Expect(err).NotTo(HaveOccurred())
		// admin matches admin binding by User, premium binding by Group
		Expect(result.MatchedRules).To(ConsistOf("admin", "premium_tier"))
	})

	t.Run("no match returns empty", func(t *testing.T) {
		RegisterTestingT(t)
		result, err := c.Classify("stranger", []string{"visitors"})
		Expect(err).NotTo(HaveOccurred())
		Expect(result.MatchedRules).To(BeEmpty())
	})

	t.Run("empty userID with bindings configured returns error", func(t *testing.T) {
		RegisterTestingT(t)
		_, err := c.Classify("", nil)
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("user identity header is empty"))
		Expect(err.Error()).To(ContainSubstring("no silent bypass"))
	})

	t.Run("empty bindings returns empty result for any user", func(t *testing.T) {
		RegisterTestingT(t)
		emptyC, err := NewAuthzClassifier(nil)
		Expect(err).NotTo(HaveOccurred())
		result, err := emptyC.Classify("", nil)
		Expect(err).NotTo(HaveOccurred())
		Expect(result.MatchedRules).To(BeEmpty())
	})

	t.Run("empty bindings returns empty result for non-empty user", func(t *testing.T) {
		RegisterTestingT(t)
		emptyC, err := NewAuthzClassifier(nil)
		Expect(err).NotTo(HaveOccurred())
		result, err := emptyC.Classify("alice", []string{"premium"})
		Expect(err).NotTo(HaveOccurred())
		Expect(result.MatchedRules).To(BeEmpty())
	})

	t.Run("multiple bindings to same role are deduplicated", func(t *testing.T) {
		RegisterTestingT(t)
		// Two separate bindings granting the same role
		multiBindings := []config.RoleBinding{
			{
				Name: "apac-premium",
				Role: "premium_tier",
				Subjects: []config.Subject{
					{Kind: "Group", Name: "apac-enterprise"},
				},
			},
			{
				Name: "emea-premium",
				Role: "premium_tier",
				Subjects: []config.Subject{
					{Kind: "Group", Name: "emea-enterprise"},
				},
			},
		}
		multiC, err := NewAuthzClassifier(multiBindings)
		Expect(err).NotTo(HaveOccurred())

		// User in both groups should get premium_tier only once
		result, err := multiC.Classify("user1", []string{"apac-enterprise", "emea-enterprise"})
		Expect(err).NotTo(HaveOccurred())
		Expect(result.MatchedRules).To(Equal([]string{"premium_tier"}))
	})

	t.Run("exact name match required — no partial matching", func(t *testing.T) {
		RegisterTestingT(t)
		// "admi" should NOT match "admin"
		result, err := c.Classify("admi", nil)
		Expect(err).NotTo(HaveOccurred())
		Expect(result.MatchedRules).To(BeEmpty())
	})

	t.Run("group name is case-sensitive", func(t *testing.T) {
		RegisterTestingT(t)
		// "Premium" should NOT match "premium"
		result, err := c.Classify("user1", []string{"Premium"})
		Expect(err).NotTo(HaveOccurred())
		Expect(result.MatchedRules).To(BeEmpty())
	})

	t.Run("user ID is case-sensitive", func(t *testing.T) {
		RegisterTestingT(t)
		// "Admin" should NOT match "admin"
		result, err := c.Classify("Admin", nil)
		Expect(err).NotTo(HaveOccurred())
		Expect(result.MatchedRules).To(BeEmpty())
	})
}

func TestIdentityConfigDefaults(t *testing.T) {
	RegisterTestingT(t)

	t.Run("default identity config uses Authorino header names", func(t *testing.T) {
		RegisterTestingT(t)
		ic := config.IdentityConfig{}
		Expect(ic.GetUserIDHeader()).To(Equal("x-authz-user-id"))
		Expect(ic.GetUserGroupsHeader()).To(Equal("x-authz-user-groups"))
	})

	t.Run("custom identity config overrides header names", func(t *testing.T) {
		RegisterTestingT(t)
		ic := config.IdentityConfig{
			UserIDHeader:     "x-jwt-sub",
			UserGroupsHeader: "x-jwt-groups",
		}
		Expect(ic.GetUserIDHeader()).To(Equal("x-jwt-sub"))
		Expect(ic.GetUserGroupsHeader()).To(Equal("x-jwt-groups"))
	})

	t.Run("partial override — only user_id_header set", func(t *testing.T) {
		RegisterTestingT(t)
		ic := config.IdentityConfig{
			UserIDHeader: "x-forwarded-user",
		}
		Expect(ic.GetUserIDHeader()).To(Equal("x-forwarded-user"))
		Expect(ic.GetUserGroupsHeader()).To(Equal("x-authz-user-groups")) // default
	})

	t.Run("partial override — only user_groups_header set", func(t *testing.T) {
		RegisterTestingT(t)
		ic := config.IdentityConfig{
			UserGroupsHeader: "x-forwarded-groups",
		}
		Expect(ic.GetUserIDHeader()).To(Equal("x-authz-user-id")) // default
		Expect(ic.GetUserGroupsHeader()).To(Equal("x-forwarded-groups"))
	})
}

func TestClassifierCustomIdentityHeaders(t *testing.T) {
	RegisterTestingT(t)

	t.Run("classifier stores resolved identity headers from config", func(t *testing.T) {
		RegisterTestingT(t)
		cfg := &config.RouterConfig{
			Authz: config.AuthzConfig{
				Identity: config.IdentityConfig{
					UserIDHeader:     "x-jwt-sub",
					UserGroupsHeader: "x-jwt-groups",
				},
			},
		}
		c, err := newClassifierWithOptions(cfg)
		Expect(err).NotTo(HaveOccurred())
		Expect(c.authzUserIDHeader).To(Equal("x-jwt-sub"))
		Expect(c.authzUserGroupsHeader).To(Equal("x-jwt-groups"))
	})

	t.Run("classifier uses default identity headers when config is empty", func(t *testing.T) {
		RegisterTestingT(t)
		cfg := &config.RouterConfig{}
		c, err := newClassifierWithOptions(cfg)
		Expect(err).NotTo(HaveOccurred())
		Expect(c.authzUserIDHeader).To(Equal("x-authz-user-id"))
		Expect(c.authzUserGroupsHeader).To(Equal("x-authz-user-groups"))
	})
}

func TestParseUserGroups(t *testing.T) {
	RegisterTestingT(t)

	t.Run("empty string", func(t *testing.T) {
		RegisterTestingT(t)
		Expect(ParseUserGroups("")).To(BeNil())
	})

	t.Run("single group", func(t *testing.T) {
		RegisterTestingT(t)
		Expect(ParseUserGroups("premium")).To(Equal([]string{"premium"}))
	})

	t.Run("multiple groups", func(t *testing.T) {
		RegisterTestingT(t)
		Expect(ParseUserGroups("premium,basic,admin")).To(Equal([]string{"premium", "basic", "admin"}))
	})

	t.Run("whitespace trimmed", func(t *testing.T) {
		RegisterTestingT(t)
		Expect(ParseUserGroups(" premium , basic , admin ")).To(Equal([]string{"premium", "basic", "admin"}))
	})

	t.Run("empty entries excluded", func(t *testing.T) {
		RegisterTestingT(t)
		Expect(ParseUserGroups("premium,,basic,")).To(Equal([]string{"premium", "basic"}))
	})
}
