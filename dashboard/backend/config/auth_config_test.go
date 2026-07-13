package config

import (
	"strings"
	"testing"
)

func TestApplyAuthConfigBoundsJWTExpiry(t *testing.T) {
	t.Parallel()

	for _, value := range []string{"0", "-1", "169", "2562048"} {
		t.Run(value, func(t *testing.T) {
			cfg := &Config{}
			flags := testAuthFlags(value)
			if err := applyAuthConfig(cfg, flags); err == nil {
				t.Fatalf("applyAuthConfig(jwtTTL=%q) succeeded", value)
			}
		})
	}
	for _, value := range []string{"1", "168"} {
		t.Run(value, func(t *testing.T) {
			cfg := &Config{}
			flags := testAuthFlags(value)
			if err := applyAuthConfig(cfg, flags); err != nil {
				t.Fatalf("applyAuthConfig(jwtTTL=%q) error = %v", value, err)
			}
		})
	}
}

func TestApplyAuthConfigEnforcesProductionPasswordCorpusContract(t *testing.T) {
	t.Parallel()

	validDigest := strings.Repeat("A", 64)
	tests := []struct {
		name        string
		profile     string
		path        string
		digest      string
		wantError   bool
		wantDigest  string
		wantProfile string
	}{
		{
			name:        "development built in",
			profile:     DashboardSecurityProfileDevelopment,
			wantProfile: DashboardSecurityProfileDevelopment,
		},
		{
			name:      "production missing file",
			profile:   DashboardSecurityProfileProduction,
			digest:    validDigest,
			wantError: true,
		},
		{
			name:      "production missing digest",
			profile:   DashboardSecurityProfileProduction,
			path:      "/passwords.txt",
			wantError: true,
		},
		{
			name:      "invalid profile",
			profile:   "staging",
			wantError: true,
		},
		{
			name:      "invalid digest",
			profile:   DashboardSecurityProfileDevelopment,
			path:      "/passwords.txt",
			digest:    "not-a-digest",
			wantError: true,
		},
		{
			name:      "digest without file",
			profile:   DashboardSecurityProfileDevelopment,
			digest:    validDigest,
			wantError: true,
		},
		{
			name:        "production complete",
			profile:     DashboardSecurityProfileProduction,
			path:        "/passwords.txt",
			digest:      validDigest,
			wantDigest:  strings.ToLower(validDigest),
			wantProfile: DashboardSecurityProfileProduction,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			flags := testAuthFlags("12")
			flags.securityProfile = &test.profile
			flags.passwordBlocklist = &test.path
			flags.blocklistSHA256 = &test.digest
			cfg := &Config{}
			err := applyAuthConfig(cfg, flags)
			if test.wantError {
				if err == nil {
					t.Fatal("applyAuthConfig() succeeded")
				}
				return
			}
			if err != nil {
				t.Fatalf("applyAuthConfig() error = %v", err)
			}
			if cfg.SecurityProfile != test.wantProfile ||
				cfg.PasswordBlocklistSHA256 != test.wantDigest {
				t.Fatalf("config = %#v", cfg)
			}
		})
	}
}

func testAuthFlags(jwtTTL string) authFlags {
	empty := ""
	development := DashboardSecurityProfileDevelopment
	dbPath := "auth.db"
	return authFlags{
		dbPath:            &dbPath,
		jwtSecret:         &empty,
		jwtTTL:            &jwtTTL,
		bootstrapEmail:    &empty,
		bootstrapPassword: &empty,
		bootstrapName:     &empty,
		securityProfile:   &development,
		passwordBlocklist: &empty,
		blocklistSHA256:   &empty,
	}
}
