package config

import "testing"

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

func testAuthFlags(jwtTTL string) authFlags {
	empty := ""
	dbPath := "auth.db"
	return authFlags{
		dbPath:            &dbPath,
		jwtSecret:         &empty,
		jwtTTL:            &jwtTTL,
		bootstrapEmail:    &empty,
		bootstrapPassword: &empty,
		bootstrapName:     &empty,
		passwordBlocklist: &empty,
	}
}
