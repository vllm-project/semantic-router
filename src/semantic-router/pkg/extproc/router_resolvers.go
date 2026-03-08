package extproc

import (
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/authz"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/ratelimit"
)

// knownAuthzProviderTypes is the set of valid provider type strings.
var knownAuthzProviderTypes = map[string]bool{
	"header-injection": true,
	"static-config":    true,
}

// knownRateLimitProviderTypes is the set of valid rate limit provider type strings.
var knownRateLimitProviderTypes = map[string]bool{
	"envoy-ratelimit": true,
	"local-limiter":   true,
}

// buildCredentialResolver constructs the credential provider chain from config.
func buildCredentialResolver(cfg *config.RouterConfig) *authz.CredentialResolver {
	authzCfg := cfg.Authz

	if len(authzCfg.Providers) == 0 {
		resolver := buildDefaultCredentialResolver(cfg, true)
		logging.Infof("Authz: using default chain [header-injection(defaults) → static-config], fail_open=true (no explicit providers configured)")
		logAuthzIdentity(cfg)
		return resolver
	}

	validationErrors := validateAuthzProviders(authzCfg.Providers)
	if len(validationErrors) > 0 {
		logValidationErrors("Authz", validationErrors)
		logging.Errorf(
			"Authz config has %d validation error(s) — falling back to default chain to prevent startup failure. FIX YOUR CONFIG.",
			len(validationErrors),
		)
		resolver := buildDefaultCredentialResolver(cfg, false)
		logAuthzIdentity(cfg)
		return resolver
	}

	providers := createAuthzProviders(cfg)
	resolver := authz.NewCredentialResolver(providers...)
	resolver.SetFailOpen(authzCfg.FailOpen)

	logCredentialResolverStatus(cfg, resolver, authzCfg.FailOpen)
	return resolver
}

func buildDefaultCredentialResolver(cfg *config.RouterConfig, failOpen bool) *authz.CredentialResolver {
	resolver := authz.NewCredentialResolver(
		authz.NewHeaderInjectionProvider(authz.DefaultHeaderMap()),
		authz.NewStaticConfigProvider(cfg),
	)
	resolver.SetFailOpen(failOpen)
	return resolver
}

func validateAuthzProviders(providers []config.AuthzProviderConfig) []string {
	validationErrors := make([]string, 0)
	for i, provider := range providers {
		validationErrors = append(validationErrors, validateAuthzProvider(i, provider)...)
	}
	return validationErrors
}

func validateAuthzProvider(index int, provider config.AuthzProviderConfig) []string {
	validationErrors := make([]string, 0)
	if !knownAuthzProviderTypes[provider.Type] {
		validationErrors = append(
			validationErrors,
			fmt.Sprintf(
				"authz.providers[%d]: unknown type %q (valid types: header-injection, static-config)",
				index,
				provider.Type,
			),
		)
	}

	if provider.Type != "header-injection" || len(provider.Headers) == 0 {
		return validationErrors
	}

	for providerName, header := range provider.Headers {
		if header == "" {
			validationErrors = append(
				validationErrors,
				fmt.Sprintf(
					"authz.providers[%d].headers: provider %q has empty header name",
					index,
					providerName,
				),
			)
		}
	}
	return validationErrors
}

func createAuthzProviders(cfg *config.RouterConfig) []authz.Provider {
	providers := make([]authz.Provider, 0, len(cfg.Authz.Providers))
	for _, provider := range cfg.Authz.Providers {
		providers = append(providers, buildAuthzProvider(cfg, provider)...)
	}
	return providers
}

func buildAuthzProvider(cfg *config.RouterConfig, provider config.AuthzProviderConfig) []authz.Provider {
	switch provider.Type {
	case "header-injection":
		headerProvider := authz.NewHeaderInjectionProvider(provider.Headers)
		if len(provider.Headers) == 0 {
			logging.Infof("Authz: header-injection provider using default headers: %v", authz.DefaultHeaderMap())
		} else {
			logging.Infof("Authz: header-injection provider using custom headers: %v", provider.Headers)
		}
		return []authz.Provider{headerProvider}
	case "static-config":
		logging.Infof("Authz: static-config provider (reads model_config.*.access_key)")
		return []authz.Provider{authz.NewStaticConfigProvider(cfg)}
	default:
		return nil
	}
}

func logCredentialResolverStatus(
	cfg *config.RouterConfig,
	resolver *authz.CredentialResolver,
	failOpen bool,
) {
	if failOpen {
		logging.Warnf("Authz fail_open=true — requests without valid credentials will be allowed through. Ensure this is intentional.")
	}
	logging.Infof("Authz: chain=%v, fail_open=%v", resolver.ProviderNames(), failOpen)
	logAuthzIdentity(cfg)
}

func logAuthzIdentity(cfg *config.RouterConfig) {
	logging.Infof(
		"Authz identity: user_id_header=%q, user_groups_header=%q",
		cfg.Authz.Identity.GetUserIDHeader(),
		cfg.Authz.Identity.GetUserGroupsHeader(),
	)
}

func logValidationErrors(prefix string, validationErrors []string) {
	for _, validationError := range validationErrors {
		logging.Errorf("%s config validation error: %s", prefix, validationError)
	}
}

// buildRateLimitResolver constructs the rate limit provider chain from config.
func buildRateLimitResolver(cfg *config.RouterConfig) *ratelimit.RateLimitResolver {
	rlCfg := cfg.RateLimit

	if len(rlCfg.Providers) == 0 {
		logging.Infof("RateLimit: no providers configured, rate limiting disabled")
		return nil
	}

	validationErrors := validateRateLimitProviders(rlCfg.Providers)
	if len(validationErrors) > 0 {
		logValidationErrors("RateLimit", validationErrors)
		logging.Errorf(
			"RateLimit config has %d validation error(s) — rate limiting disabled",
			len(validationErrors),
		)
		return nil
	}

	providers := createRateLimitProviders(rlCfg.Providers)
	if len(providers) == 0 {
		logging.Warnf("RateLimit: no valid providers after construction, rate limiting disabled")
		return nil
	}

	resolver := ratelimit.NewRateLimitResolver(providers...)
	resolver.SetFailOpen(rlCfg.FailOpen)
	if rlCfg.FailOpen {
		logging.Warnf("RateLimit fail_open=true — provider errors will not block requests")
	}
	logging.Infof("RateLimit: chain=%v, fail_open=%v", resolver.ProviderNames(), rlCfg.FailOpen)
	return resolver
}

func validateRateLimitProviders(providers []config.RateLimitProviderConfig) []string {
	validationErrors := make([]string, 0)
	for i, provider := range providers {
		validationErrors = append(validationErrors, validateRateLimitProvider(i, provider)...)
	}
	return validationErrors
}

func validateRateLimitProvider(index int, provider config.RateLimitProviderConfig) []string {
	validationErrors := make([]string, 0)
	if !knownRateLimitProviderTypes[provider.Type] {
		validationErrors = append(
			validationErrors,
			fmt.Sprintf(
				"ratelimit.providers[%d]: unknown type %q (valid types: envoy-ratelimit, local-limiter)",
				index,
				provider.Type,
			),
		)
	}
	if provider.Type == "envoy-ratelimit" && provider.Address == "" {
		validationErrors = append(
			validationErrors,
			fmt.Sprintf("ratelimit.providers[%d]: envoy-ratelimit requires 'address'", index),
		)
	}
	if provider.Type == "local-limiter" && len(provider.Rules) == 0 {
		validationErrors = append(
			validationErrors,
			fmt.Sprintf("ratelimit.providers[%d]: local-limiter requires at least one rule", index),
		)
	}
	return validationErrors
}

func createRateLimitProviders(providersCfg []config.RateLimitProviderConfig) []ratelimit.Provider {
	providers := make([]ratelimit.Provider, 0, len(providersCfg))
	for _, providerCfg := range providersCfg {
		provider := buildRateLimitProvider(providerCfg)
		if provider != nil {
			providers = append(providers, provider)
		}
	}
	return providers
}

func buildRateLimitProvider(providerCfg config.RateLimitProviderConfig) ratelimit.Provider {
	switch providerCfg.Type {
	case "envoy-ratelimit":
		return buildEnvoyRateLimitProvider(providerCfg)
	case "local-limiter":
		return buildLocalRateLimitProvider(providerCfg)
	default:
		return nil
	}
}

func buildEnvoyRateLimitProvider(providerCfg config.RateLimitProviderConfig) ratelimit.Provider {
	domain := providerCfg.Domain
	if domain == "" {
		domain = "semantic-router"
	}

	envoyProvider, err := ratelimit.NewEnvoyRLSProvider(providerCfg.Address, domain)
	if err != nil {
		logging.Errorf("RateLimit: failed to create envoy-ratelimit provider: %v", err)
		return nil
	}
	logging.Infof("RateLimit: envoy-ratelimit provider at %s (domain=%s)", providerCfg.Address, domain)
	return envoyProvider
}

func buildLocalRateLimitProvider(providerCfg config.RateLimitProviderConfig) ratelimit.Provider {
	rules := buildLocalRateLimitRules(providerCfg.Rules)
	logging.Infof("RateLimit: local-limiter provider with %d rules", len(rules))
	return ratelimit.NewLocalLimiter(rules)
}

func buildLocalRateLimitRules(rulesCfg []config.RateLimitRule) []ratelimit.Rule {
	rules := make([]ratelimit.Rule, 0, len(rulesCfg))
	for _, rule := range rulesCfg {
		rules = append(rules, ratelimit.Rule{
			Name: rule.Name,
			Match: ratelimit.RuleMatch{
				User:  rule.Match.User,
				Group: rule.Match.Group,
				Model: rule.Match.Model,
			},
			RequestsPerUnit: rule.RequestsPerUnit,
			TokensPerUnit:   rule.TokensPerUnit,
			Unit:            ratelimit.ParseUnit(rule.Unit),
		})
	}
	return rules
}
