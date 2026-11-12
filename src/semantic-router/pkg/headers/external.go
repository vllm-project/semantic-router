package headers

// Headers of external contracts. Their names belong to Envoy, HTTP proxies,
// rate-limit conventions and model providers, so they keep those names.

// Name prefixes of the namespaces the Router and Envoy own.
const (
	VSRPrefix   = "x-vsr-"
	EnvoyPrefix = "x-envoy-"
)

// Name prefixes of other header families: every extension header, the
// identity an authorization gateway sets, and the proxy headers.
const (
	ExtensionPrefix = "x-"
	AuthzPrefix     = "x-authz-"
	ForwardedPrefix = "x-forwarded-"
)

// Rate-limit headers on the Router's 429 response.
const (
	RateLimitLimit     = "x-ratelimit-limit"
	RateLimitRemaining = "x-ratelimit-remaining"
	RateLimitReset     = "x-ratelimit-reset"
)

// Proxy and HTTP security headers.
const (
	XForwardedFor       = "x-forwarded-for"
	XForwardedProto     = "x-forwarded-proto"
	XContentTypeOptions = "x-content-type-options"
)

// The Hugging Face Hub's answer to a file download names the commit the
// requested revision resolved to.
const HubRepoCommit = "x-repo-commit"

// Provider credential headers. A provider profile or the model catalog names
// the one a provider reads.
const (
	Authorization          = "authorization"
	APIKey                 = "api-key"        // #nosec G101 -- header name, not a credential
	XAPIKey                = "x-api-key"      // #nosec G101 -- header name, not a credential
	XGoogAPIKey            = "x-goog-api-key" // #nosec G101 -- header name, not a credential
	APISubscriptionKey     = "api-subscription-key"
	OcpApimSubscriptionKey = "ocp-apim-subscription-key"
)

// Envoy's request controls. Envoy's router obeys them from a caller it
// trusts, so the edge drops them from client requests; the Router sets the
// reliability ones from a decision's reliability block.
const (
	EnvoyInternal                     = "x-envoy-internal"
	EnvoyRetriableStatusCodes         = "x-envoy-retriable-status-codes"
	EnvoyRetriableHeaderNames         = "x-envoy-retriable-header-names"
	EnvoyRetryOn                      = "x-envoy-retry-on"
	EnvoyRetryGRPCOn                  = "x-envoy-retry-grpc-on"
	EnvoyMaxRetries                   = "x-envoy-max-retries"
	EnvoyUpstreamAltStatName          = "x-envoy-upstream-alt-stat-name"
	EnvoyUpstreamRqTimeoutMs          = "x-envoy-upstream-rq-timeout-ms"
	EnvoyUpstreamRqPerTryTimeoutMs    = "x-envoy-upstream-rq-per-try-timeout-ms"
	EnvoyUpstreamRqTimeoutAltResponse = "x-envoy-upstream-rq-timeout-alt-response"
	EnvoyExpectedRqTimeoutMs          = "x-envoy-expected-rq-timeout-ms"
	EnvoyForceTrace                   = "x-envoy-force-trace"
	EnvoyIPTags                       = "x-envoy-ip-tags"
	EnvoyOriginalURL                  = "x-envoy-original-url"
	EnvoyHedgeOnPerTryTimeout         = "x-envoy-hedge-on-per-try-timeout"
)

// Envoy's response headers.
const (
	EnvoyUpstreamServiceTime = "x-envoy-upstream-service-time"
	EnvoyOverloaded          = "x-envoy-overloaded"
	EnvoyRateLimited         = "x-envoy-ratelimited"
)
