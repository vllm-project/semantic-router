package config

// OnErrorAllow and OnErrorBlock are the values the removed prompt_guard
// module's on_error and on_unscanned took; migrating its response_jailbreak
// plugin reads them.
const (
	OnErrorAllow = "allow"
	OnErrorBlock = "block"
)
