package config

import "time"

// DefaultClassifyMaxRequestBytes bounds the text one http_classify call may
// send. The classifier truncates to its own context window anyway.
const DefaultClassifyMaxRequestBytes int64 = 256 * 1024

// DefaultClassifierTimeoutSeconds bounds one external classifier call when
// llm_timeout_seconds is unset. Remote chat-style guardrail calls are
// generative, so this shares the 30s majority default the vLLM and preference
// callers already used; http_classify sets its own tighter 5s default.
const DefaultClassifierTimeoutSeconds = 30

// DefaultClassifyMaxResponseBytes bounds one external classifier response.
const DefaultClassifyMaxResponseBytes int64 = 1024 * 1024

// GetMaxRequestBytes returns the per-request send ceiling, defaulting when
// unset or non-positive.
func (e *ExternalModelConfig) GetMaxRequestBytes() int64 {
	if e.MaxRequestBytes <= 0 {
		return DefaultClassifyMaxRequestBytes
	}
	return e.MaxRequestBytes
}

// GetMaxResponseBytes returns the per-response read ceiling, defaulting when
// unset or non-positive.
func (e *ExternalModelConfig) GetMaxResponseBytes() int64 {
	if e.MaxResponseBytes <= 0 {
		return DefaultClassifyMaxResponseBytes
	}
	return e.MaxResponseBytes
}

// GetTimeout returns the per-call timeout, defaulting when
// llm_timeout_seconds is unset or non-positive. Callers should use this
// instead of hardcoding their own default so the http.Client.Timeout and the
// caller's context.WithTimeout agree: when they disagree the stricter one
// wins and a configured llm_timeout_seconds above the hardcoded client
// default is silently ignored.
func (e *ExternalModelConfig) GetTimeout() time.Duration {
	if e.TimeoutSeconds <= 0 {
		return time.Duration(DefaultClassifierTimeoutSeconds) * time.Second
	}
	return time.Duration(e.TimeoutSeconds) * time.Second
}
