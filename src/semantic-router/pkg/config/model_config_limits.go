package config

// DefaultClassifyMaxRequestBytes bounds the text one http_classify call may
// send. The classifier truncates to its own context window anyway.
const DefaultClassifyMaxRequestBytes int64 = 256 * 1024

// DefaultClassifyMaxResponseBytes bounds one http_classify response. The wire
// contract is a score per label, so a well-behaved endpoint returns kilobytes.
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
