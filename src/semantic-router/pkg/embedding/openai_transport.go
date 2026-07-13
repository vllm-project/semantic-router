package embedding

import "net/http"

var fallbackEmbeddingTransport = http.DefaultTransport.(*http.Transport).Clone()

// embeddingTransportError keeps the underlying error available for timeout
// and retry classification while preventing net/http's URL-bearing error text
// from reaching logs or API responses.
type embeddingTransportError struct {
	cause error
}

func (e *embeddingTransportError) Error() string {
	return "embedding provider transport request failed"
}

func (e *embeddingTransportError) Unwrap() error {
	return e.cause
}

func redactEmbeddingTransportError(err error) error {
	if err == nil {
		return nil
	}
	return &embeddingTransportError{cause: err}
}

// newNoProxyEmbeddingTransport clones the effective default transport but
// deliberately ignores ambient HTTP(S)_PROXY settings. Provider credentials
// should only transit a proxy when the caller explicitly supplies a custom
// HTTPClient transport.
func newNoProxyEmbeddingTransport() http.RoundTripper {
	base, ok := http.DefaultTransport.(*http.Transport)
	if !ok {
		base = fallbackEmbeddingTransport
	}
	transport := base.Clone()
	transport.Proxy = nil
	return transport
}
