//go:build !windows && cgo && (amd64 || arm64)
// +build !windows
// +build cgo
// +build amd64 arm64

package candle_binding

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	urlpkg "net/url"
	"time"
)

// MultiModalEncodeImageFromURL downloads a bounded JPEG/PNG from a public HTTPS
// destination and encodes it. Redirects, URL credentials, ambient proxies, and
// private/reserved destinations are rejected before response data is consumed.
func MultiModalEncodeImageFromURL(rawURL string, targetDim int) (*MultiModalEmbeddingOutput, error) {
	if err := validateNonNegativeCInt("target dimension", targetDim); err != nil {
		return nil, fmt.Errorf("%w: %v", ErrInvalidImageInput, err)
	}
	parsed, err := urlpkg.Parse(rawURL)
	if err != nil || parsed.Scheme != "https" || parsed.Host == "" || parsed.User != nil {
		return nil, fmt.Errorf("%w: a credential-free https URL with a host is required", ErrInvalidImageInput)
	}

	client := newCandlePublicImageClient()
	resp, err := client.Get(parsed.String())
	if err != nil {
		return nil, fmt.Errorf("%w: image download rejected: %v", ErrInvalidImageInput, sanitizedImageRequestError(err))
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("image download returned status %d", resp.StatusCode)
	}
	if resp.ContentLength > MaxMultiModalImageEncodedBytes {
		return nil, fmt.Errorf("%w: remote image exceeds %d bytes", ErrInvalidImageInput, MaxMultiModalImageEncodedBytes)
	}

	data, err := io.ReadAll(io.LimitReader(resp.Body, MaxMultiModalImageEncodedBytes+1))
	if err != nil {
		return nil, fmt.Errorf("read image response: %w", err)
	}
	if len(data) > MaxMultiModalImageEncodedBytes {
		return nil, fmt.Errorf("%w: remote image exceeds %d bytes", ErrInvalidImageInput, MaxMultiModalImageEncodedBytes)
	}
	return MultiModalEncodeImageFromBytes(data, targetDim)
}

func sanitizedImageRequestError(err error) error {
	var requestError *urlpkg.Error
	if errors.As(err, &requestError) {
		return requestError.Err
	}
	return err
}

func newCandlePublicImageClient() *http.Client {
	return &http.Client{
		Timeout:   30 * time.Second,
		Transport: candlePublicImageTransport(),
		CheckRedirect: func(_ *http.Request, _ []*http.Request) error {
			return http.ErrUseLastResponse
		},
	}
}

func candlePublicImageTransport() *http.Transport {
	dialer := &net.Dialer{Timeout: 10 * time.Second, KeepAlive: 30 * time.Second}
	return &http.Transport{
		Proxy:             nil,
		DisableKeepAlives: true,
		DialContext: func(ctx context.Context, network, address string) (net.Conn, error) {
			host, port, err := net.SplitHostPort(address)
			if err != nil {
				return nil, fmt.Errorf("split image destination: %w", err)
			}
			addresses, err := net.DefaultResolver.LookupNetIP(ctx, "ip", host)
			if err != nil {
				return nil, fmt.Errorf("resolve image destination: %w", err)
			}
			if len(addresses) == 0 {
				return nil, errors.New("image destination resolved to no addresses")
			}
			if !candleImageDestinationsArePublic(ctx, addresses, candleImagePref64Cache) {
				return nil, errors.New("image destination could not be verified as public")
			}
			var dialErr error
			for _, resolved := range addresses {
				conn, err := dialer.DialContext(ctx, network, net.JoinHostPort(resolved.String(), port))
				if err == nil {
					return conn, nil
				}
				dialErr = err
			}
			return nil, fmt.Errorf("connect image destination: %w", dialErr)
		},
		ForceAttemptHTTP2: true,
	}
}
