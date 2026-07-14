package proxy

import (
	"errors"
	"net/url"
	"strings"
)

func parseProxyTarget(rawURL string) (*url.URL, error) {
	target, err := url.Parse(rawURL)
	if err != nil {
		// url.Parse errors can contain the original URL. Keep constructor errors
		// credential-free because callers commonly write them to service logs.
		return nil, errors.New("invalid proxy target URL")
	}
	target.Scheme = strings.ToLower(target.Scheme)
	if err := validateProxyTarget(target); err != nil {
		return nil, err
	}
	return target, nil
}

func validateProxyTarget(target *url.URL) error {
	if target == nil || target.Hostname() == "" {
		return errors.New("proxy target URL must include a host")
	}
	scheme := strings.ToLower(target.Scheme)
	if scheme != "http" && scheme != "https" {
		return errors.New("proxy target URL must use http or https")
	}
	if target.User != nil {
		return errors.New("proxy target URL must not contain user information")
	}
	return nil
}
