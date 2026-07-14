package proxy

import (
	"context"
	"crypto/tls"
	"fmt"
	"net"
	"net/url"
	"strings"
	"time"
)

func validateWebSocketTarget(target *url.URL) error {
	if err := validateProxyTarget(target); err != nil {
		return err
	}
	if target.RawQuery != "" || target.ForceQuery || target.Fragment != "" {
		return fmt.Errorf("WebSocket proxy target must not contain query or fragment components")
	}
	return nil
}

func webSocketTargetAddress(target *url.URL) (string, error) {
	if err := validateWebSocketTarget(target); err != nil {
		return "", err
	}
	port := target.Port()
	if port == "" {
		if strings.EqualFold(target.Scheme, "https") {
			port = "443"
		} else {
			port = "80"
		}
	}
	return net.JoinHostPort(target.Hostname(), port), nil
}

func dialWebSocketTarget(
	ctx context.Context,
	target *url.URL,
	timeout time.Duration,
) (net.Conn, error) {
	return dialWebSocketTargetWithTLSConfig(ctx, target, timeout, nil)
}

func dialWebSocketTargetWithTLSConfig(
	ctx context.Context,
	target *url.URL,
	timeout time.Duration,
	tlsConfig *tls.Config,
) (net.Conn, error) {
	address, err := webSocketTargetAddress(target)
	if err != nil {
		return nil, err
	}
	dialer := &net.Dialer{Timeout: timeout}
	if strings.EqualFold(target.Scheme, "http") {
		return dialer.DialContext(ctx, "tcp", address)
	}

	if tlsConfig == nil {
		tlsConfig = &tls.Config{}
	} else {
		tlsConfig = tlsConfig.Clone()
	}
	if tlsConfig.ServerName == "" {
		tlsConfig.ServerName = target.Hostname()
	}
	if tlsConfig.MinVersion == 0 {
		tlsConfig.MinVersion = tls.VersionTLS12
	}
	tlsDialer := &tls.Dialer{
		NetDialer: dialer,
		Config:    tlsConfig,
	}
	return tlsDialer.DialContext(ctx, "tcp", address)
}
