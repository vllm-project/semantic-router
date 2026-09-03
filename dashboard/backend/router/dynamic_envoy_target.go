package router

import (
	"errors"
	"io"
	"net"
	"net/url"
	"os"
	"strconv"
	"strings"

	"github.com/vllm-project/semantic-router/dashboard/backend/routercontract"
)

const maxDynamicEnvoyConfigBytes = 16 << 20

// resolveDynamicEnvoyTarget keeps Dashboard's stable service hostname while
// following the active runtime's primary listener port. This prevents
// Playground/chat from retaining the serve-time port after a topology switch.
func resolveDynamicEnvoyTarget(configuredURL, configPath string) (*url.URL, error) {
	target, err := parseDynamicEnvoyTarget(configuredURL)
	if err != nil {
		return nil, err
	}
	data, err := readDynamicRuntimeConfig(configPath)
	if err != nil {
		return nil, err
	}
	port, err := dynamicEnvoyListenerPort(target, data)
	if err != nil {
		return nil, err
	}
	clone := *target
	clone.Host = net.JoinHostPort(target.Hostname(), strconv.Itoa(port))
	return &clone, nil
}

func parseDynamicEnvoyTarget(configuredURL string) (*url.URL, error) {
	target, err := url.Parse(strings.TrimSpace(configuredURL))
	if err != nil || target.Host == "" || (target.Scheme != "http" && target.Scheme != "https") {
		return nil, errors.New("envoy target URL is invalid")
	}
	return target, nil
}

func readDynamicRuntimeConfig(configPath string) ([]byte, error) {
	info, err := os.Lstat(configPath)
	if err != nil || info.Mode()&os.ModeSymlink != 0 || !info.Mode().IsRegular() || info.Size() > maxDynamicEnvoyConfigBytes {
		return nil, errors.New("runtime config is unavailable")
	}
	file, err := os.Open(configPath)
	if err != nil {
		return nil, errors.New("runtime config is unavailable")
	}
	defer func() { _ = file.Close() }()
	openedInfo, err := file.Stat()
	if err != nil || !os.SameFile(info, openedInfo) {
		return nil, errors.New("runtime config changed while opening")
	}
	data, err := io.ReadAll(io.LimitReader(file, maxDynamicEnvoyConfigBytes+1))
	if err != nil || len(data) > maxDynamicEnvoyConfigBytes {
		return nil, errors.New("runtime config is unavailable")
	}
	return data, nil
}

func dynamicEnvoyListenerPort(target *url.URL, data []byte) (int, error) {
	listener, ok, err := routercontract.ParseFirstListenerEndpoint(data)
	if err != nil || !ok {
		return 0, errors.New("runtime listener is unavailable")
	}
	port := listener.Port
	if !isManagedEnvoyHostname(target.Hostname()) {
		if raw := strings.TrimSpace(os.Getenv("VLLM_SR_PORT_OFFSET")); raw != "" {
			offset, parseErr := strconv.Atoi(raw)
			if parseErr != nil || offset < 0 || port+offset > 65535 {
				return 0, errors.New("runtime port offset is invalid")
			}
			port += offset
		}
	}
	return port, nil
}

func isManagedEnvoyHostname(host string) bool {
	expected := strings.TrimSpace(os.Getenv("VLLM_SR_ENVOY_CONTAINER_NAME"))
	if expected == "" {
		stack := strings.TrimSpace(os.Getenv("VLLM_SR_STACK_NAME"))
		if stack == "" || stack == "vllm-sr" {
			expected = "vllm-sr-envoy-container"
		} else {
			expected = stack + "-vllm-sr-envoy-container"
		}
	}
	return strings.EqualFold(strings.TrimSpace(host), expected)
}
