package handlers

import (
	"errors"
	"log"
	"net/url"
	"os"
	"strconv"
	"strings"
)

const (
	publicBaseURLEnv     = "VLLM_SR_PUBLIC_BASE_URL"
	publicRequiresKeyEnv = "VLLM_SR_PUBLIC_REQUIRES_KEY"
)

// publicEndpoint is the gateway address a deployment declares for clients,
// for a Router behind a proxy or edge the Dashboard cannot see from its
// listeners.
type publicEndpoint struct {
	baseURL     string
	requiresKey bool
}

// publicEndpointFromEnv ignores an invalid value with a warning rather than
// stopping the Dashboard over it. The key requirement describes the public
// address, so it is dropped without one.
func publicEndpointFromEnv() publicEndpoint {
	var endpoint publicEndpoint
	raw := strings.TrimSpace(os.Getenv(publicBaseURLEnv))
	if raw == "" {
		return endpoint
	}
	baseURL, err := normalizePublicBaseURL(raw)
	if err != nil {
		log.Printf("WARNING: ignoring %s: %v", publicBaseURLEnv, err)
		return endpoint
	}
	endpoint.baseURL = baseURL
	if raw := strings.TrimSpace(os.Getenv(publicRequiresKeyEnv)); raw != "" {
		requiresKey, parseErr := strconv.ParseBool(raw)
		if parseErr != nil {
			log.Printf("WARNING: ignoring %s: it must be true or false", publicRequiresKeyEnv)
		} else {
			endpoint.requiresKey = requiresKey
		}
	}
	return endpoint
}

// normalizePublicBaseURL accepts an http or https URL without credentials,
// query or fragment, and drops a trailing slash. Its errors never quote the
// value, which may carry credentials.
func normalizePublicBaseURL(raw string) (string, error) {
	parsed, err := url.Parse(raw)
	if err != nil {
		return "", errors.New("it is not a URL")
	}
	if parsed.Scheme != "http" && parsed.Scheme != "https" {
		return "", errors.New("it must be an http or https URL")
	}
	if parsed.Hostname() == "" {
		return "", errors.New("it has no host")
	}
	if parsed.User != nil {
		return "", errors.New("it must not carry credentials")
	}
	if parsed.RawQuery != "" || parsed.ForceQuery || parsed.Fragment != "" {
		return "", errors.New("it must not have a query or fragment")
	}
	parsed.Path = strings.TrimRight(parsed.Path, "/")
	parsed.RawPath = ""
	return parsed.String(), nil
}
