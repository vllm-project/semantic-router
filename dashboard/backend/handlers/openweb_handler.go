package handlers

import (
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"net/http"
	"net/netip"
	"strings"
	"time"

	dashboardauth "github.com/vllm-project/semantic-router/dashboard/backend/auth"
	"github.com/vllm-project/semantic-router/dashboard/backend/safefetch"
)

type openWebFetchPlan struct {
	request   OpenWebRequest
	timeout   time.Duration
	format    string
	maxLength int
	forceJina bool
}

func OpenWebHandler() http.HandlerFunc {
	return handleOpenWeb
}

func handleOpenWeb(w http.ResponseWriter, r *http.Request) {
	setOpenWebCORSHeaders(w)

	if r.Method == http.MethodOptions {
		w.WriteHeader(http.StatusOK)
		return
	}
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	req, err := decodeOpenWebRequest(r)
	if err != nil {
		log.Printf("[OpenWeb] Failed to parse request: %v", err)
		writeOpenWebJSON(w, http.StatusBadRequest, OpenWebResponse{Error: "Invalid request format"})
		return
	}

	if invalidResponse, ok := validateOpenWebRequest(req); ok {
		writeOpenWebJSON(w, http.StatusBadRequest, invalidResponse)
		return
	}

	plan := buildOpenWebFetchPlan(req)
	logOpenWebFetchPlan(plan)

	result, revoked, fetchErr := fetchOpenWeb(plan, func() bool {
		return dashboardauth.RejectRevokedMutation(w, r)
	})
	if revoked {
		return
	}
	if fetchErr != nil {
		log.Printf(
			"[OpenWeb] All fetch methods failed for %s: %v",
			redactURLForLog(req.URL),
			redactURLsForLog(fetchErr.Error()),
		)
		writeOpenWebJSON(w, http.StatusBadGateway, OpenWebResponse{
			URL:   req.URL,
			Error: fmt.Sprintf("Unable to fetch web content: %v", fetchErr),
		})
		return
	}

	writeOpenWebJSON(w, http.StatusOK, *result)
}

func setOpenWebCORSHeaders(w http.ResponseWriter) {
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "POST, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type")
}

func decodeOpenWebRequest(r *http.Request) (OpenWebRequest, error) {
	var req OpenWebRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		return OpenWebRequest{}, err
	}
	return req, nil
}

func validateOpenWebRequest(req OpenWebRequest) (OpenWebResponse, bool) {
	if req.URL == "" {
		return OpenWebResponse{Error: "URL cannot be empty"}, true
	}

	parsed, err := outboundPolicy(openWebDefaultTimeout).ValidateURL(req.URL)
	if err != nil {
		return OpenWebResponse{
			URL:   req.URL,
			Error: "Invalid URL format",
		}, true
	}
	// Reader mode forwards this URL to an external service, so our transport
	// cannot inspect the target's address. Refuse literal non-public targets
	// before choosing either the direct or reader path.
	if isNonPublicReaderTarget(parsed.Hostname()) {
		return OpenWebResponse{
			URL:   req.URL,
			Error: errOpenWebForbiddenTarget.Error(),
		}, true
	}

	return OpenWebResponse{}, false
}

func isNonPublicReaderTarget(host string) bool {
	if address, err := netip.ParseAddr(host); err == nil {
		return !safefetch.IsPublicAddr(address)
	}

	// Other URL consumers can accept shorthand, octal or hexadecimal IPv4
	// forms. Reject numeric-looking hosts rather than trusting Jina to parse
	// them the same way as Go. Local names are not public web targets either.
	host = strings.TrimSuffix(strings.ToLower(host), ".")
	if host == "localhost" || strings.HasSuffix(host, ".localhost") ||
		host == "local" || strings.HasSuffix(host, ".local") ||
		host == "internal" || strings.HasSuffix(host, ".internal") {
		return true
	}
	if host == "" {
		return true
	}
	labels := strings.Split(host, ".")
	for _, label := range labels {
		if label == "" {
			return true
		}
	}
	for _, label := range labels {
		base := "0123456789"
		if strings.HasPrefix(label, "0x") {
			label = strings.TrimPrefix(label, "0x")
			base = "0123456789abcdef"
		}
		if label == "" || strings.IndexFunc(label, func(r rune) bool { return !strings.ContainsRune(base, r) }) >= 0 {
			return false
		}
	}
	return true
}

func buildOpenWebFetchPlan(req OpenWebRequest) openWebFetchPlan {
	timeout := openWebDefaultTimeout
	if req.Timeout > 0 {
		timeout = time.Duration(req.Timeout) * time.Second
		if timeout > openWebMaxTimeout {
			timeout = openWebMaxTimeout
		}
	}

	return openWebFetchPlan{
		request:   req,
		timeout:   timeout,
		format:    normalizeOpenWebFormat(req.Format),
		maxLength: normalizeOpenWebMaxLength(req.MaxLength),
		forceJina: shouldPreferJinaFetch(req.URL, req),
	}
}

func logOpenWebFetchPlan(plan openWebFetchPlan) {
	log.Printf(
		"[OpenWeb] Request: url=%s, timeout=%v, force_jina=%v, format=%s, max_length=%d, with_images=%v",
		redactURLForLog(plan.request.URL),
		plan.timeout,
		plan.forceJina,
		plan.format,
		plan.maxLength,
		plan.request.WithImages,
	)
}

func fetchOpenWeb(plan openWebFetchPlan, rejectRevoked func() bool) (*OpenWebResponse, bool, error) {
	if !plan.forceJina {
		if rejectRevoked() {
			return nil, true, nil
		}
		log.Printf("[OpenWeb] Strategy 1: Trying direct fetch...")
		result, err := fetchWebDirect(plan.request.URL, plan.timeout, plan.maxLength)
		if err == nil {
			if rejectRevoked() {
				return nil, true, nil
			}
			log.Printf("[OpenWeb] Direct fetch succeeded")
			return result, false, nil
		}
		// A refused destination fails closed. Only a transport or upstream
		// failure earns the reader fallback; retrying a blocked URL through a
		// second path would make the policy advisory.
		if isForbiddenFetchTarget(err) {
			log.Printf("[OpenWeb] Direct fetch refused by outbound policy")
			return nil, false, errOpenWebForbiddenTarget
		}
		log.Printf("[OpenWeb] Direct fetch failed: %v", redactURLsForLog(err.Error()))
		log.Printf("[OpenWeb] Strategy 2: Falling back to Jina Reader...")
	} else {
		log.Printf("[OpenWeb] Skipping direct fetch, using Jina Reader directly")
	}

	if rejectRevoked() {
		return nil, true, nil
	}
	result, err := fetchWebWithJina(
		plan.request.URL,
		plan.timeout,
		plan.format,
		plan.maxLength,
		plan.request.WithImages,
	)
	if err != nil {
		return nil, false, err
	}
	if rejectRevoked() {
		return nil, true, nil
	}

	log.Printf("[OpenWeb] Jina Reader fetch succeeded")
	return result, false, nil
}

// errOpenWebForbiddenTarget is what the caller sees when the outbound policy
// refuses a destination. It names no address and no reason, so the endpoint
// cannot be used to map the dashboard's network by reading error text.
var errOpenWebForbiddenTarget = errors.New("destination is not permitted")

// isForbiddenFetchTarget reports whether err is the outbound policy refusing
// the destination, as opposed to the upstream being unreachable or slow.
func isForbiddenFetchTarget(err error) bool {
	return errors.Is(err, errOpenWebForbiddenTarget) ||
		errors.Is(err, safefetch.ErrDestinationForbidden) ||
		errors.Is(err, safefetch.ErrSchemeNotAllowed) ||
		errors.Is(err, safefetch.ErrInvalidURL) ||
		errors.Is(err, safefetch.ErrTooManyRedirects)
}

func writeOpenWebJSON(w http.ResponseWriter, status int, response OpenWebResponse) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(response)
}
