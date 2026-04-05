package handlers

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"net/url"
	"time"
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

	result, fetchErr := fetchOpenWeb(plan)
	if fetchErr != nil {
		log.Printf("[OpenWeb] ❌ All fetch methods failed: %v", fetchErr)
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

	parsedURL, err := url.Parse(req.URL)
	if err != nil || (parsedURL.Scheme != "http" && parsedURL.Scheme != "https") {
		return OpenWebResponse{
			URL:   req.URL,
			Error: "Invalid URL format",
		}, true
	}

	return OpenWebResponse{}, false
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
		plan.request.URL,
		plan.timeout,
		plan.forceJina,
		plan.format,
		plan.maxLength,
		plan.request.WithImages,
	)
}

func fetchOpenWeb(plan openWebFetchPlan) (*OpenWebResponse, error) {
	if !plan.forceJina {
		log.Printf("[OpenWeb] Strategy 1: Trying direct fetch...")
		result, err := fetchWebDirect(plan.request.URL, plan.timeout, plan.maxLength)
		if err == nil {
			log.Printf("[OpenWeb] ✅ Direct fetch succeeded")
			return result, nil
		}
		log.Printf("[OpenWeb] ⚠️ Direct fetch failed: %v", err)
		log.Printf("[OpenWeb] Strategy 2: Falling back to Jina Reader...")
	} else {
		log.Printf("[OpenWeb] Skipping direct fetch, using Jina Reader directly")
	}

	result, err := fetchWebWithJina(
		plan.request.URL,
		plan.timeout,
		plan.format,
		plan.maxLength,
		plan.request.WithImages,
	)
	if err != nil {
		return nil, err
	}

	log.Printf("[OpenWeb] ✅ Jina Reader fetch succeeded")
	return result, nil
}

func writeOpenWebJSON(w http.ResponseWriter, status int, response OpenWebResponse) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(response)
}
