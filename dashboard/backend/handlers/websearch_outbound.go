package handlers

import (
	"context"
	"errors"
	"log"
	"math/rand"
	"net/http"
	"net/url"
	"strings"
	"time"
)

const (
	webSearchHTTPTimeout        = 15 * time.Second
	webSearchMaxResponseSize    = 2 * 1024 * 1024
	webSearchMaxRetries         = 3
	webSearchRetryBaseDelay     = 1 * time.Second
	webSearchMaxConcurrent      = 2
	webSearchMinRequestInterval = 1 * time.Second
	webSearchMaxRequestInterval = 3 * time.Second
	duckDuckGoSearchURL         = "https://html.duckduckgo.com/html/"
)

type webSearchOutbound struct {
	client      outboundHTTPClient
	concurrent  chan struct{}
	randomDelay func(context.Context) error
}

type webSearchAttemptError struct {
	code      SearchErrorCode
	retryable bool
}

func (e *webSearchAttemptError) Error() string {
	return string(e.code)
}

func newWebSearchOutbound(client outboundHTTPClient) *webSearchOutbound {
	return &webSearchOutbound{
		client:     client,
		concurrent: make(chan struct{}, webSearchMaxConcurrent),
		randomDelay: func(ctx context.Context) error {
			return waitWebSearchDelay(ctx, webSearchMinRequestInterval, webSearchMaxRequestInterval)
		},
	}
}

func (s *webSearchOutbound) searchDuckDuckGo(
	ctx context.Context,
	query string,
	numResults int,
) ([]SearchResult, error) {
	if numResults <= 0 {
		numResults = defaultNumResults
	}
	if numResults > maxResultsLimit {
		numResults = maxResultsLimit
	}

	var lastErr error
	for attempt := 0; attempt < webSearchMaxRetries; attempt++ {
		results, err := s.doSearchRequest(ctx, query, numResults)
		if err == nil {
			return results, nil
		}
		lastErr = err

		var attemptErr *webSearchAttemptError
		if !errors.As(err, &attemptErr) || !attemptErr.retryable || attempt == webSearchMaxRetries-1 {
			break
		}
		delay := webSearchRetryBaseDelay * time.Duration(1<<attempt)
		if waitErr := waitWebSearchDelay(ctx, delay, delay); waitErr != nil {
			return nil, newWebSearchError(ErrCodeTimeout, false)
		}
		log.Printf("[WebSearch] Retrying upstream: attempt=%d/%d", attempt+2, webSearchMaxRetries)
	}
	return nil, lastErr
}

func (s *webSearchOutbound) doSearchRequest(
	ctx context.Context,
	query string,
	numResults int,
) ([]SearchResult, error) {
	select {
	case <-ctx.Done():
		return nil, newWebSearchError(ErrCodeTimeout, false)
	case s.concurrent <- struct{}{}:
		defer func() { <-s.concurrent }()
	default:
		return nil, newWebSearchError(ErrCodeRateLimited, false)
	}

	if err := s.randomDelay(ctx); err != nil {
		return nil, newWebSearchError(ErrCodeTimeout, false)
	}

	searchURL, err := url.Parse(duckDuckGoSearchURL)
	if err != nil {
		return nil, newWebSearchError(ErrCodeSearchFailed, false)
	}
	queryValues := searchURL.Query()
	queryValues.Set("q", query)
	searchURL.RawQuery = queryValues.Encode()
	if validationErr := s.client.ValidateURL(ctx, searchURL.String()); validationErr != nil {
		return nil, newWebSearchError(ErrCodeSearchFailed, false)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, searchURL.String(), nil)
	if err != nil {
		return nil, newWebSearchError(ErrCodeSearchFailed, false)
	}
	req.Header.Set("User-Agent", getRandomUserAgent())
	req.Header.Set("Accept", "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8")
	req.Header.Set("Accept-Language", "en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7")
	req.Header.Set("DNT", "1")
	req.Header.Set("Connection", "keep-alive")
	req.Header.Set("Upgrade-Insecure-Requests", "1")

	resp, err := s.client.Do(req)
	if err != nil {
		if errors.Is(err, errOutboundRequestTimeout) || errors.Is(ctx.Err(), context.DeadlineExceeded) {
			return nil, newWebSearchError(ErrCodeTimeout, true)
		}
		return nil, newWebSearchError(ErrCodeSearchFailed, true)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		retryable := resp.StatusCode == http.StatusRequestTimeout ||
			resp.StatusCode == http.StatusTooManyRequests ||
			resp.StatusCode >= http.StatusInternalServerError
		return nil, newWebSearchError(ErrCodeUpstreamError, retryable)
	}

	body, err := readBoundedOutboundBody(resp.Body, webSearchMaxResponseSize)
	if err != nil {
		return nil, newWebSearchError(ErrCodeUpstreamError, false)
	}
	bodyText := string(body)
	log.Printf("[WebSearch] Upstream response: status=%d bytes=%d", resp.StatusCode, len(body))
	if strings.Contains(bodyText, "captcha") || strings.Contains(bodyText, "robot") || strings.Contains(bodyText, "blocked") {
		log.Printf("[WebSearch] Upstream anti-automation response detected")
	}
	return parseDuckDuckGoHTML(bodyText, numResults)
}

func waitWebSearchDelay(ctx context.Context, minDelay time.Duration, maxDelay time.Duration) error {
	if minDelay < 0 || maxDelay < minDelay {
		return context.Canceled
	}
	delay := minDelay
	if maxDelay > minDelay {
		delay += time.Duration(rand.Int63n(int64(maxDelay-minDelay) + 1))
	}
	timer := time.NewTimer(delay)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-timer.C:
		return nil
	}
}

func newWebSearchError(code SearchErrorCode, retryable bool) error {
	return &webSearchAttemptError{code: code, retryable: retryable}
}

func webSearchErrorCode(err error) SearchErrorCode {
	var searchErr *webSearchAttemptError
	if errors.As(err, &searchErr) && searchErr.code != "" {
		return searchErr.code
	}
	return ErrCodeSearchFailed
}
