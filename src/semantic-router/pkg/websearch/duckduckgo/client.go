// Package duckduckgo searches the web through DuckDuckGo's HTML endpoint.
//
// DuckDuckGo publishes no official search API. The HTML endpoint is scraped
// with browser-like headers, a bounded number of concurrent requests and a
// random delay between them, so it is suitable for evaluation rather than
// production traffic. Some network egresses receive a challenge page instead
// of results; that surfaces as a *StatusError with a non-200 status.
package duckduckgo

import (
	"context"
	"errors"
	"fmt"
	"io"
	"math/rand"
	"net"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/websearch"
)

const (
	// DefaultEndpoint is DuckDuckGo's HTML search page.
	DefaultEndpoint = "https://html.duckduckgo.com/html/"
	// DefaultNumResults is used when the caller asks for zero or fewer results.
	DefaultNumResults = 5
	// MaxResults caps how many results one search returns.
	MaxResults = 10

	defaultHTTPTimeout     = 15 * time.Second
	defaultMaxRetries      = 3
	defaultRetryBaseDelay  = time.Second
	defaultMaxConcurrent   = 2
	defaultMinRequestDelay = time.Second
	defaultMaxRequestDelay = 3 * time.Second
)

var (
	// ErrTimeout reports that the upstream request timed out.
	ErrTimeout = errors.New("duckduckgo: upstream request timed out")
	// ErrRequestFailed reports a transport failure other than a timeout.
	ErrRequestFailed = errors.New("duckduckgo: upstream request failed")
	// ErrTooManyConcurrentRequests reports that every outbound slot is busy.
	ErrTooManyConcurrentRequests = errors.New("duckduckgo: too many concurrent requests")
	// ErrBudgetExhausted reports that the caller's upstream budget refused
	// the search before any request was made.
	ErrBudgetExhausted = errors.New("duckduckgo: upstream budget exhausted")
)

// StatusError reports a non-200 answer from DuckDuckGo. A 202 carries a
// challenge page for restricted egress addresses; 4xx answers are not retried.
type StatusError struct {
	Status int
}

func (e *StatusError) Error() string {
	return fmt.Sprintf("duckduckgo: unexpected status %d", e.Status)
}

// Result is one search hit. The JSON names are the wire names consumers rely
// on; keep them stable.
type Result struct {
	Title   string `json:"title"`
	URL     string `json:"url"`
	Snippet string `json:"snippet"`
	Domain  string `json:"domain"`
}

// Option configures a Client.
type Option func(*Client)

// Client searches DuckDuckGo. Every limit is instance-scoped; nothing is
// shared between clients and no background goroutine is started.
type Client struct {
	endpoint       string
	httpClient     *http.Client
	maxRetries     int
	retryBaseDelay time.Duration
	minDelay       time.Duration
	maxDelay       time.Duration
	userAgents     []string
	logf           func(format string, args ...any)
	budget         func() bool
	maxConcurrent  int
	slots          chan struct{}
}

// WithBudget charges one upstream reservation per Search, before the request
// delay and the outbound call. A refusal returns ErrBudgetExhausted without
// contacting DuckDuckGo; retries within one Search share the reservation.
// Callers that must keep DuckDuckGo from banning a shared egress address
// supply the budget (the Dashboard passes its process-wide limiter).
func WithBudget(reserve func() bool) Option {
	return func(c *Client) { c.budget = reserve }
}

// WithEndpoint points the client at another HTML search page, such as a
// test server.
func WithEndpoint(endpoint string) Option {
	return func(c *Client) { c.endpoint = endpoint }
}

// WithHTTPClient replaces the pooled default HTTP client.
func WithHTTPClient(httpClient *http.Client) Option {
	return func(c *Client) { c.httpClient = httpClient }
}

// WithMaxRetries sets how many attempts one search makes. Values below one
// mean a single attempt.
func WithMaxRetries(attempts int) Option {
	return func(c *Client) { c.maxRetries = attempts }
}

// WithRetryBaseDelay sets the first backoff delay; later attempts double it.
func WithRetryBaseDelay(delay time.Duration) Option {
	return func(c *Client) { c.retryBaseDelay = delay }
}

// WithRequestDelay sets the random pause taken before every upstream request.
func WithRequestDelay(minDelay, maxDelay time.Duration) Option {
	return func(c *Client) {
		c.minDelay = minDelay
		c.maxDelay = maxDelay
	}
}

// WithMaxConcurrent bounds the outbound requests in flight at once.
func WithMaxConcurrent(limit int) Option {
	return func(c *Client) { c.maxConcurrent = limit }
}

// WithUserAgents replaces the browser user agents the client rotates through.
func WithUserAgents(agents []string) Option {
	return func(c *Client) {
		if len(agents) > 0 {
			c.userAgents = append([]string(nil), agents...)
		}
	}
}

// WithLogf receives content-free diagnostics such as upstream status codes.
func WithLogf(logf func(format string, args ...any)) Option {
	return func(c *Client) {
		if logf != nil {
			c.logf = logf
		}
	}
}

// NewClient builds a client with DuckDuckGo-friendly defaults: a 15 second
// timeout, three attempts, two concurrent requests and a one to three second
// pause before each request.
func NewClient(opts ...Option) *Client {
	client := &Client{
		endpoint:       DefaultEndpoint,
		maxRetries:     defaultMaxRetries,
		retryBaseDelay: defaultRetryBaseDelay,
		minDelay:       defaultMinRequestDelay,
		maxDelay:       defaultMaxRequestDelay,
		userAgents:     websearch.DefaultUserAgents(),
		logf:           func(string, ...any) {},
		maxConcurrent:  defaultMaxConcurrent,
	}
	for _, opt := range opts {
		opt(client)
	}
	if client.httpClient == nil {
		client.httpClient = newPooledHTTPClient()
	}
	if client.maxRetries < 1 {
		client.maxRetries = 1
	}
	if client.maxConcurrent < 1 {
		client.maxConcurrent = 1
	}
	client.slots = make(chan struct{}, client.maxConcurrent)
	return client
}

func newPooledHTTPClient() *http.Client {
	return &http.Client{
		Timeout: defaultHTTPTimeout,
		Transport: &http.Transport{
			MaxIdleConns:        100,
			MaxIdleConnsPerHost: 10,
			IdleConnTimeout:     90 * time.Second,
		},
	}
}

// Search returns up to numResults hits for query. Zero or negative asks for
// DefaultNumResults; anything above MaxResults is clamped. Transient failures
// are retried with exponential backoff; 4xx answers are not. A configured
// budget is charged once, before the first attempt.
func (c *Client) Search(ctx context.Context, query string, numResults int) ([]Result, error) {
	if numResults <= 0 {
		numResults = DefaultNumResults
	}
	if numResults > MaxResults {
		numResults = MaxResults
	}
	if c.budget != nil && !c.budget() {
		return nil, ErrBudgetExhausted
	}

	var lastErr error
	for attempt := 0; attempt < c.maxRetries; attempt++ {
		results, err := c.doRequest(ctx, query, numResults)
		if err == nil {
			return results, nil
		}
		lastErr = err
		if isClientError(err) || ctx.Err() != nil {
			break
		}
		if attempt < c.maxRetries-1 {
			if err := sleepWithContext(ctx, c.retryBaseDelay*time.Duration(1<<attempt)); err != nil {
				return nil, err
			}
			c.logf("Retrying search (attempt %d/%d, error_class=%T)", attempt+2, c.maxRetries, lastErr)
		}
	}
	return nil, lastErr
}

func isClientError(err error) bool {
	var statusErr *StatusError
	return errors.As(err, &statusErr) && statusErr.Status >= 400 && statusErr.Status < 500
}

func (c *Client) doRequest(ctx context.Context, query string, numResults int) ([]Result, error) {
	select {
	case c.slots <- struct{}{}:
		defer func() { <-c.slots }()
	default:
		return nil, ErrTooManyConcurrentRequests
	}

	if err := sleepWithContext(ctx, c.randomDelay()); err != nil {
		return nil, err
	}

	target, err := searchURL(c.endpoint, query)
	if err != nil {
		return nil, fmt.Errorf("%w: %w", ErrRequestFailed, err)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, target, nil)
	if err != nil {
		return nil, fmt.Errorf("%w: %w", ErrRequestFailed, err)
	}
	// Accept-Encoding is left to the transport so it transparently
	// decompresses the answer.
	req.Header.Set("User-Agent", c.userAgents[rand.Intn(len(c.userAgents))])
	req.Header.Set("Accept", "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8")
	req.Header.Set("Accept-Language", "en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7")
	req.Header.Set("DNT", "1")
	req.Header.Set("Connection", "keep-alive")
	req.Header.Set("Upgrade-Insecure-Requests", "1")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		c.logTransportFailure(err)
		return nil, classifyTransportError(ctx, err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		return nil, &StatusError{Status: resp.StatusCode}
	}

	// A body read error names the connection, never the request URL.
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("%w: reading response: %w", ErrRequestFailed, err)
	}

	// Keep diagnostics content-free: the page carries the query and snippets.
	c.logf("DuckDuckGo response: status=%d, length=%d", resp.StatusCode, len(body))
	page := string(body)
	if strings.Contains(page, "captcha") || strings.Contains(page, "robot") || strings.Contains(page, "blocked") {
		c.logf("WARNING: DuckDuckGo may be blocking requests (captcha/robot detected)")
	}

	return parseResults(page, numResults, c.logf), nil
}

// searchURL adds the query to the endpoint, keeping any parameters the
// endpoint already carries (a region pin such as kl=us-en, for example).
func searchURL(endpoint, query string) (string, error) {
	target, err := url.Parse(endpoint)
	if err != nil {
		return "", err
	}
	params := target.Query()
	params.Set("q", query)
	target.RawQuery = params.Encode()
	return target.String(), nil
}

// logTransportFailure records why the transport failed. *url.Error quotes the
// request URL, query included, so only its inner error is logged.
func (c *Client) logTransportFailure(err error) {
	var urlErr *url.Error
	if errors.As(err, &urlErr) {
		c.logf("DuckDuckGo request failed: %v", urlErr.Err)
		return
	}
	c.logf("DuckDuckGo request failed: error_class=%T", err)
}

// classifyTransportError keeps the caller's query out of the returned error:
// net/http errors quote the full request URL.
func classifyTransportError(ctx context.Context, err error) error {
	if ctx.Err() != nil {
		return ctx.Err()
	}
	var netErr net.Error
	if errors.As(err, &netErr) && netErr.Timeout() {
		return ErrTimeout
	}
	return ErrRequestFailed
}

func (c *Client) randomDelay() time.Duration {
	if c.maxDelay <= c.minDelay {
		return c.minDelay
	}
	return c.minDelay + time.Duration(rand.Int63n(int64(c.maxDelay-c.minDelay)))
}

func sleepWithContext(ctx context.Context, delay time.Duration) error {
	if delay <= 0 {
		return ctx.Err()
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
