package candle_binding

import (
	"context"
	"fmt"
	"regexp"
	"strings"
	"time"
)

// RegexProviderConfig holds the configuration for the regex provider.
type RegexProviderConfig struct {
	MaxPatterns      int            `yaml:"max_patterns"`
	MaxPatternLength int            `yaml:"max_pattern_length"`
	MaxInputLength   int            `yaml:"max_input_length"`
	DefaultTimeoutMs int            `yaml:"default_timeout_ms"`
	Patterns         []RegexPattern `yaml:"patterns"`
}

// RegexPattern defines a single regex pattern.
type RegexPattern struct {
	ID       string `yaml:"id"`
	Pattern  string `yaml:"pattern"`
	Flags    string `yaml:"flags"`
	Category string `yaml:"category"`
}

// RegexProvider is a ReDoS-safe regex scanner.
// It uses Go's built-in regexp package, which is based on RE2 and is not
// vulnerable to regular expression denial of service attacks.
type RegexProvider struct {
	compiled       []*regexp.Regexp
	patterns       []RegexPattern
	timeout        time.Duration
	maxInputLength int
	testDelay      time.Duration // For testing purposes
}

// MatchResult represents a single regex match.
type MatchResult struct {
	PatternID  string
	Category   string
	Match      string
	StartIndex int
	EndIndex   int
}

// NewRegexProvider creates a new RegexProvider.
func NewRegexProvider(cfg RegexProviderConfig, options ...func(*RegexProvider)) (*RegexProvider, error) {
	if len(cfg.Patterns) > cfg.MaxPatterns {
		return nil, fmt.Errorf("number of patterns (%d) exceeds max_patterns (%d)", len(cfg.Patterns), cfg.MaxPatterns)
	}

	compiled := make([]*regexp.Regexp, 0, len(cfg.Patterns))
	for _, p := range cfg.Patterns {
		if len(p.Pattern) > cfg.MaxPatternLength {
			return nil, fmt.Errorf("pattern length for ID '%s' (%d) exceeds max_pattern_length (%d)", p.ID, len(p.Pattern), cfg.MaxPatternLength)
		}

		pattern := p.Pattern
		if strings.Contains(p.Flags, "i") {
			pattern = "(?i)" + pattern
		}

		re, err := regexp.Compile(pattern)
		if err != nil {
			return nil, fmt.Errorf("failed to compile pattern ID '%s': %w", p.ID, err)
		}
		compiled = append(compiled, re)
	}

	rp := &RegexProvider{
		compiled:       compiled,
		patterns:       cfg.Patterns,
		timeout:        time.Duration(cfg.DefaultTimeoutMs) * time.Millisecond,
		maxInputLength: cfg.MaxInputLength,
	}

	for _, option := range options {
		option(rp)
	}

	return rp, nil
}

// WithTestDelay is a functional option to add a delay for testing timeouts.
func WithTestDelay(d time.Duration) func(*RegexProvider) {
	return func(rp *RegexProvider) {
		rp.testDelay = d
	}
}

// Scan scans the input string for matches.
// The scan is performed in a separate goroutine and is subject to a timeout.
// The timeout check is performed between each pattern, so a single very slow
// pattern can still block for longer than the timeout. However, Go's regex
// engine is very fast and not vulnerable to ReDoS, so this is not a major
// concern in practice.
func (rp *RegexProvider) Scan(input string) ([]MatchResult, error) {
	if len(input) > rp.maxInputLength {
		return nil, fmt.Errorf("input length (%d) exceeds max_input_length (%d)", len(input), rp.maxInputLength)
	}

	ctx, cancel := context.WithTimeout(context.Background(), rp.timeout)
	defer cancel()

	resultChan := make(chan struct {
		matches []MatchResult
		err     error
	}, 1)

	go func() {
		var matches []MatchResult
		for i, re := range rp.compiled {
			select {
			case <-ctx.Done():
				// The context was cancelled, so we don't need to continue.
				return
			default:
				// Introduce a delay for testing purposes
				if rp.testDelay > 0 {
					time.Sleep(rp.testDelay)
				}

				locs := re.FindAllStringIndex(input, -1)
				for _, loc := range locs {
					matches = append(matches, MatchResult{
						PatternID:  rp.patterns[i].ID,
						Category:   rp.patterns[i].Category,
						Match:      input[loc[0]:loc[1]],
						StartIndex: loc[0],
						EndIndex:   loc[1],
					})
				}
			}
		}
		resultChan <- struct {
			matches []MatchResult
			err     error
		}{matches, nil}
	}()

	select {
	case res := <-resultChan:
		return res.matches, res.err
	case <-ctx.Done():
		return nil, fmt.Errorf("regex scan timed out after %v", rp.timeout)
	}
}
