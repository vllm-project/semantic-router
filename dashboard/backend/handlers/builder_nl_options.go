package handlers

import "time"

const (
	builderNLDefaultTemperature = 0.1
	builderNLDefaultMaxRetries  = 1
	builderNLDefaultTimeout     = 180 * time.Second
	builderNLMinTemperature     = 0.0
	builderNLMaxTemperature     = 2.0
	builderNLMinMaxRetries      = 0
	builderNLMaxMaxRetries      = 4
	builderNLMinTimeout         = 30 * time.Second
	builderNLMaxTimeout         = 10 * time.Minute
	builderNLDefaultMaxTokens   = 65536
	builderNLHeartbeatInterval  = 5 * time.Second
)

type builderNLRuntimeOptions struct {
	Temperature  float64
	MaxRetries   int
	Timeout      time.Duration
	TimeoutLabel string
}

func resolveBuilderNLRuntimeOptions(req BuilderNLGenerateRequest) builderNLRuntimeOptions {
	timeout := clampBuilderNLTimeout(req.TimeoutSeconds)
	return builderNLRuntimeOptions{
		Temperature:  clampBuilderNLTemperature(req.Temperature),
		MaxRetries:   clampBuilderNLMaxRetries(req.MaxRetries),
		Timeout:      timeout,
		TimeoutLabel: timeout.Round(time.Second).String(),
	}
}

func resolveBuilderNLVerifyTimeout(req BuilderNLVerifyRequest) time.Duration {
	return clampBuilderNLTimeout(req.TimeoutSeconds)
}

func resolveBuilderNLVerifyRuntimeOptions(req BuilderNLVerifyRequest) builderNLRuntimeOptions {
	timeout := resolveBuilderNLVerifyTimeout(req)
	return builderNLRuntimeOptions{
		Timeout:      timeout,
		TimeoutLabel: timeout.Round(time.Second).String(),
	}
}

func clampBuilderNLTemperature(raw *float64) float64 {
	if raw == nil {
		return builderNLDefaultTemperature
	}
	if *raw < builderNLMinTemperature {
		return builderNLMinTemperature
	}
	if *raw > builderNLMaxTemperature {
		return builderNLMaxTemperature
	}
	return *raw
}

func clampBuilderNLMaxRetries(raw *int) int {
	if raw == nil {
		return builderNLDefaultMaxRetries
	}
	if *raw < builderNLMinMaxRetries {
		return builderNLMinMaxRetries
	}
	if *raw > builderNLMaxMaxRetries {
		return builderNLMaxMaxRetries
	}
	return *raw
}

func clampBuilderNLTimeout(raw *int) time.Duration {
	if raw == nil {
		return builderNLDefaultTimeout
	}
	timeout := time.Duration(*raw) * time.Second
	if timeout < builderNLMinTimeout {
		return builderNLMinTimeout
	}
	if timeout > builderNLMaxTimeout {
		return builderNLMaxTimeout
	}
	return timeout
}
