package config

// MCPConfigServerConfig controls the in-router MCP config synthesis server.
type MCPConfigServerConfig struct {
	Enabled            bool     `yaml:"enabled,omitempty"`
	LoopbackOnly       bool     `yaml:"loopback_only,omitempty"`
	AllowDestructive   bool     `yaml:"allow_destructive,omitempty"`
	AllowedTools       []string `yaml:"allowed_tools,omitempty"`
	RateLimitPerMinute int      `yaml:"rate_limit_per_minute,omitempty"`
}

// WithDefaults returns cfg with router-owned defaults applied.
func (c MCPConfigServerConfig) WithDefaults() MCPConfigServerConfig {
	out := c
	if !out.Enabled {
		return out
	}
	if out.RateLimitPerMinute <= 0 {
		out.RateLimitPerMinute = 60
	}
	return out
}
