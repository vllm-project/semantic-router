package config

import "github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"

// Operator overrides are bounded here; the request body cap still bounds the
// whole request.
const (
	MaxProtocolToolNameBytes        = 1 << 10
	MaxProtocolToolDescriptionBytes = 1 << 20
)

// ProtocolLimitsConfig overrides the codec's tool text limits in UTF-8 bytes.
// Zero keeps the default.
type ProtocolLimitsConfig struct {
	ToolNameBytes        int `yaml:"tool_name_bytes,omitempty" json:"tool_name_bytes,omitempty" jsonschema:"minimum=0,maximum=1024"`
	ToolDescriptionBytes int `yaml:"tool_description_bytes,omitempty" json:"tool_description_bytes,omitempty" jsonschema:"minimum=0,maximum=1048576"`
}

// ProtocolPolicy is the codec policy for every engine built from this config,
// so ingress and internal re-encodes accept the same requests.
func (c *RouterConfig) ProtocolPolicy() llmprotocol.Policy {
	policy := llmprotocol.DefaultPolicy()
	if c == nil {
		return policy
	}
	policy.AllowedLossyFields = c.AllowLossyTranslation
	if c.ProtocolLimits.ToolNameBytes > 0 {
		policy.Limits.ToolNameBytes = c.ProtocolLimits.ToolNameBytes
	}
	if c.ProtocolLimits.ToolDescriptionBytes > 0 {
		policy.Limits.ToolDescriptionBytes = c.ProtocolLimits.ToolDescriptionBytes
	}
	return policy
}
