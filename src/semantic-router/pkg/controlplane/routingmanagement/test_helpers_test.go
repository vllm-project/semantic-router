package routingmanagement

import (
	"bytes"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/securitykeyring"
)

func testRoutingCursorKeyring() securitykeyring.Symmetric {
	return securitykeyring.Symmetric{
		ActiveVersion: "cursor-v1",
		Keys:          map[string][]byte{"cursor-v1": bytes.Repeat([]byte{0x6d}, 32)},
	}
}
