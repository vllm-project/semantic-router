package usageledger

import (
	_ "embed"

	"github.com/redis/go-redis/v9"
)

var (
	//go:embed scripts/acknowledge.lua
	acknowledgeUsageScriptSource string
	acknowledgeUsageScript       = redis.NewScript(acknowledgeUsageScriptSource)

	//go:embed scripts/quarantine.lua
	quarantineUsageScriptSource string
	quarantineUsageScript       = redis.NewScript(quarantineUsageScriptSource)
)
