//go:build !riscv64

package memory

import (
	"context"
	"fmt"
	"strconv"
	"sync"
	"time"

	glideoptions "github.com/valkey-io/valkey-glide/go/v2/options"
)

// valkeyForgetIfCurrentScript deletes the hash only when every compared field
// still matches. HMGET and DEL run in one script, so an update that lands
// after the last Get is not removed.
const valkeyForgetIfCurrentScriptSource = `
local fields = redis.call('HMGET', KEYS[1], 'id', 'user_id', 'project_id', 'memory_type', 'content', 'created_at', 'updated_at', 'importance')
if fields[1] == false or fields[1] == nil then
  return 0
end
if fields[1] ~= ARGV[1] or fields[2] ~= ARGV[2] or fields[3] ~= ARGV[3] or fields[4] ~= ARGV[4] or fields[5] ~= ARGV[5] or fields[6] ~= ARGV[6] or fields[7] ~= ARGV[7] or fields[8] ~= ARGV[8] then
  return 0
end
redis.call('DEL', KEYS[1])
return 1
`

var (
	valkeyForgetIfCurrentScriptOnce sync.Once
	valkeyForgetIfCurrentScript     *glideoptions.Script
)

func valkeyConditionalScript() *glideoptions.Script {
	valkeyForgetIfCurrentScriptOnce.Do(func() {
		valkeyForgetIfCurrentScript = glideoptions.NewScript(valkeyForgetIfCurrentScriptSource)
	})
	return valkeyForgetIfCurrentScript
}

func (v *ValkeyStore) forgetIfCurrent(ctx context.Context, want memoryVersion) (bool, error) {
	startTime := time.Now()
	status := "success"
	defer func() {
		RecordMemoryStoreOperation("valkey", "forget", status, time.Since(startTime).Seconds())
	}()

	if !v.enabled || v.client == nil {
		status = "error"
		return false, fmt.Errorf("valkey store is not enabled")
	}
	if err := ctx.Err(); err != nil {
		status = "error"
		return false, err
	}
	if want.id == "" {
		status = "error"
		return false, fmt.Errorf("memory ID is required")
	}

	scriptOptions := glideoptions.NewScriptOptions().
		WithKeys([]string{v.hashKey(want.id)}).
		WithArgs(valkeyForgetIfCurrentArgs(want))

	var result any
	err := v.retryWithBackoff(ctx, func() error {
		var runErr error
		result, runErr = v.client.InvokeScriptWithOptions(ctx, *valkeyConditionalScript(), *scriptOptions)
		return runErr
	})
	if err != nil {
		status = "error"
		return false, fmt.Errorf("valkey conditional delete failed for memory id=%s: %w", want.id, err)
	}
	return valkeyToInt64(result) == 1, nil
}

func valkeyForgetIfCurrentArgs(want memoryVersion) []string {
	projectID, _ := normalizedMemoryScopeFields(&Memory{ProjectID: want.projectID})
	return []string{
		want.id,
		want.userID,
		projectID,
		string(want.typ),
		want.content,
		strconv.FormatInt(want.createdAt.UnixMilli(), 10),
		strconv.FormatInt(want.updatedAt.UnixMilli(), 10),
		strconv.FormatFloat(float64(want.importance), 'f', -1, 32),
	}
}
