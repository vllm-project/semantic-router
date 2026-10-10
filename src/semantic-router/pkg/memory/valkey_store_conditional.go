package memory

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"sort"
	"strconv"
	"sync"
	"time"

	glideoptions "github.com/valkey-io/valkey-glide/go/v2/options"
)

var (
	valkeyReplaceCurrentGroupScriptOnce sync.Once
	valkeyReplaceCurrentGroupScript     *glideoptions.Script
	valkeyUpdateIfCurrentScriptOnce     sync.Once
	valkeyUpdateIfCurrentCompiledScript *glideoptions.Script
	valkeyTrackRetrievalScriptOnce      sync.Once
	valkeyTrackRetrievalCompiledScript  *glideoptions.Script
)

// valkeyReplaceCurrentGroupScriptSource validates every source snapshot,
// creates the summary, and deletes the sources in a single Valkey script. A
// failed comparison has no side effects, so a concurrent source update cannot
// leave a summary containing stale content.
//
// The script is safe to retry. If Valkey commits the merge and the client
// loses the reply, a replay sees every source gone and the same summary
// already stored with its immutable consolidation receipt, and it returns the
// committed delete count instead of 0. Retrieval tracking changes summary
// access fields, so replay detection must not compare every summary field.
// Callers treat that count as success and invalidate cached retrievals.
const valkeyReplaceCurrentGroupScriptSource = `
local source_count = tonumber(ARGV[1])
local arg = 2
local missing = 0

for i = 1, source_count do
  local fields = redis.call('HMGET', KEYS[i], 'id', 'user_id', 'project_id', 'memory_type', 'content', 'created_at', 'updated_at', 'importance')
  if fields[1] == false or fields[1] == nil then
    missing = missing + 1
  else
    for j = 1, 8 do
      if fields[j] ~= ARGV[arg + j - 1] then
        return 0
      end
    end
  end
  arg = arg + 8
end

local summary_key = KEYS[source_count + 1]
if missing > 0 then
  if missing == source_count and
      redis.call('HGET', summary_key, 'consolidation_receipt') == ARGV[arg] then
    return source_count
  end
  return 0
end

if redis.call('EXISTS', summary_key) == 1 then
  return -1
end

arg = arg + 1
local field_count = tonumber(ARGV[arg])
arg = arg + 1
local hset_args = { summary_key }
for i = 1, field_count do
  table.insert(hset_args, ARGV[arg])
  table.insert(hset_args, ARGV[arg + 1])
  arg = arg + 2
end
redis.call('HSET', unpack(hset_args))

for i = 1, source_count do
  redis.call('DEL', KEYS[i])
end
return source_count
`

func valkeyAtomicGroupReplacementScript() *glideoptions.Script {
	valkeyReplaceCurrentGroupScriptOnce.Do(func() {
		valkeyReplaceCurrentGroupScript = glideoptions.NewScript(valkeyReplaceCurrentGroupScriptSource)
	})
	return valkeyReplaceCurrentGroupScript
}

// valkeyUpdateIfCurrentScriptSource prevents Update from recreating a source
// deleted by atomic consolidation between Update's initial Get and its write.
// The ID comparison and HSET must run in the same script because HSET creates
// a missing hash key. Checking the ID, rather than only key existence, also
// rejects an incomplete hash left by a stale background operation.
const valkeyUpdateIfCurrentScriptSource = `
if redis.call('HGET', KEYS[1], 'id') ~= ARGV[1] then
  return 0
end
redis.call('HSET', KEYS[1], unpack(ARGV, 2))
return 1
`

func valkeyUpdateIfCurrentScript() *glideoptions.Script {
	valkeyUpdateIfCurrentScriptOnce.Do(func() {
		valkeyUpdateIfCurrentCompiledScript = glideoptions.NewScript(valkeyUpdateIfCurrentScriptSource)
	})
	return valkeyUpdateIfCurrentCompiledScript
}

// valkeyTrackRetrievalScriptSource records retrieval metadata only when the
// hash still holds the requested memory ID. The check and writes are atomic so
// queued retrieval tracking cannot recreate a source deleted by consolidation
// or Forget.
const valkeyTrackRetrievalScriptSource = `
if redis.call('HGET', KEYS[1], 'id') ~= ARGV[1] then
  return 0
end
redis.call('HINCRBY', KEYS[1], 'access_count', 1)
redis.call('HSET', KEYS[1], 'last_accessed', ARGV[2])
return 1
`

func valkeyTrackRetrievalScript() *glideoptions.Script {
	valkeyTrackRetrievalScriptOnce.Do(func() {
		valkeyTrackRetrievalCompiledScript = glideoptions.NewScript(valkeyTrackRetrievalScriptSource)
	})
	return valkeyTrackRetrievalCompiledScript
}

// updateHashIfCurrent updates a memory hash only while it still holds the
// expected ID, without allowing HSET to recreate it after consolidation or
// Forget removes it.
func (v *ValkeyStore) updateHashIfCurrent(ctx context.Context, key, id string, fields map[string]string) (bool, error) {
	scriptOptions := glideoptions.NewScriptOptions().
		WithKeys([]string{key}).
		WithArgs(append([]string{id}, valkeyHashFieldArgs(fields)...))

	var result any
	err := v.retryWithBackoff(ctx, func() error {
		var runErr error
		result, runErr = v.client.InvokeScriptWithOptions(ctx, *valkeyUpdateIfCurrentScript(), *scriptOptions)
		return runErr
	})
	if err != nil {
		return false, err
	}
	return valkeyToInt64(result) == 1, nil
}

func (v *ValkeyStore) supportsAtomicGroupReplacement() bool {
	return v.enabled && !v.clusterMode
}

// replaceCurrentGroup atomically creates a summary and removes all source
// memories only while each source still matches the snapshot from List.
func (v *ValkeyStore) replaceCurrentGroup(ctx context.Context, versions []memoryVersion, summary *Memory) (bool, int, error) {
	startTime := time.Now()
	status := "success"
	defer func() {
		RecordMemoryStoreOperation("valkey", "consolidate", status, time.Since(startTime).Seconds())
	}()

	if !v.enabled || v.client == nil {
		status = "error"
		return false, 0, fmt.Errorf("valkey store is not enabled")
	}
	if err := ctx.Err(); err != nil {
		status = "error"
		return false, 0, err
	}
	if len(versions) < 2 {
		status = "error"
		return false, 0, fmt.Errorf("at least two source memories are required")
	}
	if err := valkeyValidateMemory(summary); err != nil {
		status = "error"
		return false, 0, err
	}

	embedding := summary.Embedding
	if len(embedding) == 0 {
		var err error
		embedding, err = embedForWrite(ctx, summary.Content, v.embeddingConfig)
		if err != nil {
			status = "error"
			return false, 0, fmt.Errorf("failed to generate summary embedding: %w", err)
		}
		summary.Embedding = embedding
	}
	now := time.Now()
	if summary.CreatedAt.IsZero() {
		summary.CreatedAt = now
	}
	summary.UpdatedAt = now
	if summary.LastAccessed.IsZero() {
		summary.LastAccessed = now
	}
	fields, err := valkeyBuildHashFields(summary, embedding)
	if err != nil {
		status = "error"
		return false, 0, fmt.Errorf("failed to build summary fields: %w", err)
	}
	fields["consolidation_receipt"] = valkeyConsolidationReceipt(versions, summary.ID)

	keys := make([]string, 0, len(versions)+1)
	for _, want := range versions {
		if want.id == "" {
			status = "error"
			return false, 0, fmt.Errorf("source memory ID is required")
		}
		keys = append(keys, v.hashKey(want.id))
	}
	keys = append(keys, v.hashKey(summary.ID))
	scriptOptions := glideoptions.NewScriptOptions().
		WithKeys(keys).
		WithArgs(valkeyReplaceCurrentGroupArgs(versions, fields))

	var result any
	err = v.retryWithBackoff(ctx, func() error {
		var runErr error
		result, runErr = v.client.InvokeScriptWithOptions(ctx, *valkeyAtomicGroupReplacementScript(), *scriptOptions)
		return runErr
	})
	if err != nil {
		status = "error"
		return false, 0, fmt.Errorf("valkey atomic consolidation failed: %w", err)
	}
	switch n := valkeyToInt64(result); {
	case n == int64(len(versions)):
		return true, len(versions), nil
	case n == 0:
		return false, 0, nil
	case n == -1:
		status = "error"
		return false, 0, fmt.Errorf("valkey summary memory ID already exists: %s", summary.ID)
	default:
		status = "error"
		return false, 0, fmt.Errorf("unexpected valkey atomic consolidation result: %d", n)
	}
}

func valkeySourceVersionArgs(want memoryVersion) []string {
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

func valkeyReplaceCurrentGroupArgs(versions []memoryVersion, fields map[string]string) []string {
	args := make([]string, 0, 1+len(versions)*8+2+len(fields)*2)
	args = append(args, strconv.Itoa(len(versions)))
	for _, want := range versions {
		args = append(args, valkeySourceVersionArgs(want)...)
	}
	args = append(args, fields["consolidation_receipt"])
	fieldArgs := valkeyHashFieldArgs(fields)
	args = append(args, strconv.Itoa(len(fieldArgs)/2))
	args = append(args, fieldArgs...)
	return args
}

// valkeyConsolidationReceipt is the durable identity of one atomic
// consolidation. It includes the new summary ID and every source snapshot,
// which lets a retry distinguish its committed merge from separately deleted
// sources or an unrelated summary with the same key. It intentionally omits
// summary access fields because reads update those after the merge commits.
func valkeyConsolidationReceipt(versions []memoryVersion, summaryID string) string {
	hash := sha256.New()
	write := func(value string) {
		// The length prefix makes the encoded field sequence unambiguous.
		_, _ = fmt.Fprintf(hash, "%d:%s", len(value), value)
	}

	write(summaryID)
	write(strconv.Itoa(len(versions)))
	for _, version := range versions {
		for _, value := range valkeySourceVersionArgs(version) {
			write(value)
		}
	}
	return hex.EncodeToString(hash.Sum(nil))
}

func valkeyHashFieldArgs(fields map[string]string) []string {
	names := make([]string, 0, len(fields))
	for name := range fields {
		names = append(names, name)
	}
	sort.Strings(names)

	args := make([]string, 0, len(fields)*2)
	for _, name := range names {
		args = append(args, name, fields[name])
	}
	return args
}
