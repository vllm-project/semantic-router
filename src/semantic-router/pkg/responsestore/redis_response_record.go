package responsestore

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/google/uuid"
	"github.com/redis/go-redis/v9"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/responseapi"
)

const responseGenerationField = "_vsr_generation"

// persistedResponse is deliberately a flat extension of StoredResponse.
// Older binaries ignore the generation field while decoding, and this code
// can still decode payloads written before the field existed.
type persistedResponse struct {
	responseapi.StoredResponse
	Generation *string `json:"_vsr_generation,omitempty"`
}

// responseRecord is the internal representation of one persisted response.
// A blank generation denotes legacy data. New writes always carry a UUID-v4.
type responseRecord struct {
	response   *responseapi.StoredResponse
	generation string
	raw        []byte
}

func newResponseGeneration() string {
	return uuid.NewString()
}

func marshalResponseRecord(response *responseapi.StoredResponse, generation string) ([]byte, error) {
	if response == nil || generation == "" {
		return nil, ErrInvalidInput
	}
	if err := validateResponseGeneration(generation); err != nil {
		return nil, err
	}

	record := persistedResponse{StoredResponse: *response, Generation: &generation}
	data, err := json.Marshal(&record)
	if err != nil {
		return nil, fmt.Errorf("failed to serialize response: %w", err)
	}
	return data, nil
}

func decodeResponseRecord(data []byte) (responseRecord, error) {
	var persisted persistedResponse
	if err := json.Unmarshal(data, &persisted); err != nil {
		return responseRecord{}, fmt.Errorf("failed to deserialize response: %w", err)
	}

	generation := ""
	if persisted.Generation != nil {
		generation = *persisted.Generation
		if err := validateResponseGeneration(generation); err != nil {
			return responseRecord{}, err
		}
	}

	response := persisted.StoredResponse
	return responseRecord{response: &response, generation: generation, raw: data}, nil
}

func validateResponseGeneration(generation string) error {
	parsed, err := uuid.Parse(generation)
	if err != nil || parsed.Version() != 4 {
		return fmt.Errorf("invalid response generation")
	}
	return nil
}

// compareDeleteGenerationScript removes a payload only while the persisted
// generation is the generation owned by the caller. Legacy payloads and
// malformed JSON fail closed. No byte-equality fallback is permitted.
var compareDeleteGenerationScript = redis.NewScript(`
local current = redis.call("GET", KEYS[1])
if not current then
	return 0
end
local ok, decoded = pcall(cjson.decode, current)
if not ok or type(decoded) ~= "table" then
	return 0
end
if decoded["_vsr_generation"] ~= ARGV[1] then
	return 0
end
return redis.call("DEL", KEYS[1])
`)

// compareRestoreGenerationScript restores ARGV[2] only while the current
// payload is still owned by ARGV[1]. ARGV[2] must carry a newly minted
// generation, never the snapshot's old generation. Legacy current payloads
// fail closed. ARGV[3] follows the package TTL convention: zero deletes,
// negative persists, and positive values are milliseconds.
var compareRestoreGenerationScript = redis.NewScript(`
local current = redis.call("GET", KEYS[1])
if not current then
	return 0
end
local ok, decoded = pcall(cjson.decode, current)
if not ok or type(decoded) ~= "table" then
	return 0
end
if decoded["_vsr_generation"] ~= ARGV[1] then
	return 0
end

local ttl = tonumber(ARGV[3])
if ttl == 0 then
	redis.call("DEL", KEYS[1])
	return 2
elseif ttl < 0 then
	redis.call("SET", KEYS[1], ARGV[2])
else
	redis.call("SET", KEYS[1], ARGV[2], "PX", ttl)
end
return 1
`)

// takeResponsePayloadScript atomically returns and removes one payload. The
// returned bytes identify the exact generation whose index witness may be
// removed afterwards; a concurrent recreation receives a different witness.
var takeResponsePayloadScript = redis.NewScript(`
local current = redis.call("GET", KEYS[1])
if not current then
	return nil
end
redis.call("DEL", KEYS[1])
return current
`)

// spliceResponseGeneration adds the generation to a payload's raw JSON object
// without round-tripping it through StoredResponse.
//
// Remarshaling would silently drop every field this build does not know about
// — one written by a newer binary, or one retired from the struct but still
// present in stored data. Promotion's entire justification is that it changes
// nothing but the generation, so it must preserve the object it found.
// json.RawMessage keeps each value's bytes verbatim; only key ordering and
// inter-token whitespace change, and nothing depends on either — the promotion
// CAS compares the *previous* bytes, which are untouched.
func spliceResponseGeneration(raw []byte, generation string) ([]byte, error) {
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(raw, &fields); err != nil {
		return nil, fmt.Errorf("failed to decode response payload for upgrade: %w", err)
	}
	if fields == nil {
		return nil, ErrInvalidInput
	}
	if _, present := fields[responseGenerationField]; present {
		return nil, fmt.Errorf("response payload already carries a generation")
	}

	encoded, err := json.Marshal(generation)
	if err != nil {
		return nil, fmt.Errorf("failed to encode response generation: %w", err)
	}
	fields[responseGenerationField] = encoded

	data, err := json.Marshal(fields)
	if err != nil {
		return nil, fmt.Errorf("failed to re-encode upgraded response payload: %w", err)
	}
	return data, nil
}

// promoteLegacyPayloadScript upgrades a generation-less payload in place,
// conditional on the exact bytes the caller read, and returns the lifetime the
// key actually holds at that instant.
//
// Byte equality is the only witness a legacy payload has, and it is sound here
// in a way byte-equality *deletion* is not: if the value this replaces was
// concurrently deleted and recreated byte-identically by an index-unaware
// writer, the promotion still writes content identical to what that writer
// stored, under the TTL Redis holds right now. Nothing is destroyed. What must
// never happen is byte equality authorizing destruction *transitively* — so no
// caller may promote and then delete the minted generation in the same step.
// The generation this yields authorizes an index write and nothing else; a
// later, direct observation of a generational payload is what authorizes its
// deletion. See resolveCascadeDeleteOutcome.
//
// Returning the PTTL matters as much as preserving it. A caller that scanned
// the payload earlier holds a lifetime that may predate a byte-identical
// recreation with a longer TTL, and indexing under that stale bound would
// retire the conversation index ahead of the payload it names — exactly what
// conversationIndexAddScript's extend-only expiry exists to prevent. The
// lifetime is therefore read in the same atomic step as the write.
//
// Single-key: KEYS[1] only, so it stays legal in Redis Cluster.
var promoteLegacyPayloadScript = redis.NewScript(`
local current = redis.call("GET", KEYS[1])
if not current or current ~= ARGV[1] then
	return {0, 0}
end
local pttl = redis.call("PTTL", KEYS[1])
if pttl == -1 then
	redis.call("SET", KEYS[1], ARGV[2])
	return {1, -1}
end
if pttl <= 0 then
	return {0, 0}
end
redis.call("SET", KEYS[1], ARGV[2], "PX", pttl)
return {1, pttl}
`)

// promoteLegacyResponsePayload stamps a fresh generation onto a payload that
// has none, returning that generation and the payload's remaining lifetime as
// observed in the same atomic step.
//
// ok=false is not an error: it means the payload changed under the caller
// (deleted, expired, or already promoted by someone else), and the caller
// re-reads on its next round. A lost promotion is never a licence to fall back
// to a weaker deletion or pruning rule.
//
// Takes an explicit client because the scan paths run per Cluster master.
func promoteLegacyResponsePayload(ctx context.Context, client redis.UniversalClient, key string, record responseRecord) (string, int64, bool, error) {
	if record.generation != "" {
		return record.generation, unknownPayloadTTL, true, nil
	}
	if record.response == nil || len(record.raw) == 0 {
		return "", 0, false, ErrInvalidInput
	}

	generation := newResponseGeneration()
	data, err := spliceResponseGeneration(record.raw, generation)
	if err != nil {
		return "", 0, false, err
	}

	res, err := promoteLegacyPayloadScript.Run(ctx, client, []string{key}, record.raw, data).Result()
	if err != nil {
		return "", 0, false, fmt.Errorf("failed to upgrade legacy response payload %s: %w", key, err)
	}
	items, ok := res.([]interface{})
	if !ok || len(items) != 2 {
		return "", 0, false, fmt.Errorf("unexpected legacy upgrade result shape %#v for %s", res, key)
	}
	promoted, promotedOK := items[0].(int64)
	ttlMillis, ttlOK := items[1].(int64)
	if !promotedOK || !ttlOK {
		return "", 0, false, fmt.Errorf("unexpected legacy upgrade result types %T/%T for %s", items[0], items[1], key)
	}
	if promoted == 0 {
		return "", 0, false, nil
	}

	return generation, ttlMillis, true, nil
}
