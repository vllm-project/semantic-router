package responsestore

import (
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
