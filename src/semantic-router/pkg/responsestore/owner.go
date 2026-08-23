package responsestore

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/responseapi"
)

func validateOwner(owner responseapi.ResponseOwner) error {
	if !owner.Valid() {
		return ErrInvalidInput
	}
	return nil
}

// ownerPartition returns a fixed-width opaque Redis partition. Ownership is
// also persisted in each value and verified after decoding.
func ownerPartition(owner responseapi.ResponseOwner) string {
	hash := sha256.New()
	for _, field := range []string{string(owner.Mode), owner.NamespaceID, owner.APIKeyID, owner.UserID} {
		var length [8]byte
		binary.BigEndian.PutUint64(length[:], uint64(len(field)))
		_, _ = hash.Write(length[:])
		_, _ = hash.Write([]byte(field))
	}
	return hex.EncodeToString(hash.Sum(nil))
}
