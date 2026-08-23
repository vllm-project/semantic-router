package accesscredential

import (
	"bytes"
	"testing"
)

func TestSecretKeyringCloneOwnsAndCloseErasesKeyMaterial(t *testing.T) {
	pepperSource := PepperKeyring{ActiveVersion: "pepper-v1", Keys: map[string][]byte{
		"pepper-v1": bytes.Repeat([]byte{0x11}, 32),
	}}
	pepper := pepperSource.Clone()
	pepperSource.Keys["pepper-v1"][0] = 0x22
	if pepper.Keys["pepper-v1"][0] != 0x11 {
		t.Fatal("pepper clone retained caller-owned key bytes")
	}
	pepperBytes := pepper.Keys["pepper-v1"]
	pepper.Close()
	if pepper.ActiveVersion != "" || pepper.Keys != nil || !allZero(pepperBytes) {
		t.Fatal("pepper Close did not erase owned key bytes")
	}

	kekSource := KEKKeyring{ActiveVersion: "kek-v1", Keys: map[string][]byte{
		"kek-v1": bytes.Repeat([]byte{0x33}, 32),
	}}
	kek := kekSource.Clone()
	kekSource.Keys["kek-v1"][0] = 0x44
	if kek.Keys["kek-v1"][0] != 0x33 {
		t.Fatal("KEK clone retained caller-owned key bytes")
	}
	kekBytes := kek.Keys["kek-v1"]
	kek.Close()
	if kek.ActiveVersion != "" || kek.Keys != nil || !allZero(kekBytes) {
		t.Fatal("KEK Close did not erase owned key bytes")
	}
}

func allZero(value []byte) bool {
	for _, item := range value {
		if item != 0 {
			return false
		}
	}
	return true
}
