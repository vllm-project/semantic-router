package accesspublisher

import (
	"strings"
	"testing"
)

func TestKeyspacePinsNamespaceStateToQuotaPartitionSlot(t *testing.T) {
	keys, err := NewKeyspace("tenant-test", "namespace one", "partition-one")
	if err != nil {
		t.Fatal(err)
	}
	for name, key := range map[string]string{
		"access gate": keys.AccessGate(), "routing gate": keys.RoutingGate(),
		"publication": keys.Publication("pub-1"), "policy": keys.AccessDocument("key-1", 1),
		"logical key": keys.LogicalKey("key-1"), "credential": keys.CredentialPointer("api_key", "kid-1"),
	} {
		if !strings.HasPrefix(key, "tenant-test:") || !strings.Contains(key, "{partition-one}") {
			t.Fatalf("%s key %q does not use the prefixed partition slot", name, key)
		}
	}
	directory, err := keys.CredentialDirectory("api_key", "kid-1")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(directory, "tenant-test:") || strings.Contains(directory, "{partition-one}") {
		t.Fatalf("credential directory %q must be global and prefixed", directory)
	}
}

func TestSelectPointerUsesGateAndFailsClosed(t *testing.T) {
	values := map[string]string{
		"publication_id": "pub-old", "state": "active", "revision": "1", "digest": strings.Repeat("a", 64),
		"pending_publication_id": "pub-new", "pending_state": "active", "pending_revision": "2", "pending_digest": strings.Repeat("b", 64),
	}
	old, state, err := SelectPointer(values, "pub-old")
	if err != nil || state != PointerStateActive || old["revision"] != "1" {
		t.Fatalf("old pointer selection = %+v, %q, %v", old, state, err)
	}
	next, state, err := SelectPointer(values, "pub-new")
	if err != nil || state != PointerStateActive || next["revision"] != "2" || next["publication_id"] != "pub-new" {
		t.Fatalf("pending pointer selection = %+v, %q, %v", next, state, err)
	}
	if _, _, err := SelectPointer(values, "pub-unknown"); err == nil {
		t.Fatal("pointer selected a publication not named by the gate")
	}
}

func TestParsePublicationGateRejectsIncompleteIdentity(t *testing.T) {
	values := map[string]string{
		"publication_id": "pub-1", "revision": "3", "runtime_epoch": "2",
		"publication_digest": strings.Repeat("a", 64), "manifest_digest": strings.Repeat("b", 64),
	}
	gate, err := ParsePublicationGate(values)
	if err != nil || gate.Revision != 3 || gate.RuntimeEpoch != 2 {
		t.Fatalf("ParsePublicationGate() = %+v, %v", gate, err)
	}
	delete(values, "publication_digest")
	if _, err := ParsePublicationGate(values); err == nil {
		t.Fatal("gate without publication digest unexpectedly parsed")
	}
}
