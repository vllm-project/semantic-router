package cache

import "testing"

func TestPartitionKeyKeepsEscapedPartsDistinct(t *testing.T) {
	colon := CachePartition{Recipe: "a:b"}.Key()
	literal := CachePartition{Recipe: "a%3Ab"}.Key()
	if colon == literal {
		t.Fatalf("two distinct recipes share one partition key: %q", colon)
	}

	base := CachePartition{Recipe: "recipe", Decision: "decision"}
	if base.Key() != (CachePartition{Recipe: "recipe", Decision: "decision"}).Key() {
		t.Fatal("partition key is not deterministic")
	}
}
