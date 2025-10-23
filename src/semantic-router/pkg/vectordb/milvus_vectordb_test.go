package vectordb

import (
	"testing"
	"context"
)
func TestMilvusConnection(t *testing.T) {
	opts := MilvusVectorDbOptions{
		Endpoint:   "127.0.0.1:19530",
		Collection: "test_collection",
	}
	mdb, err := NewMilvusVectorDb(opts)
	if err != nil {
		t.Fatalf("Failed to connect: %v", err)
	}
	if mdb.client == nil {
		t.Fatal("Milvus client is nil")
	}
	t.Log("Milvus connection established successfully!")
}
func TestMilvusCollection(t *testing.T) {
    ctx := context.Background()

    m, err := NewMilvusVectorDb(MilvusVectorDbOptions{
        Endpoint:   "127.0.0.1:19530",
        Collection: "semantic_test",
    })
    if err != nil {
        t.Fatalf("Connection failed: %v", err)
    }

    // Create or load collection
    err = m.CreateOrLoadCollection(ctx, "semantic_test", 384)
    if err != nil {
        t.Fatalf("Collection creation failed: %v", err)
    }

    cols, _ := m.ListCollections(ctx)
    t.Logf("Collections found: %+v", cols)
}