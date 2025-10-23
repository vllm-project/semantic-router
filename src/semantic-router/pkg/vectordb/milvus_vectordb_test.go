package vectordb

import "testing"

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
