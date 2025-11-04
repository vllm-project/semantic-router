package vectordb

import "testing"

func TestChromaVectorDb(t *testing.T) {
	t.Run("ChromaVectorDb", func(t *testing.T) {
		es := NewOpenAIEmbeddingService(NewOpenAIEmbeddingServiceOptions{
			Endpoint: "http://localhost:11434/v1/",
		})
		options := ChromaVectorDbOptions{
			Endpoint:         "http://localhost:8000",
			Collection:       "my_collection",
			EmbeddingService: es,
			EmbeddingModel:   "all-minilm",
		}
		c, err := NewChromaVectorDb(options)
		if err != nil {
			t.Fatalf("Got error: %s", err)
		}
		t.Log("Vector DB instance created")
		res, err := c.Query("Hello")
		t.Log("Query complete")
		if err != nil {
			t.Fatalf("Got error: %s", err)
		}
		t.Log("Content:")
		for _, content := range res {
			t.Logf("%s \n", content)
		}
	})
}
