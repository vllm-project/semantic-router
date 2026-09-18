package memory

import (
	"net"
	"os"

	"github.com/vllm-project/semantic-router/src/semantic-router/internal/testutil/storagetest"
)

func storageRedisAddress() string {
	host, port := os.Getenv("REDIS_HOST"), os.Getenv("REDIS_PORT")
	if host == "" {
		host = "localhost"
	}
	if port == "" {
		port = "6379"
	}
	return net.JoinHostPort(host, port)
}

func storageMemoryVectors() storagetest.Vectors {
	return storagetest.Vectors{Size: 384, Aliases: map[string]string{
		"What are the user's display preferences?":             "display",
		"The user prefers dark mode in all their applications": "display",
		"Python programming":                                   "python",
		"User likes Python programming language very much":     "python",
		"To run Python tests use pytest command in terminal":   "python",
		"chocolate preference":                                 "chocolate",
		"User A's secret preference is dark chocolate":         "chocolate",
		"User B's secret preference is white chocolate":        "chocolate",
	}}
}
