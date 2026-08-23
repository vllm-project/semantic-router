package postgres

import (
	"crypto/sha256"
	"errors"
	"sync"
	"testing"
)

func TestRefreshBootstrapTokenStateIsConcurrentAndOneWay(t *testing.T) {
	service := &BootstrapService{
		tokenConfigured: true,
		tokenDigest:     sha256.Sum256([]byte("bootstrap-token-that-is-at-least-thirty-two-bytes")),
		tokenPresent:    func() (bool, error) { return false, nil },
	}
	const workers = 64
	var wait sync.WaitGroup
	errorsSeen := make(chan error, workers)
	for index := 0; index < workers; index++ {
		wait.Add(1)
		go func() {
			defer wait.Done()
			configured, err := service.refreshTokenState()
			if err != nil || configured {
				errorsSeen <- errors.New("bootstrap token remained configured")
			}
		}()
	}
	wait.Wait()
	close(errorsSeen)
	for err := range errorsSeen {
		t.Fatal(err)
	}
	if service.tokenConfigured || service.tokenDigest != ([sha256.Size]byte{}) {
		t.Fatal("bootstrap credential was not erased")
	}
	service.tokenPresent = func() (bool, error) { return true, nil }
	if configured, err := service.refreshTokenState(); err != nil || configured {
		t.Fatalf("removed bootstrap credential was restored: %v, %v", configured, err)
	}
}

func TestRefreshBootstrapTokenStatePreservesAuthorityOnProbeFailure(t *testing.T) {
	digest := sha256.Sum256([]byte("bootstrap-token-that-is-at-least-thirty-two-bytes"))
	service := &BootstrapService{
		tokenConfigured: true, tokenDigest: digest,
		tokenPresent: func() (bool, error) { return false, errors.New("permission denied") },
	}
	if configured, err := service.refreshTokenState(); err == nil || configured {
		t.Fatalf("refreshTokenState() = %v, %v; want failure", configured, err)
	}
	if !service.tokenConfigured || service.tokenDigest != digest {
		t.Fatal("probe failure destroyed live bootstrap authority")
	}
}
