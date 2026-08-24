package routerauth

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"testing"
)

func TestFirstAdminProvisionerCompletesRouterAuthorityAndFinalizesToken(t *testing.T) {
	t.Parallel()
	setup := newFirstAdminProvisioningSetup(t)
	if provisionErr := setup.provider.ProvisionFirstAdmin(context.Background(), setup.fixture.identity); provisionErr != nil {
		t.Fatalf("ProvisionFirstAdmin() error = %v", provisionErr)
	}
	assertFirstAdminProvisioningCalls(t, setup.fixture)
	assertBootstrapTokenFinalized(t, setup.fixture.tokenPath)
	if provisionErr := setup.provider.ProvisionFirstAdmin(context.Background(), setup.fixture.identity); provisionErr != nil {
		t.Fatalf("idempotent ProvisionFirstAdmin() error = %v", provisionErr)
	}
}

func TestBootstrapTokenRequiresOwnerOnlyFileAndRefusesReplacement(t *testing.T) {
	t.Parallel()
	directory := t.TempDir()
	path := filepath.Join(directory, "router-token")
	if err := os.WriteFile(path, []byte("router-bootstrap-token-which-is-at-least-32-bytes"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := observeBootstrapToken(path); err == nil {
		t.Fatal("observeBootstrapToken() accepted group/world-readable secret")
	}
	if err := os.Chmod(path, 0o600); err != nil {
		t.Fatal(err)
	}
	observed, err := observeBootstrapToken(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Remove(path); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte("replacement-bootstrap-token-at-least-32-bytes"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := finalizeBootstrapToken(path, observed); err == nil {
		t.Fatal("finalizeBootstrapToken() removed a replacement file")
	}
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("replacement token was removed: %v", err)
	}
}

func TestFinalizeBootstrapTokenRejectsReplacementAfterFileIdentityReuse(t *testing.T) {
	t.Parallel()
	directory := t.TempDir()
	path := filepath.Join(directory, "router-token")
	original := []byte("router-bootstrap-token-which-is-at-least-32-bytes")
	if err := os.WriteFile(path, original, 0o600); err != nil {
		t.Fatal(err)
	}
	observed, err := observeBootstrapToken(path)
	if err != nil {
		t.Fatal(err)
	}
	if removeErr := os.Remove(path); removeErr != nil {
		t.Fatal(removeErr)
	}
	replacement := []byte("replacement-bootstrap-token-at-least-32-bytes")
	if writeErr := os.WriteFile(path, replacement, 0o600); writeErr != nil {
		t.Fatal(writeErr)
	}
	replacementInfo, err := os.Lstat(path)
	if err != nil {
		t.Fatal(err)
	}
	// Model inode reuse explicitly: a FileInfo comparison alone now reports
	// the replacement as the observed file, while the stable content identity
	// still describes the credential that was actually consumed.
	observed.fileInfo = replacementInfo
	if err := finalizeBootstrapToken(path, observed); err == nil {
		t.Fatal("finalizeBootstrapToken() accepted different content after file identity reuse")
	}
	if payload, err := os.ReadFile(path); err != nil || string(payload) != string(replacement) {
		t.Fatalf("replacement token = %q, %v", payload, err)
	}
}

func TestFinalizeBootstrapTokenRejectsRepeatedReplacement(t *testing.T) {
	t.Parallel()
	directory := t.TempDir()
	path := filepath.Join(directory, "router-token")
	for iteration := 0; iteration < 128; iteration++ {
		original := []byte(fmt.Sprintf("router-bootstrap-token-original-%032d", iteration))
		if err := os.WriteFile(path, original, 0o600); err != nil {
			t.Fatal(err)
		}
		observed, err := observeBootstrapToken(path)
		if err != nil {
			t.Fatal(err)
		}
		if err := os.Remove(path); err != nil {
			t.Fatal(err)
		}
		replacement := []byte(fmt.Sprintf("router-bootstrap-token-replaced-%032d", iteration))
		if err := os.WriteFile(path, replacement, 0o600); err != nil {
			t.Fatal(err)
		}
		if err := finalizeBootstrapToken(path, observed); err == nil {
			t.Fatalf("iteration %d accepted replacement token", iteration)
		}
		if payload, err := os.ReadFile(path); err != nil || string(payload) != string(replacement) {
			t.Fatalf("iteration %d replacement token = %q, %v", iteration, payload, err)
		}
		if err := os.Remove(path); err != nil {
			t.Fatal(err)
		}
	}
}

func TestFinalizeBootstrapTokenAcceptsConcurrentRemoval(t *testing.T) {
	t.Parallel()
	directory := t.TempDir()
	path := filepath.Join(directory, "router-token")
	if err := os.WriteFile(path, []byte("router-bootstrap-token-which-is-at-least-32-bytes"), 0o600); err != nil {
		t.Fatal(err)
	}
	observed, err := observeBootstrapToken(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Remove(path); err != nil {
		t.Fatal(err)
	}
	if err := finalizeBootstrapToken(path, observed); err != nil {
		t.Fatalf("finalizeBootstrapToken() after concurrent removal = %v", err)
	}
}

func TestFinalizeVerifiedBootstrapTokenClaimRejectsReplacement(t *testing.T) {
	t.Parallel()
	directory := t.TempDir()
	path := filepath.Join(directory, "router-token")
	claimDirectory := filepath.Join(directory, ".vllm-sr-bootstrap-finalize-test")
	claimPath := filepath.Join(claimDirectory, "token")
	if err := os.Mkdir(claimDirectory, 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(claimPath, []byte("router-bootstrap-token-which-is-at-least-32-bytes"), 0o600); err != nil {
		t.Fatal(err)
	}
	replacement := []byte("replacement-bootstrap-token-at-least-32-bytes")
	if err := os.WriteFile(path, replacement, 0o600); err != nil {
		t.Fatal(err)
	}

	if err := finalizeVerifiedBootstrapTokenClaim(path, claimPath, claimDirectory); err == nil {
		t.Fatal("finalizeVerifiedBootstrapTokenClaim() accepted a replacement token")
	}
	if payload, err := os.ReadFile(path); err != nil || string(payload) != string(replacement) {
		t.Fatalf("replacement token = %q, %v", payload, err)
	}
	if _, err := os.Stat(claimPath); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("verified token claim still exists: %v", err)
	}
	if _, err := os.Stat(claimDirectory); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("token claim directory still exists: %v", err)
	}
}
