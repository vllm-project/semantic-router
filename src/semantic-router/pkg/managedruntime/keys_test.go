package managedruntime

import (
	"crypto/ed25519"
	"encoding/base64"
	"fmt"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/securitykeyring"
)

func TestControlPlaneHKDFIsStableAndDomainSeparated(t *testing.T) {
	root := securitykeyring.Symmetric{
		ActiveVersion: "root-2",
		Keys: map[string][]byte{
			"root-1": []byte(strings.Repeat("1", 32)),
			"root-2": []byte(strings.Repeat("2", 32)),
		},
	}
	first, err := deriveControlPlaneKeyrings(root)
	if err != nil {
		t.Fatalf("deriveControlPlaneKeyrings() error = %v", err)
	}
	second, err := deriveControlPlaneKeyrings(root)
	if err != nil {
		t.Fatalf("deriveControlPlaneKeyrings() second error = %v", err)
	}
	firstCatalog := first.CatalogCursor.Symmetric()
	secondCatalog := second.CatalogCursor.Symmetric()
	if string(firstCatalog.Keys["root-2"]) != string(secondCatalog.Keys["root-2"]) {
		t.Fatal("same root/domain/version produced different derived keys")
	}
	domains := map[string]HMACKeyring{
		"catalog cursor":        first.CatalogCursor,
		"discovery claim":       first.DiscoveryClaim,
		"management command":    first.ManagementCommand,
		"management cursor":     first.ManagementCursor,
		"bootstrap idempotency": first.BootstrapIdempotency,
		"backend dispatch":      first.BackendDispatch,
	}
	seen := make(map[string]string, len(domains))
	for name, candidate := range domains {
		encoded := string(candidate.Symmetric().Keys["root-2"])
		if prior, found := seen[encoded]; found {
			t.Fatalf("%s reused the %s derived key", name, prior)
		}
		seen[encoded] = name
	}
}

func TestControlPlaneKeyringSignsOnlyActiveAndVerifiesRetained(t *testing.T) {
	keys := map[string][]byte{
		"root-1": []byte(strings.Repeat("1", 32)),
		"root-2": []byte(strings.Repeat("2", 32)),
	}
	old, err := deriveControlPlaneKeyrings(securitykeyring.Symmetric{ActiveVersion: "root-1", Keys: keys})
	if err != nil {
		t.Fatal(err)
	}
	current, err := deriveControlPlaneKeyrings(securitykeyring.Symmetric{ActiveVersion: "root-2", Keys: keys})
	if err != nil {
		t.Fatal(err)
	}
	payload := []byte("opaque cursor payload")
	oldVersion, oldSignature, err := old.ManagementCursor.Sign(payload)
	if err != nil {
		t.Fatal(err)
	}
	if oldVersion != "root-1" || !current.ManagementCursor.Verify(oldVersion, payload, oldSignature) {
		t.Fatal("retained old version did not verify")
	}
	currentVersion, _, err := current.ManagementCursor.Sign(payload)
	if err != nil {
		t.Fatal(err)
	}
	if currentVersion != "root-2" {
		t.Fatalf("new signature version = %q, want root-2", currentVersion)
	}
	retired, err := deriveControlPlaneKeyrings(securitykeyring.Symmetric{
		ActiveVersion: "root-2", Keys: map[string][]byte{"root-2": keys["root-2"]},
	})
	if err != nil {
		t.Fatal(err)
	}
	if retired.ManagementCursor.Verify(oldVersion, payload, oldSignature) {
		t.Fatal("removed key version unexpectedly verified")
	}
}

func TestManagedKeyringLoadingFailsClosedForMissingOrWeakControlRoot(t *testing.T) {
	cfg := managedKeyringConfig(t)
	t.Setenv("TEST_PROVIDER_KEK", symmetricDocument("v1", 32))
	t.Setenv("TEST_MANAGEMENT_SIGNING", signingDocument(t, "v1"))
	t.Setenv("TEST_SERVICE_HMAC", symmetricDocument("v1", 32))
	t.Setenv("TEST_INVITATION_HMAC", symmetricDocument("v1", 32))
	t.Setenv("TEST_RESPONSE_KEK", symmetricDocument("v1", 32))

	for _, test := range []struct {
		name  string
		value *string
	}{
		{name: "missing"},
		{name: "weak", value: stringPointer(symmetricDocument("v1", 16))},
	} {
		t.Run(test.name, func(t *testing.T) {
			if test.value != nil {
				t.Setenv("TEST_CONTROL_HMAC", *test.value)
			}
			_, err := loadDeploymentKeyrings(cfg)
			if err == nil || !strings.Contains(err.Error(), "control-plane HMAC root") {
				t.Fatalf("loadDeploymentKeyrings() error = %v", err)
			}
		})
	}
}

func managedKeyringConfig(t *testing.T) *config.RouterConfig {
	t.Helper()
	cfg := config.DefaultGlobalConfig()
	cfg.ControlPlane.Mode = config.ControlPlaneModeManaged
	cfg.Access.Enabled = false
	cfg.BackendCredentials.ProviderKEKKeyringEnv = "TEST_PROVIDER_KEK"
	cfg.ManagementAPI.Auth.TokenSigningKeyringEnv = "TEST_MANAGEMENT_SIGNING"
	cfg.ManagementAPI.Auth.ServiceAccountHMACKeyringEnv = "TEST_SERVICE_HMAC"
	cfg.ManagementAPI.Auth.InvitationHMACKeyringEnv = "TEST_INVITATION_HMAC"
	cfg.ManagementAPI.Auth.ResponseKEKKeyringEnv = "TEST_RESPONSE_KEK"
	cfg.ManagementAPI.Auth.ControlPlaneHMACKeyringEnv = "TEST_CONTROL_HMAC"
	return &cfg
}

func symmetricDocument(version string, size int) string {
	key := base64.RawURLEncoding.EncodeToString([]byte(strings.Repeat("k", size)))
	return fmt.Sprintf(`{"activeVersion":%q,"keys":[{"version":%q,"key":%q}]}`, version, version, key)
}

func signingDocument(t *testing.T, version string) string {
	t.Helper()
	public, private, err := ed25519.GenerateKey(nil)
	if err != nil {
		t.Fatal(err)
	}
	return fmt.Sprintf(
		`{"activeVersion":%q,"keys":[{"version":%q,"publicKey":%q,"privateKey":%q}]}`,
		version, version,
		base64.RawURLEncoding.EncodeToString(public),
		base64.RawURLEncoding.EncodeToString(private.Seed()),
	)
}

func stringPointer(value string) *string { return &value }
