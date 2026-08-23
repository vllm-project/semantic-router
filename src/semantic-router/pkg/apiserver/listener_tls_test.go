//go:build !windows && cgo

package apiserver

import (
	"bufio"
	"context"
	"crypto/ed25519"
	"crypto/rand"
	"crypto/tls"
	"crypto/x509"
	"encoding/pem"
	"io"
	"math/big"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
)

func TestManagedListenerTerminatesTLSAndKeepsRoutesIsolated(t *testing.T) {
	now := time.Now().UTC()
	certificatePEM, privateKeyPEM := generateManagementServerCertificate(t, now, now.Add(time.Hour))
	certificateFile, privateKeyFile := writeManagementTLSFiles(t, certificatePEM, privateKeyPEM)
	port := reserveManagementListenerPort(t)
	cfg := managedListenerTestConfig(port, certificateFile, privateKeyFile)

	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	configPath := filepath.Join(t.TempDir(), "router.yaml")
	listenerStarted := make(chan error, 1)
	serverDone := make(chan error, 1)
	go func() {
		serverDone <- InitWithOptions(InitOptions{
			Context: ctx, OnListenerStart: func(err error) { listenerStarted <- err }, ConfigPath: configPath,
			Port: port, RuntimeRegistry: routerruntime.NewRegistry(&cfg),
			ManagedAPI: &managedAPIStub{},
		})
	}()
	select {
	case err := <-listenerStarted:
		if err != nil {
			t.Fatalf("managed listener startup failed: %v", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("managed listener did not report startup")
	}

	rootCAs := x509.NewCertPool()
	if !rootCAs.AppendCertsFromPEM(certificatePEM) {
		t.Fatal("failed to install test root certificate")
	}
	client := &http.Client{Transport: &http.Transport{TLSClientConfig: &tls.Config{
		MinVersion: tls.VersionTLS13, RootCAs: rootCAs, ServerName: "127.0.0.1",
	}}}
	baseURL := "https://127.0.0.1:" + portString(port)
	response := waitForManagedTLSResponse(t, client, baseURL+"/health", serverDone)
	if response.StatusCode != http.StatusOK {
		response.Body.Close()
		t.Fatalf("TLS health status = %d", response.StatusCode)
	}
	response.Body.Close()

	response, err := client.Get(baseURL + "/config/router")
	if err != nil {
		t.Fatalf("TLS route-isolation request failed: %v", err)
	}
	response.Body.Close()
	if response.StatusCode != http.StatusNotFound {
		t.Fatalf("legacy route over managed TLS status = %d, want 404", response.StatusCode)
	}

	assertManagedListenerRejectsPlaintext(t, port)
	if _, err := tls.Dial("tcp", net.JoinHostPort("127.0.0.1", portString(port)), &tls.Config{
		MaxVersion: tls.VersionTLS12, RootCAs: rootCAs, ServerName: "127.0.0.1",
	}); err == nil {
		t.Fatal("managed listener accepted TLS 1.2")
	}

	cancel()
	select {
	case err := <-serverDone:
		if err != nil {
			t.Fatalf("managed listener shutdown error: %v", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("managed listener did not shut down after context cancellation")
	}
}

func TestStandaloneListenerRemainsPlaintext(t *testing.T) {
	port := reserveManagementListenerPort(t)
	server := &http.Server{
		Addr: net.JoinHostPort("127.0.0.1", portString(port)),
		Handler: http.HandlerFunc(func(response http.ResponseWriter, _ *http.Request) {
			response.WriteHeader(http.StatusNoContent)
		}),
	}
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	serverDone := make(chan error, 1)
	go func() { serverDone <- serveManagementListener(ctx, server, false, nil) }()
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		response, err := http.Get("http://" + server.Addr)
		if err == nil {
			response.Body.Close()
			if response.StatusCode != http.StatusNoContent {
				t.Fatalf("standalone plaintext status = %d", response.StatusCode)
			}
			cancel()
			select {
			case err := <-serverDone:
				if err != nil {
					t.Fatalf("standalone shutdown error: %v", err)
				}
			case <-time.After(5 * time.Second):
				t.Fatal("standalone listener did not shut down")
			}
			return
		}
		time.Sleep(20 * time.Millisecond)
	}
	t.Fatal("standalone listener did not accept plaintext")
}

func TestManagedListenerRejectsInvalidCertificateBeforeBinding(t *testing.T) {
	directory := t.TempDir()
	certificateFile := filepath.Join(directory, "certificate.pem")
	privateKeyFile := filepath.Join(directory, "private-key.pem")
	if err := os.WriteFile(certificateFile, []byte("not a certificate"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(privateKeyFile, []byte("not a private key"), 0o600); err != nil {
		t.Fatal(err)
	}
	cfg := managedListenerTestConfig(reserveManagementListenerPort(t), certificateFile, privateKeyFile)
	err := InitWithOptions(InitOptions{
		RuntimeRegistry: routerruntime.NewRegistry(&cfg), ManagedAPI: &managedAPIStub{},
	})
	if err == nil || !strings.Contains(err.Error(), "invalid or do not match") {
		t.Fatalf("invalid certificate startup error = %v", err)
	}
}

func TestManagedListenerTLSRejectsMissingEnvironmentMaterial(t *testing.T) {
	const certificateEnv = "VLLM_SR_TEST_MISSING_MANAGEMENT_CERTIFICATE"
	const privateKeyEnv = "VLLM_SR_TEST_MISSING_MANAGEMENT_PRIVATE_KEY"
	_ = os.Unsetenv(certificateEnv)
	_ = os.Unsetenv(privateKeyEnv)
	_, err := loadManagementListenerTLS(config.ManagementAPITLSConfig{
		CertificateEnv: certificateEnv, PrivateKeyEnv: privateKeyEnv,
	}, time.Now())
	if err == nil || !strings.Contains(err.Error(), "environment source is unset") {
		t.Fatalf("missing TLS environment error = %v", err)
	}
}

func TestManagedListenerRejectsMissingEnvironmentMaterialBeforeBinding(t *testing.T) {
	const certificateEnv = "VLLM_SR_TEST_MISSING_STARTUP_CERTIFICATE"
	const privateKeyEnv = "VLLM_SR_TEST_MISSING_STARTUP_PRIVATE_KEY"
	_ = os.Unsetenv(certificateEnv)
	_ = os.Unsetenv(privateKeyEnv)
	cfg := managedListenerTestConfig(reserveManagementListenerPort(t), "", "")
	cfg.ManagementAPI.TLS.CertificateEnv = certificateEnv
	cfg.ManagementAPI.TLS.PrivateKeyEnv = privateKeyEnv
	err := InitWithOptions(InitOptions{
		RuntimeRegistry: routerruntime.NewRegistry(&cfg), ManagedAPI: &managedAPIStub{},
	})
	if err == nil || !strings.Contains(err.Error(), "environment source is unset") {
		t.Fatalf("missing TLS startup material error = %v", err)
	}
}

func TestManagedListenerTLSRejectsBroadPrivateKeyPermissions(t *testing.T) {
	now := time.Now().UTC()
	certificatePEM, privateKeyPEM := generateManagementServerCertificate(t, now, now.Add(time.Hour))
	certificateFile, privateKeyFile := writeManagementTLSFiles(t, certificatePEM, privateKeyPEM)
	if err := os.Chmod(privateKeyFile, 0o640); err != nil {
		t.Fatal(err)
	}
	_, err := loadManagementListenerTLS(config.ManagementAPITLSConfig{
		CertificateFile: certificateFile, PrivateKeyFile: privateKeyFile,
	}, now)
	if err == nil || !strings.Contains(err.Error(), "permissions are too broad") {
		t.Fatalf("broad private-key permissions error = %v", err)
	}
}

func TestManagedListenerTLSClientCABundleVerifiesOptionalClientCertificate(t *testing.T) {
	now := time.Now().UTC()
	certificatePEM, privateKeyPEM := generateManagementServerCertificate(t, now, now.Add(time.Hour))
	certificateFile, privateKeyFile := writeManagementTLSFiles(t, certificatePEM, privateKeyPEM)
	clientCAFile := filepath.Join(t.TempDir(), "client-ca.pem")
	if err := os.WriteFile(clientCAFile, certificatePEM, 0o600); err != nil {
		t.Fatal(err)
	}
	listenerTLS, err := loadManagementListenerTLS(config.ManagementAPITLSConfig{
		CertificateFile: certificateFile, PrivateKeyFile: privateKeyFile,
		ClientCABundleFile: clientCAFile,
	}, now)
	if err != nil {
		t.Fatalf("loadManagementListenerTLS() error = %v", err)
	}
	active, err := listenerTLS.configForClient(nil)
	if err != nil {
		t.Fatal(err)
	}
	if active.ClientAuth != tls.VerifyClientCertIfGiven || active.ClientCAs == nil {
		t.Fatal("configured client CA did not enable optional verified mTLS")
	}
}

func TestManagedListenerTLSReadinessFailsBeforeCertificateExpiry(t *testing.T) {
	listenerTLS := &managementListenerTLS{
		config: &tls.Config{MinVersion: tls.VersionTLS13},
		leaf: &x509.Certificate{
			NotBefore: time.Now().Add(-time.Hour),
			NotAfter:  time.Now().Add(managementTLSValidityMargin / 2),
		},
	}
	if err := listenerTLS.Ready(time.Now()); err == nil {
		t.Fatal("TLS readiness should fail inside the certificate validity margin")
	}
}

func TestManagedListenerTLSReloadRetainsLastCertificateAndRecoversReadiness(t *testing.T) {
	now := time.Now().UTC()
	certificatePEM, privateKeyPEM := generateManagementServerCertificate(t, now, now.Add(time.Hour))
	certificateFile, privateKeyFile := writeManagementTLSFiles(t, certificatePEM, privateKeyPEM)
	listenerTLS, testManagedListenerTLSReloadRetainsLastCertificateAndRecoversReadinessErr := loadManagementListenerTLS(config.ManagementAPITLSConfig{
		CertificateFile: certificateFile, PrivateKeyFile: privateKeyFile,
	}, now)
	if testManagedListenerTLSReloadRetainsLastCertificateAndRecoversReadinessErr != nil {
		t.Fatal(testManagedListenerTLSReloadRetainsLastCertificateAndRecoversReadinessErr)
	}
	activeBefore, testManagedListenerTLSReloadRetainsLastCertificateAndRecoversReadinessErr := listenerTLS.configForClient(nil)
	if testManagedListenerTLSReloadRetainsLastCertificateAndRecoversReadinessErr != nil {
		t.Fatal(testManagedListenerTLSReloadRetainsLastCertificateAndRecoversReadinessErr)
	}
	if err := os.WriteFile(privateKeyFile, []byte("invalid replacement"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := listenerTLS.Reload(now.Add(time.Minute)); err == nil {
		t.Fatal("invalid atomic replacement should fail reload")
	}
	if err := listenerTLS.Ready(now.Add(time.Minute)); err == nil {
		t.Fatal("failed TLS reload should make readiness fail closed")
	}
	activeAfter, testManagedListenerTLSReloadRetainsLastCertificateAndRecoversReadinessErr := listenerTLS.configForClient(nil)
	if testManagedListenerTLSReloadRetainsLastCertificateAndRecoversReadinessErr != nil || len(activeAfter.Certificates) != 1 ||
		activeAfter.Certificates[0].Leaf != activeBefore.Certificates[0].Leaf {
		t.Fatal("failed TLS reload did not retain the last valid certificate")
	}
	if err := os.WriteFile(privateKeyFile, privateKeyPEM, 0o600); err != nil {
		t.Fatal(err)
	}
	if err := listenerTLS.Reload(now.Add(2 * time.Minute)); err != nil {
		t.Fatalf("valid TLS replacement did not recover: %v", err)
	}
	if err := listenerTLS.Ready(now.Add(2 * time.Minute)); err != nil {
		t.Fatalf("TLS readiness did not recover after valid replacement: %v", err)
	}
}

func generateManagementServerCertificate(t *testing.T, notBefore, notAfter time.Time) ([]byte, []byte) {
	t.Helper()
	publicKey, privateKey, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	template := &x509.Certificate{
		SerialNumber: big.NewInt(1),
		NotBefore:    notBefore.Add(-time.Minute), NotAfter: notAfter,
		KeyUsage:    x509.KeyUsageDigitalSignature,
		ExtKeyUsage: []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
		IPAddresses: []net.IP{net.ParseIP("127.0.0.1")},
	}
	encodedCertificate, err := x509.CreateCertificate(rand.Reader, template, template, publicKey, privateKey)
	if err != nil {
		t.Fatal(err)
	}
	encodedPrivateKey, err := x509.MarshalPKCS8PrivateKey(privateKey)
	if err != nil {
		t.Fatal(err)
	}
	return pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: encodedCertificate}),
		pem.EncodeToMemory(&pem.Block{Type: "PRIVATE KEY", Bytes: encodedPrivateKey})
}

func writeManagementTLSFiles(t *testing.T, certificatePEM, privateKeyPEM []byte) (string, string) {
	t.Helper()
	directory := t.TempDir()
	certificateFile := filepath.Join(directory, "certificate.pem")
	privateKeyFile := filepath.Join(directory, "private-key.pem")
	if err := os.WriteFile(certificateFile, certificatePEM, 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(privateKeyFile, privateKeyPEM, 0o600); err != nil {
		t.Fatal(err)
	}
	return certificateFile, privateKeyFile
}

func managedListenerTestConfig(port int, certificateFile, privateKeyFile string) config.RouterConfig {
	cfg := config.DefaultGlobalConfig()
	cfg.ControlPlane.Mode = config.ControlPlaneModeManaged
	cfg.ManagementAPI.BindAddress = "127.0.0.1"
	cfg.ManagementAPI.Port = port
	cfg.ManagementAPI.Auth.Mode = config.ManagementAuthModeRouter
	cfg.ManagementAPI.Auth.Roles = nil
	cfg.ManagementAPI.Auth.Tokens = nil
	cfg.ManagementAPI.Auth.TokenSigningKeyringFile = "/unused/management-signing"
	cfg.ManagementAPI.Auth.ServiceAccountHMACKeyringFile = "/unused/service-account-hmac"
	cfg.ManagementAPI.Auth.InvitationHMACKeyringFile = "/unused/invitation-hmac"
	cfg.ManagementAPI.Auth.ControlPlaneHMACKeyringFile = "/unused/control-plane-hmac"
	cfg.ManagementAPI.Auth.ResponseKEKKeyringFile = "/unused/response-kek"
	cfg.ManagementAPI.TLS.CertificateFile = certificateFile
	cfg.ManagementAPI.TLS.PrivateKeyFile = privateKeyFile
	return cfg
}

func reserveManagementListenerPort(t *testing.T) int {
	t.Helper()
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer listener.Close()
	return listener.Addr().(*net.TCPAddr).Port
}

func waitForManagedTLSResponse(
	t *testing.T,
	client *http.Client,
	url string,
	serverDone <-chan error,
) *http.Response {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		select {
		case err := <-serverDone:
			t.Fatalf("managed listener exited before accepting TLS: %v", err)
		default:
		}
		response, err := client.Get(url)
		if err == nil {
			return response
		}
		time.Sleep(20 * time.Millisecond)
	}
	t.Fatal("managed listener did not accept TLS before deadline")
	return nil
}

func assertManagedListenerRejectsPlaintext(t *testing.T, port int) {
	t.Helper()
	connection, assertManagedListenerRejectsPlaintextErr := net.DialTimeout("tcp", net.JoinHostPort("127.0.0.1", portString(port)), time.Second)
	if assertManagedListenerRejectsPlaintextErr != nil {
		t.Fatalf("connect plaintext client: %v", assertManagedListenerRejectsPlaintextErr)
	}
	defer connection.Close()
	_ = connection.SetDeadline(time.Now().Add(time.Second))
	if _, err := io.WriteString(connection, "GET /health HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n"); err != nil {
		t.Fatalf("write plaintext request: %v", err)
	}
	response, assertManagedListenerRejectsPlaintextErr := http.ReadResponse(bufio.NewReader(connection), nil)
	if assertManagedListenerRejectsPlaintextErr == nil {
		defer response.Body.Close()
		if response.StatusCode >= 200 && response.StatusCode < 300 {
			t.Fatalf("managed listener served plaintext with status %d", response.StatusCode)
		}
	}
}

func portString(port int) string {
	return strconv.Itoa(port)
}
