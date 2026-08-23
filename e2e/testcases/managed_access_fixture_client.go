package testcases

import (
	"context"
	"crypto/ed25519"
	"crypto/tls"
	"crypto/x509"
	"fmt"
	"net"
	"net/http"
	"runtime"
	"strconv"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"

	"github.com/vllm-project/semantic-router/e2e/pkg/helpers"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

type managedAccessIdentityMaterial struct {
	claims     managedAccessDashboardClaims
	privateKey ed25519.PrivateKey
	rootCAs    *x509.CertPool
}

func openManagedAccessClient(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
	dashboardClient *http.Client,
	dashboardURL string,
) (*managedAccessClient, func(), error) {
	identity, err := loadManagedAccessIdentity(ctx, client, dashboardClient, dashboardURL, opts.Verbose)
	if err != nil {
		return nil, nil, err
	}
	defer clear(identity.privateKey)

	managementClient, cleanup, err := openManagementTransport(ctx, client, opts, identity.rootCAs)
	if err != nil {
		return nil, nil, err
	}
	if err := exchangeManagementToken(ctx, managementClient, identity); err != nil {
		cleanup()
		return nil, nil, err
	}
	return managementClient, cleanup, nil
}

func loadManagedAccessIdentity(
	ctx context.Context,
	client *kubernetes.Clientset,
	dashboardClient *http.Client,
	dashboardURL string,
	verbose bool,
) (managedAccessIdentityMaterial, error) {
	dashboardToken, err := dashboardAuthToken(ctx, dashboardClient, dashboardURL, verbose)
	if err != nil {
		return managedAccessIdentityMaterial{}, fmt.Errorf("authenticate managed-access fixture principal: %w", err)
	}
	claims, err := parseManagedAccessDashboardClaims(dashboardToken)
	if err != nil {
		return managedAccessIdentityMaterial{}, err
	}
	secret, err := client.CoreV1().Secrets(managedAccessNamespace).Get(
		ctx, managedAccessIdentitySecret, metav1.GetOptions{},
	)
	if err != nil {
		return managedAccessIdentityMaterial{}, fmt.Errorf("read managed-access E2E trust material: %w", err)
	}
	privateKey, err := parseManagedAccessAssertionKey(secret.Data[managedAccessAssertionKey])
	if err != nil {
		return managedAccessIdentityMaterial{}, err
	}
	rootCAs := x509.NewCertPool()
	if !rootCAs.AppendCertsFromPEM(secret.Data[managedAccessCABundle]) {
		clear(privateKey)
		return managedAccessIdentityMaterial{}, fmt.Errorf("managed-access E2E CA bundle is invalid")
	}
	return managedAccessIdentityMaterial{claims: claims, privateKey: privateKey, rootCAs: rootCAs}, nil
}

func openManagementTransport(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
	rootCAs *x509.CertPool,
) (*managedAccessClient, func(), error) {
	localPort, err := managedAccessAvailablePort()
	if err != nil {
		return nil, nil, err
	}
	stop, err := helpers.StartPortForward(
		ctx, client, opts.RestConfig, managedAccessNamespace, managedAccessService,
		localPort+":"+managedAccessServicePort, opts.Verbose,
	)
	if err != nil {
		return nil, nil, fmt.Errorf("open direct Management API connection: %w", err)
	}
	transport := &http.Transport{TLSClientConfig: &tls.Config{
		MinVersion: tls.VersionTLS13,
		RootCAs:    rootCAs,
		ServerName: managedAccessServerName,
	}}
	managementClient := &managedAccessClient{
		baseURL: "https://127.0.0.1:" + localPort,
		client:  &http.Client{Timeout: 45 * time.Second, Transport: transport},
		verbose: opts.Verbose,
	}
	cleanup := func() {
		managementClient.token = ""
		transport.CloseIdleConnections()
		stop()
	}
	return managementClient, cleanup, nil
}

func exchangeManagementToken(
	ctx context.Context,
	client *managedAccessClient,
	identity managedAccessIdentityMaterial,
) error {
	var challenge struct {
		ExchangeChallengeID string `json:"exchangeChallengeId"`
		Nonce               string `json:"nonce"`
	}
	if _, _, err := client.request(
		ctx, "", http.MethodPost, "/auth/exchange-challenges", "",
		map[string]string{"issuerId": managedAccessIssuerID}, nil,
		[]int{http.StatusCreated}, &challenge,
	); err != nil {
		return fmt.Errorf("create direct Management exchange challenge: %w", err)
	}
	if challenge.ExchangeChallengeID == "" || challenge.Nonce == "" {
		return fmt.Errorf("direct Management exchange challenge is incomplete")
	}
	assertion, err := signManagedAccessAssertion(identity.privateKey, identity.claims, challenge.Nonce)
	if err != nil {
		return err
	}
	var exchanged struct {
		AccessToken string `json:"accessToken"`
		TokenType   string `json:"tokenType"`
		ExpiresIn   int64  `json:"expiresIn"`
	}
	if _, _, err := client.request(
		ctx, "", http.MethodPost, "/auth/token-exchange", "",
		map[string]string{
			"issuerId": managedAccessIssuerID, "exchangeChallengeId": challenge.ExchangeChallengeID,
			"subjectToken": assertion, "subjectTokenType": "router_local_assertion",
		}, nil, []int{http.StatusOK}, &exchanged,
	); err != nil {
		return fmt.Errorf("exchange direct Management token: %w", err)
	}
	assertion = ""
	runtime.KeepAlive(assertion)
	if exchanged.TokenType != "Bearer" || exchanged.AccessToken == "" || exchanged.ExpiresIn <= 0 {
		return fmt.Errorf("direct Management token exchange returned an invalid envelope")
	}
	client.token = exchanged.AccessToken
	exchanged.AccessToken = ""
	return nil
}

func managedAccessAvailablePort() (string, error) {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return "", fmt.Errorf("allocate direct Management port: %w", err)
	}
	port := listener.Addr().(*net.TCPAddr).Port
	if err := listener.Close(); err != nil {
		return "", fmt.Errorf("release direct Management port: %w", err)
	}
	return strconv.Itoa(port), nil
}
