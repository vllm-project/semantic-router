//go:build !windows && cgo

package apiserver

import (
	"context"
	"crypto/tls"
	"crypto/x509"
	"errors"
	"fmt"
	"io"
	"net"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

const (
	maximumManagementTLSPEMBytes = 1 << 20
	managementTLSValidityMargin  = 5 * time.Minute
	managementTLSReloadInterval  = 15 * time.Second
)

// managementListenerTLS owns the validated, Router-terminated TLS context for
// one managed listener. The certificate is parsed before the socket is opened;
// readiness continues checking its validity so an expiring certificate cannot
// leave a replica advertising ready.
type managementListenerTLS struct {
	references config.ManagementAPITLSConfig
	config     *tls.Config

	mu        sync.RWMutex
	active    *tls.Config
	leaf      *x509.Certificate
	reloadErr error
}

func loadManagementListenerTLS(
	references config.ManagementAPITLSConfig,
	now time.Time,
) (*managementListenerTLS, error) {
	active, leaf, err := parseManagementListenerTLS(references, now)
	if err != nil {
		return nil, err
	}
	listener := &managementListenerTLS{
		references: references,
		active:     active,
		leaf:       leaf,
	}
	listener.config = &tls.Config{
		MinVersion:         tls.VersionTLS13,
		GetConfigForClient: listener.configForClient,
	}
	return listener, nil
}

func parseManagementListenerTLS(
	references config.ManagementAPITLSConfig,
	now time.Time,
) (*tls.Config, *x509.Certificate, error) {
	certificatePEM, err := readManagementTLSSecret(
		"certificate",
		references.CertificateFile,
		references.CertificateEnv,
		false,
	)
	if err != nil {
		return nil, nil, err
	}
	defer zeroManagementTLSBytes(certificatePEM)

	privateKeyPEM, err := readManagementTLSSecret(
		"private key",
		references.PrivateKeyFile,
		references.PrivateKeyEnv,
		true,
	)
	if err != nil {
		return nil, nil, err
	}
	defer zeroManagementTLSBytes(privateKeyPEM)

	pair, err := tls.X509KeyPair(certificatePEM, privateKeyPEM)
	if err != nil {
		return nil, nil, errors.New("management listener TLS certificate and private key are invalid or do not match")
	}
	leaf, err := x509.ParseCertificate(pair.Certificate[0])
	if err != nil {
		return nil, nil, errors.New("management listener TLS certificate is invalid")
	}
	pair.Leaf = leaf
	if err := validateManagementCertificate(pair.Certificate, now); err != nil {
		return nil, nil, err
	}

	tlsConfig := &tls.Config{
		MinVersion:   tls.VersionTLS13,
		Certificates: []tls.Certificate{pair},
	}
	if references.ClientCABundleFile != "" || references.ClientCABundleEnv != "" {
		clientCAPEM, readErr := readManagementTLSSecret(
			"client CA bundle",
			references.ClientCABundleFile,
			references.ClientCABundleEnv,
			false,
		)
		if readErr != nil {
			return nil, nil, readErr
		}
		defer zeroManagementTLSBytes(clientCAPEM)
		clientCAs := x509.NewCertPool()
		if !clientCAs.AppendCertsFromPEM(clientCAPEM) {
			return nil, nil, errors.New("management listener TLS client CA bundle is invalid")
		}
		// Management supports both human/session clients and workload mTLS on the
		// same private listener. A presented certificate is mandatory-valid, but
		// clients without one may still use the explicit exchange endpoints.
		tlsConfig.ClientAuth = tls.VerifyClientCertIfGiven
		tlsConfig.ClientCAs = clientCAs
	}
	if err := validateManagementTLSHandshake(tlsConfig); err != nil {
		return nil, nil, err
	}

	return tlsConfig, leaf, nil
}

func (listener *managementListenerTLS) Ready(now time.Time) error {
	if listener == nil || listener.config == nil {
		return errors.New("management listener TLS is unavailable")
	}
	listener.mu.RLock()
	defer listener.mu.RUnlock()
	if listener.reloadErr != nil {
		return errors.New("management listener TLS reload is unhealthy")
	}
	return validateManagementCertificateValidity(listener.leaf, now)
}

func (listener *managementListenerTLS) Reload(now time.Time) error {
	if listener == nil {
		return errors.New("management listener TLS is unavailable")
	}
	active, leaf, err := parseManagementListenerTLS(listener.references, now)
	listener.mu.Lock()
	defer listener.mu.Unlock()
	if err != nil {
		listener.reloadErr = err
		return err
	}
	listener.active = active
	listener.leaf = leaf
	listener.reloadErr = nil
	return nil
}

func (listener *managementListenerTLS) Watch(ctx context.Context) <-chan struct{} {
	done := make(chan struct{})
	go func() {
		defer close(done)
		ticker := time.NewTicker(managementTLSReloadInterval)
		defer ticker.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case now := <-ticker.C:
				_ = listener.Reload(now)
			}
		}
	}()
	return done
}

func (listener *managementListenerTLS) configForClient(*tls.ClientHelloInfo) (*tls.Config, error) {
	listener.mu.RLock()
	defer listener.mu.RUnlock()
	if listener.active == nil {
		return nil, errors.New("management listener TLS is unavailable")
	}
	return listener.active.Clone(), nil
}

func validateManagementCertificate(chain [][]byte, now time.Time) error {
	if len(chain) == 0 {
		return errors.New("management listener TLS certificate chain is empty")
	}
	certificates := make([]*x509.Certificate, 0, len(chain))
	for _, encoded := range chain {
		certificate, err := x509.ParseCertificate(encoded)
		if err != nil {
			return errors.New("management listener TLS certificate chain is invalid")
		}
		certificates = append(certificates, certificate)
	}
	for _, certificate := range certificates {
		if err := validateManagementCertificateValidity(certificate, now); err != nil {
			return err
		}
	}
	leaf := certificates[0]
	if len(leaf.DNSNames) == 0 && len(leaf.IPAddresses) == 0 {
		return errors.New("management listener TLS certificate requires a DNS or IP subject alternative name")
	}
	if !supportsServerAuthentication(leaf) {
		return errors.New("management listener TLS certificate does not permit server authentication")
	}
	for index := 0; index+1 < len(certificates); index++ {
		if err := certificates[index].CheckSignatureFrom(certificates[index+1]); err != nil {
			return errors.New("management listener TLS certificate chain is invalid")
		}
	}
	return nil
}

func supportsServerAuthentication(certificate *x509.Certificate) bool {
	if certificate == nil || len(certificate.ExtKeyUsage) == 0 {
		return certificate != nil
	}
	for _, usage := range certificate.ExtKeyUsage {
		if usage == x509.ExtKeyUsageAny || usage == x509.ExtKeyUsageServerAuth {
			return true
		}
	}
	return false
}

func validateManagementTLSHandshake(serverConfig *tls.Config) error {
	if serverConfig == nil {
		return errors.New("management listener TLS configuration is unavailable")
	}
	// The client-identity chain is validated independently when the CA bundle is
	// parsed. This local handshake validates the server certificate, private key,
	// and TLS protocol context without requiring a deployment client credential.
	handshakeConfig := serverConfig.Clone()
	handshakeConfig.ClientAuth = tls.NoClientCert
	handshakeConfig.ClientCAs = nil
	serverConnection, clientConnection := net.Pipe()
	deadline := time.Now().Add(time.Second)
	_ = serverConnection.SetDeadline(deadline)
	_ = clientConnection.SetDeadline(deadline)
	serverTLS := tls.Server(serverConnection, handshakeConfig)
	clientTLS := tls.Client(clientConnection, &tls.Config{
		MinVersion: tls.VersionTLS13,
		// This connection never leaves process memory. Certificate identity is
		// validated above; the handshake only proves the server context is usable.
		InsecureSkipVerify: true, //nolint:gosec
	})
	serverResult := make(chan error, 1)
	go func() { serverResult <- serverTLS.Handshake() }()
	clientErr := clientTLS.Handshake()
	serverErr := <-serverResult
	_ = clientConnection.Close()
	_ = serverConnection.Close()
	if clientErr != nil || serverErr != nil {
		return errors.New("management listener TLS loopback handshake failed")
	}
	return nil
}

func validateManagementCertificateValidity(certificate *x509.Certificate, now time.Time) error {
	if certificate == nil || now.Before(certificate.NotBefore) {
		return errors.New("management listener TLS certificate is not valid yet")
	}
	if !now.Add(managementTLSValidityMargin).Before(certificate.NotAfter) {
		return errors.New("management listener TLS certificate is expired or too close to expiry")
	}
	return nil
}

func readManagementTLSSecret(label, file, environment string, ownerOnly bool) ([]byte, error) {
	if (file == "") == (environment == "") {
		return nil, fmt.Errorf("management listener TLS %s requires exactly one file or environment source", label)
	}
	if file != "" {
		handle, err := os.Open(file)
		if err != nil {
			return nil, fmt.Errorf("read Management listener TLS %s file", label)
		}
		defer handle.Close()
		info, err := handle.Stat()
		if err != nil || !info.Mode().IsRegular() {
			return nil, fmt.Errorf("management listener TLS %s source must be a regular file", label)
		}
		if ownerOnly && info.Mode().Perm()&0o077 != 0 {
			return nil, errors.New("management listener TLS private key file permissions are too broad")
		}
		payload, err := io.ReadAll(io.LimitReader(handle, maximumManagementTLSPEMBytes+1))
		if err != nil {
			return nil, fmt.Errorf("read Management listener TLS %s file", label)
		}
		if len(payload) == 0 || len(payload) > maximumManagementTLSPEMBytes {
			zeroManagementTLSBytes(payload)
			return nil, fmt.Errorf("management listener TLS %s is empty or too large", label)
		}
		return payload, nil
	}
	value, found := os.LookupEnv(environment)
	if !found || strings.TrimSpace(value) == "" || len(value) > maximumManagementTLSPEMBytes {
		return nil, fmt.Errorf("management listener TLS %s environment source is unset or invalid", label)
	}
	return []byte(value), nil
}

func zeroManagementTLSBytes(payload []byte) {
	for index := range payload {
		payload[index] = 0
	}
}
