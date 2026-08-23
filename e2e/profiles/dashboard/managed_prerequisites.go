package dashboard

import (
	"context"
	"crypto/ecdsa"
	"crypto/ed25519"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/base64"
	"encoding/json"
	"encoding/pem"
	"errors"
	"fmt"
	"math/big"
	"time"

	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"

	"github.com/vllm-project/semantic-router/e2e/pkg/framework"
	"github.com/vllm-project/semantic-router/e2e/pkg/helpers"
)

const (
	dashboardRouterSecretName   = "semantic-router-dashboard-e2e-router"
	dashboardIdentitySecretName = "semantic-router-dashboard-e2e-dashboard"
	dashboardStoreSecretName    = "semantic-router-dashboard-e2e-store"
	dashboardBootstrapSeedName  = "semantic-router-dashboard-e2e-bootstrap-seed"
	dashboardBootstrapWriter    = "semantic-router-dashboard-e2e-bootstrap-writer"

	dashboardPostgresDeployment         = "dashboard-e2e-postgres"
	dashboardValkeyDeployment           = "dashboard-e2e-valkey"
	dashboardPublicInferenceServiceName = "semantic-router-public"

	dashboardIssuerDNS  = "semantic-router-dashboard-issuer.vllm-semantic-router-system.svc.cluster.local"
	routerManagementDNS = "semantic-router-management.vllm-semantic-router-system.svc.cluster.local"
)

var e2eManagedLabels = map[string]string{
	"app.kubernetes.io/part-of":   "semantic-router",
	"app.kubernetes.io/component": "dashboard-e2e",
	"vllm.ai/e2e-managed":         "true",
}

type dashboardManagedMaterial struct {
	routerSecret    map[string][]byte
	dashboardSecret map[string][]byte
	storeSecret     map[string][]byte
	bootstrapToken  []byte
}

type encodedKeyring struct {
	ActiveVersion string                `json:"activeVersion"`
	Keys          []encodedKeyringEntry `json:"keys"`
}

type encodedKeyringEntry struct {
	Version    string `json:"version"`
	Key        string `json:"key,omitempty"`
	PrivateKey string `json:"privateKey,omitempty"`
	PublicKey  string `json:"publicKey,omitempty"`
}

func (p *Profile) prepareManagedPrerequisites(ctx context.Context, opts *framework.SetupOptions) error {
	if opts == nil || opts.KubeClient == nil || opts.ImageTag == "" {
		return errors.New("Kubernetes client and Router image tag are required")
	}
	if err := ensureDashboardNamespace(ctx, opts); err != nil {
		return err
	}
	material, err := newDashboardManagedMaterial()
	if err != nil {
		return fmt.Errorf("generate ephemeral trust material: %w", err)
	}
	defer zeroManagedMaterial(&material)

	if err := upsertE2ESecret(
		ctx,
		opts,
		newE2ESecret(namespaceRouter, dashboardRouterSecretName, material.routerSecret, false),
	); err != nil {
		return err
	}
	if err := upsertE2ESecret(ctx, opts, newDashboardIdentitySecret(material.dashboardSecret)); err != nil {
		return err
	}
	if err := upsertE2ESecret(
		ctx,
		opts,
		newE2ESecret("default", dashboardStoreSecretName, material.storeSecret, false),
	); err != nil {
		return err
	}
	if err := resetDashboardDataClaim(ctx, opts); err != nil {
		return err
	}
	if err := stageBootstrapToken(ctx, opts, material.bootstrapToken); err != nil {
		return err
	}

	if err := p.kubectlApplyWithNamespace(ctx, opts.KubeConfig, "default", managedStoresManifest); err != nil {
		return fmt.Errorf("apply managed stores: %w", err)
	}
	for _, deployment := range []string{dashboardPostgresDeployment, dashboardValkeyDeployment} {
		if err := helpers.WaitForDeploymentReady(
			ctx, opts.KubeClient, "default", deployment, 5*time.Minute, 2*time.Second, opts.Verbose,
		); err != nil {
			return fmt.Errorf("wait for %s: %w", deployment, err)
		}
	}
	return nil
}

// preparePublicInferenceFrontDoor gives managed Router replicas one stable
// in-cluster name for their ordinary public Envoy listener. The Agent worker
// uses this path with a delegated API key, so access, quota, logs, and usage
// remain identical to direct public API calls.
func (p *Profile) preparePublicInferenceFrontDoor(ctx context.Context, opts *framework.SetupOptions) error {
	if opts == nil || opts.KubeClient == nil {
		return errors.New("Kubernetes client is required")
	}
	serviceConfig := p.stack.ServiceConfig()
	targetName := serviceConfig.Name
	if targetName == "" {
		matches, err := opts.KubeClient.CoreV1().Services(serviceConfig.Namespace).List(
			ctx, metav1.ListOptions{LabelSelector: serviceConfig.LabelSelector},
		)
		if err != nil {
			return fmt.Errorf("list public inference Gateway Services: %w", err)
		}
		targetName, err = uniquePublicInferenceServiceName(
			matches.Items, serviceConfig.Namespace, serviceConfig.LabelSelector,
		)
		if err != nil {
			return err
		}
	}
	desired := &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name: dashboardPublicInferenceServiceName, Namespace: namespaceRouter, Labels: e2eManagedLabels,
		},
		Spec: corev1.ServiceSpec{
			Type:         corev1.ServiceTypeExternalName,
			ExternalName: fmt.Sprintf("%s.%s.svc.cluster.local", targetName, serviceConfig.Namespace),
		},
	}
	services := opts.KubeClient.CoreV1().Services(namespaceRouter)
	existing, err := services.Get(ctx, desired.Name, metav1.GetOptions{})
	if apierrors.IsNotFound(err) {
		_, err = services.Create(ctx, desired, metav1.CreateOptions{})
	} else if err == nil {
		if existing.Labels["vllm.ai/e2e-managed"] != "true" {
			return fmt.Errorf("refuse to replace unmanaged Service %s/%s", namespaceRouter, desired.Name)
		}
		desired.ResourceVersion = existing.ResourceVersion
		_, err = services.Update(ctx, desired, metav1.UpdateOptions{})
	}
	if err != nil {
		return fmt.Errorf("create stable public inference Service: %w", err)
	}
	return nil
}

func uniquePublicInferenceServiceName(services []corev1.Service, namespace, selector string) (string, error) {
	if len(services) != 1 || services[0].Name == "" {
		return "", fmt.Errorf(
			"public inference selector %q in namespace %q resolved %d Services; require exactly one",
			selector, namespace, len(services),
		)
	}
	return services[0].Name, nil
}

func ensureDashboardNamespace(ctx context.Context, opts *framework.SetupOptions) error {
	_, err := opts.KubeClient.CoreV1().Namespaces().Create(ctx, &corev1.Namespace{
		ObjectMeta: metav1.ObjectMeta{Name: namespaceRouter, Labels: e2eManagedLabels},
	}, metav1.CreateOptions{})
	if err != nil && !apierrors.IsAlreadyExists(err) {
		return fmt.Errorf("create Router namespace: %w", err)
	}
	return nil
}

func upsertE2ESecret(
	ctx context.Context,
	opts *framework.SetupOptions,
	desired *corev1.Secret,
) error {
	namespace := desired.Namespace
	name := desired.Name
	secrets := opts.KubeClient.CoreV1().Secrets(namespace)
	existing, err := secrets.Get(ctx, name, metav1.GetOptions{})
	if apierrors.IsNotFound(err) {
		_, err = secrets.Create(ctx, desired, metav1.CreateOptions{})
	} else if err == nil {
		if validationErr := validateE2ESecretReplacement(existing); validationErr != nil {
			return validationErr
		}
		desired.ResourceVersion = existing.ResourceVersion
		_, err = secrets.Update(ctx, desired, metav1.UpdateOptions{})
	}
	if err != nil {
		return fmt.Errorf("store ephemeral Secret %s/%s: %w", namespace, name, err)
	}
	return nil
}

func validateE2ESecretReplacement(existing *corev1.Secret) error {
	if existing.Labels["vllm.ai/e2e-managed"] != "true" {
		return fmt.Errorf("refuse to replace unmanaged Secret %s/%s", existing.Namespace, existing.Name)
	}
	if existing.Immutable != nil && *existing.Immutable {
		return fmt.Errorf(
			"managed Secret %s/%s is immutable; rerun the profile in a clean E2E cluster",
			existing.Namespace,
			existing.Name,
		)
	}
	return nil
}

func newE2ESecret(namespace, name string, data map[string][]byte, immutable bool) *corev1.Secret {
	return &corev1.Secret{
		ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: namespace, Labels: e2eManagedLabels},
		Type:       corev1.SecretTypeOpaque,
		Data:       cloneSecretData(data),
		Immutable:  boolPointer(immutable),
	}
}

func newDashboardIdentitySecret(data map[string][]byte) *corev1.Secret {
	return newE2ESecret(namespaceRouter, dashboardIdentitySecretName, data, true)
}

func resetDashboardDataClaim(ctx context.Context, opts *framework.SetupOptions) error {
	claims := opts.KubeClient.CoreV1().PersistentVolumeClaims(namespaceRouter)
	existing, err := claims.Get(ctx, "semantic-router-dashboard-e2e-data", metav1.GetOptions{})
	if err == nil {
		if existing.Labels["vllm.ai/e2e-managed"] != "true" {
			return errors.New("refuse to replace unmanaged Dashboard E2E data claim")
		}
		if err := claims.Delete(ctx, existing.Name, metav1.DeleteOptions{}); err != nil && !apierrors.IsNotFound(err) {
			return fmt.Errorf("delete stale Dashboard E2E data claim: %w", err)
		}
		if err := wait.PollUntilContextTimeout(ctx, time.Second, 2*time.Minute, true, func(ctx context.Context) (bool, error) {
			_, getErr := claims.Get(ctx, existing.Name, metav1.GetOptions{})
			switch {
			case apierrors.IsNotFound(getErr):
				return true, nil
			case getErr != nil:
				return false, getErr
			default:
				return false, nil
			}
		}); err != nil {
			return fmt.Errorf("wait for stale Dashboard E2E data claim deletion: %w", err)
		}
	} else if !apierrors.IsNotFound(err) {
		return fmt.Errorf("inspect Dashboard E2E data claim: %w", err)
	}
	storageClass := "standard"
	_, err = claims.Create(ctx, &corev1.PersistentVolumeClaim{
		ObjectMeta: metav1.ObjectMeta{
			Name: "semantic-router-dashboard-e2e-data", Namespace: namespaceRouter, Labels: e2eManagedLabels,
		},
		Spec: corev1.PersistentVolumeClaimSpec{
			StorageClassName: &storageClass,
			AccessModes:      []corev1.PersistentVolumeAccessMode{corev1.ReadWriteOnce},
			Resources: corev1.VolumeResourceRequirements{Requests: corev1.ResourceList{
				corev1.ResourceStorage: resource.MustParse("128Mi"),
			}},
		},
	}, metav1.CreateOptions{})
	if err != nil {
		return fmt.Errorf("create Dashboard E2E data claim: %w", err)
	}
	return nil
}

func stageBootstrapToken(ctx context.Context, opts *framework.SetupOptions, token []byte) error {
	if err := upsertE2ESecret(
		ctx,
		opts,
		newE2ESecret(
			namespaceRouter,
			dashboardBootstrapSeedName,
			map[string][]byte{"router-token": append([]byte(nil), token...)},
			false,
		),
	); err != nil {
		return err
	}
	pods := opts.KubeClient.CoreV1().Pods(namespaceRouter)
	if err := pods.Delete(ctx, dashboardBootstrapWriter, metav1.DeleteOptions{}); err != nil && !apierrors.IsNotFound(err) {
		return fmt.Errorf("delete stale bootstrap writer: %w", err)
	}
	if err := wait.PollUntilContextTimeout(ctx, 250*time.Millisecond, 30*time.Second, true, func(ctx context.Context) (bool, error) {
		_, getErr := pods.Get(ctx, dashboardBootstrapWriter, metav1.GetOptions{})
		switch {
		case apierrors.IsNotFound(getErr):
			return true, nil
		case getErr != nil:
			return false, getErr
		default:
			return false, nil
		}
	}); err != nil {
		return fmt.Errorf("wait for stale bootstrap writer deletion: %w", err)
	}
	root := int64(0)
	writer := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: dashboardBootstrapWriter, Namespace: namespaceRouter, Labels: e2eManagedLabels,
		},
		Spec: corev1.PodSpec{
			RestartPolicy: corev1.RestartPolicyNever,
			Containers: []corev1.Container{{
				Name: "writer", Image: "ghcr.io/vllm-project/semantic-router/extproc:" + opts.ImageTag,
				ImagePullPolicy: corev1.PullNever,
				Command:         []string{"/usr/bin/bash", "-ceu"},
				Args: []string{
					"install -d -m 0700 -o 65532 -g 65532 /shared/bootstrap\n" +
						"install -m 0600 -o 65532 -g 65532 /source/router-token /shared/bootstrap/router-token",
				},
				SecurityContext: &corev1.SecurityContext{RunAsUser: &root, RunAsNonRoot: boolPointer(false)},
				VolumeMounts: []corev1.VolumeMount{
					{Name: "source", MountPath: "/source", ReadOnly: true},
					{Name: "shared", MountPath: "/shared"},
				},
			}},
			Volumes: []corev1.Volume{
				{Name: "source", VolumeSource: corev1.VolumeSource{Secret: &corev1.SecretVolumeSource{
					SecretName: dashboardBootstrapSeedName,
				}}},
				{Name: "shared", VolumeSource: corev1.VolumeSource{PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{
					ClaimName: "semantic-router-dashboard-e2e-data",
				}}},
			},
		},
	}
	if _, err := pods.Create(ctx, writer, metav1.CreateOptions{}); err != nil {
		return fmt.Errorf("create bootstrap writer: %w", err)
	}
	err := wait.PollUntilContextTimeout(ctx, time.Second, 3*time.Minute, true, func(ctx context.Context) (bool, error) {
		pod, getErr := pods.Get(ctx, dashboardBootstrapWriter, metav1.GetOptions{})
		if getErr != nil {
			return false, getErr
		}
		switch pod.Status.Phase {
		case corev1.PodSucceeded:
			return true, nil
		case corev1.PodFailed:
			return false, fmt.Errorf("bootstrap writer failed: %s", pod.Status.Message)
		default:
			return false, nil
		}
	})
	if err != nil {
		return err
	}
	_ = pods.Delete(ctx, dashboardBootstrapWriter, metav1.DeleteOptions{})
	_ = opts.KubeClient.CoreV1().Secrets(namespaceRouter).Delete(
		ctx, dashboardBootstrapSeedName, metav1.DeleteOptions{},
	)
	return nil
}

func newDashboardManagedMaterial() (dashboardManagedMaterial, error) {
	caCertificate, caKey, caPEM, err := newE2ECA()
	if err != nil {
		return dashboardManagedMaterial{}, err
	}
	routerCert, routerKey, err := newE2EServerCertificate(caCertificate, caKey, []string{
		"semantic-router-management", "semantic-router-management.vllm-semantic-router-system",
		"semantic-router-management.vllm-semantic-router-system.svc", routerManagementDNS,
	})
	if err != nil {
		return dashboardManagedMaterial{}, err
	}
	dashboardCert, dashboardKey, err := newE2EServerCertificate(caCertificate, caKey, []string{
		"semantic-router-dashboard-issuer", "semantic-router-dashboard-issuer.vllm-semantic-router-system",
		"semantic-router-dashboard-issuer.vllm-semantic-router-system.svc", dashboardIssuerDNS,
	})
	if err != nil {
		return dashboardManagedMaterial{}, err
	}
	dashboardAssertion, err := newDashboardAssertionKey()
	if err != nil {
		return dashboardManagedMaterial{}, err
	}
	postgresPassword, err := randomURLToken(32)
	if err != nil {
		return dashboardManagedMaterial{}, err
	}
	bootstrapToken, err := randomURLToken(48)
	if err != nil {
		return dashboardManagedMaterial{}, err
	}
	jwtSecret, err := randomURLToken(48)
	if err != nil {
		return dashboardManagedMaterial{}, err
	}
	dsn := "postgres://router:" + postgresPassword + "@dashboard-e2e-postgres.default.svc.cluster.local:5432/vsr?sslmode=disable"

	router := map[string][]byte{
		"VLLM_SR_ACCESS_DATABASE_URL":        []byte(dsn),
		"VLLM_SR_ACCESS_REDIS_URL":           []byte("redis://dashboard-e2e-valkey.default.svc.cluster.local:6379/0"),
		"VLLM_SR_MANAGEMENT_TLS_CERTIFICATE": routerCert,
		"VLLM_SR_MANAGEMENT_TLS_PRIVATE_KEY": routerKey,
		"ca.crt":                             caPEM,
	}
	for _, environment := range []string{
		"VLLM_SR_API_KEY_HMAC_KEYRING", "VLLM_SR_DELEGATION_HMAC_KEYRING",
		"VLLM_SR_PROVIDER_KEK_KEYRING", "VLLM_SR_SERVICE_ACCOUNT_HMAC_KEYRING",
		"VLLM_SR_INVITATION_HMAC_KEYRING", "VLLM_SR_CONTROL_PLANE_HMAC_KEYRING",
		"VLLM_SR_RESPONSE_KEK_KEYRING",
	} {
		value, keyErr := newSymmetricKeyring()
		if keyErr != nil {
			return dashboardManagedMaterial{}, keyErr
		}
		router[environment] = value
	}
	for _, environment := range []string{
		"VLLM_SR_TENANT_CONTEXT_KEYRING", "VLLM_SR_MANAGEMENT_TOKEN_KEYRING",
	} {
		value, keyErr := newSigningKeyring()
		if keyErr != nil {
			return dashboardManagedMaterial{}, keyErr
		}
		router[environment] = value
	}
	return dashboardManagedMaterial{
		routerSecret: router,
		dashboardSecret: map[string][]byte{
			"jwt-secret":                []byte(jwtSecret),
			"assertion-signing-key.pem": dashboardAssertion,
			"tls.crt":                   dashboardCert,
			"tls.key":                   dashboardKey,
			"ca.crt":                    caPEM,
		},
		storeSecret:    map[string][]byte{"postgres-password": []byte(postgresPassword)},
		bootstrapToken: []byte(bootstrapToken),
	}, nil
}

func newSymmetricKeyring() ([]byte, error) {
	key := make([]byte, 32)
	if _, err := rand.Read(key); err != nil {
		return nil, err
	}
	defer zeroBytes(key)
	return json.Marshal(encodedKeyring{
		ActiveVersion: "e2e-v1",
		Keys:          []encodedKeyringEntry{{Version: "e2e-v1", Key: base64.RawURLEncoding.EncodeToString(key)}},
	})
}

func newSigningKeyring() ([]byte, error) {
	publicKey, privateKey, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		return nil, err
	}
	defer zeroBytes(privateKey)
	seed := privateKey.Seed()
	defer zeroBytes(seed)
	return json.Marshal(encodedKeyring{
		ActiveVersion: "e2e-v1",
		Keys: []encodedKeyringEntry{{
			Version: "e2e-v1", PublicKey: base64.RawURLEncoding.EncodeToString(publicKey),
			PrivateKey: base64.RawURLEncoding.EncodeToString(seed),
		}},
	})
}

func newDashboardAssertionKey() ([]byte, error) {
	_, privateKey, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		return nil, err
	}
	defer zeroBytes(privateKey)
	encoded, err := x509.MarshalPKCS8PrivateKey(privateKey)
	if err != nil {
		return nil, err
	}
	defer zeroBytes(encoded)
	return pem.EncodeToMemory(&pem.Block{Type: "PRIVATE KEY", Bytes: encoded}), nil
}

func newE2ECA() (*x509.Certificate, *ecdsa.PrivateKey, []byte, error) {
	key, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		return nil, nil, nil, err
	}
	serial, err := randomSerial()
	if err != nil {
		return nil, nil, nil, err
	}
	now := time.Now().UTC()
	template := &x509.Certificate{
		SerialNumber: serial, Subject: pkix.Name{CommonName: "vLLM-SR Dashboard E2E CA"},
		NotBefore: now.Add(-time.Hour), NotAfter: now.Add(24 * time.Hour),
		IsCA: true, BasicConstraintsValid: true,
		KeyUsage: x509.KeyUsageCertSign | x509.KeyUsageCRLSign | x509.KeyUsageDigitalSignature,
	}
	der, err := x509.CreateCertificate(rand.Reader, template, template, &key.PublicKey, key)
	if err != nil {
		return nil, nil, nil, err
	}
	return template, key, pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: der}), nil
}

func newE2EServerCertificate(
	ca *x509.Certificate,
	caKey *ecdsa.PrivateKey,
	dnsNames []string,
) ([]byte, []byte, error) {
	key, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		return nil, nil, err
	}
	serial, err := randomSerial()
	if err != nil {
		return nil, nil, err
	}
	now := time.Now().UTC()
	template := &x509.Certificate{
		SerialNumber: serial, Subject: pkix.Name{CommonName: dnsNames[0]}, DNSNames: dnsNames,
		NotBefore: now.Add(-time.Hour), NotAfter: now.Add(12 * time.Hour),
		KeyUsage:    x509.KeyUsageDigitalSignature | x509.KeyUsageKeyEncipherment,
		ExtKeyUsage: []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
	}
	der, err := x509.CreateCertificate(rand.Reader, template, ca, &key.PublicKey, caKey)
	if err != nil {
		return nil, nil, err
	}
	keyDER, err := x509.MarshalECPrivateKey(key)
	if err != nil {
		return nil, nil, err
	}
	return pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: der}),
		pem.EncodeToMemory(&pem.Block{Type: "EC PRIVATE KEY", Bytes: keyDER}), nil
}

func randomSerial() (*big.Int, error) {
	limit := new(big.Int).Lsh(big.NewInt(1), 128)
	return rand.Int(rand.Reader, limit)
}

func randomURLToken(bytes int) (string, error) {
	payload := make([]byte, bytes)
	if _, err := rand.Read(payload); err != nil {
		return "", err
	}
	defer zeroBytes(payload)
	return base64.RawURLEncoding.EncodeToString(payload), nil
}

func cloneSecretData(source map[string][]byte) map[string][]byte {
	result := make(map[string][]byte, len(source))
	for key, value := range source {
		result[key] = append([]byte(nil), value...)
	}
	return result
}

func zeroManagedMaterial(material *dashboardManagedMaterial) {
	if material == nil {
		return
	}
	for _, values := range []map[string][]byte{material.routerSecret, material.dashboardSecret, material.storeSecret} {
		for key, value := range values {
			zeroBytes(value)
			delete(values, key)
		}
	}
	zeroBytes(material.bootstrapToken)
	*material = dashboardManagedMaterial{}
}

func zeroBytes(value []byte) {
	for index := range value {
		value[index] = 0
	}
}

func boolPointer(value bool) *bool { return &value }
