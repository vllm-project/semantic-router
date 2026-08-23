package dashboard

import (
	"strings"
	"testing"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestUniquePublicInferenceServiceName(t *testing.T) {
	services := []corev1.Service{{ObjectMeta: metav1.ObjectMeta{Name: "generated-public-gateway"}}}
	name, err := uniquePublicInferenceServiceName(services, "gateway-system", "gateway=public")
	if err != nil {
		t.Fatal(err)
	}
	if name != services[0].Name {
		t.Fatalf("service name = %q, want %q", name, services[0].Name)
	}
}

func TestUniquePublicInferenceServiceNameRejectsAmbiguousDiscovery(t *testing.T) {
	for _, services := range [][]corev1.Service{
		nil,
		{{ObjectMeta: metav1.ObjectMeta{Name: "one"}}, {ObjectMeta: metav1.ObjectMeta{Name: "two"}}},
	} {
		if _, err := uniquePublicInferenceServiceName(services, "gateway-system", "gateway=public"); err == nil || !strings.Contains(err.Error(), "exactly one") {
			t.Fatalf("uniquePublicInferenceServiceName() error = %v, want exact-one rejection", err)
		}
	}
}
