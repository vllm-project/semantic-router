package v1alpha1

import networkingv1 "k8s.io/api/networking/v1"

func boolPtr(v bool) *bool {
	return &v
}

func int32Ptr(v int32) *int32 {
	return &v
}

func ingressPathTypePrefix() string {
	return string(networkingv1.PathTypePrefix)
}
