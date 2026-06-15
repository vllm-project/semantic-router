package classification

import (
	"testing"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

func TestClassifier(t *testing.T) {
	RegisterFailHandler(Fail)
	RunSpecs(t, "Classifier Suite")
}

func TestUpdateBestModel(t *testing.T) {
	classifier := &Classifier{}

	bestScore := 0.5
	bestModel := "old-model"

	classifier.updateBestModel(0.8, "new-model", &bestScore, &bestModel)
	if bestScore != 0.8 || bestModel != "new-model" {
		t.Errorf("update: got bestScore=%v, bestModel=%v", bestScore, bestModel)
	}

	classifier.updateBestModel(0.7, "another-model", &bestScore, &bestModel)
	if bestScore != 0.8 || bestModel != "new-model" {
		t.Errorf("not update: got bestScore=%v, bestModel=%v", bestScore, bestModel)
	}
}
