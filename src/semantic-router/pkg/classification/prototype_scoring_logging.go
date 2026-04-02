package classification

import (
	"fmt"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

const prototypeMedoidPreviewLimit = 96

func logPrototypeBankSummary(family string, owner string, bank *prototypeBank) {
	representatives := bank.representatives()
	if len(representatives) == 0 {
		logging.Infof("[%s] prototype bank owner=%s prototypes=0", family, owner)
		return
	}

	medoids := make([]string, 0, len(representatives))
	for _, representative := range representatives {
		medoids = append(medoids, fmt.Sprintf("%q(cluster_size=%d, avg_similarity=%.3f)",
			truncatePrototypePreview(representative.Text),
			representative.ClusterSize,
			representative.AvgSimilarity,
		))
	}

	logging.Infof("[%s] prototype bank owner=%s prototypes=%d medoids=[%s]",
		family,
		owner,
		len(representatives),
		strings.Join(medoids, ", "),
	)
}

func truncatePrototypePreview(text string) string {
	text = strings.TrimSpace(text)
	if len(text) <= prototypeMedoidPreviewLimit {
		return text
	}
	return text[:prototypeMedoidPreviewLimit-3] + "..."
}
