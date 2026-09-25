import React from 'react'
import { translate } from '@docusaurus/Translate'
import PaperViewerPage from '@site/src/components/PaperViewerPage'

const PDF_URL = '/decision-paper.pdf'

export default function DecisionPaper(): JSX.Element {
  return (
    <PaperViewerPage
      title={translate({ id: 'decisionPaper.title', message: 'Decision Paper' })}
      socialTitle={translate({
        id: 'decisionPaper.socialTitle',
        message: 'Decision 1.0: Towards Open Decision Foundation Models — vLLM Semantic Router',
      })}
      metaDescription={translate({
        id: 'decisionPaper.metaDescription',
        message: 'Decision 1.0: Towards Open Decision Foundation Models',
      })}
      heroDescription={translate({
        id: 'decisionPaper.heroDescription',
        message: 'Decision 1.0: Towards Open Decision Foundation Models. A technical report on the models, data, and training behind the Decision family.',
      })}
      pdfUrl={PDF_URL}
    />
  )
}
