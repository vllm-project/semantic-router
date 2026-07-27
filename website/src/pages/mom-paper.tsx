import React from 'react'
import { translate } from '@docusaurus/Translate'
import PaperViewerPage from '@site/src/components/PaperViewerPage'

const PDF_URL = '/mom-paper.pdf'

export default function MixtureOfModelsPaper(): JSX.Element {
  return (
    <PaperViewerPage
      title={translate({
        id: 'momPaper.title',
        message: 'Mixture-of-Models',
      })}
      socialTitle={translate({
        id: 'momPaper.socialTitle',
        message: 'Mixture-of-Models — vLLM Semantic Router',
      })}
      metaDescription={translate({
        id: 'momPaper.metaDescription',
        message: 'Mixture-of-Models: Conditional Computation Across Model Boundaries, a position paper from the vLLM Semantic Router Project.',
      })}
      heroDescription={translate({
        id: 'momPaper.heroDescription',
        message: 'Conditional computation across independently versioned models, presented as a full PDF reader for the vLLM Semantic Router Mixture-of-Models position paper.',
      })}
      pdfUrl={PDF_URL}
    />
  )
}
