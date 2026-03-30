import React from 'react'
import { translate } from '@docusaurus/Translate'
import PaperViewerPage from '@site/src/components/PaperViewerPage'

const PDF_URL = '/vision-paper.pdf'

export default function VisionPaper(): JSX.Element {
  return (
    <PaperViewerPage
      title={translate({ id: 'visionPaper.title', message: 'Vision Paper' })}
      socialTitle={translate({
        id: 'visionPaper.socialTitle',
        message: 'Our Vision — vLLM Semantic Router',
      })}
      metaDescription={translate({
        id: 'visionPaper.metaDescription',
        message: 'The Workload-Router-Pool Architecture for LLM Inference Optimization: A Vision Paper from the vLLM Semantic Router Project',
      })}
      heroDescription={translate({
        id: 'visionPaper.heroDescription',
        message: 'The Workload-Router-Pool Architecture for LLM Inference Optimization, presented as a full PDF reader for the vLLM Semantic Router vision paper.',
      })}
      pdfUrl={PDF_URL}
    />
  )
}
