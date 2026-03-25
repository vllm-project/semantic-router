import React from 'react'
import PaperViewerPage from '@site/src/components/PaperViewerPage'

const PDF_URL = '/white-paper.pdf'
const SHARE_IMAGE = '/img/vllm-sr-logo.light.png'

export default function WhitePaper(): JSX.Element {
  return (
    <PaperViewerPage
      title="White Paper"
      metaDescription="Signal Driven Decision Routing for Mixture-of-Modality Models"
      heroDescription="Signal-driven decision routing for mixture-of-modality models, presented as a full PDF reader inside the same website shell."
      pdfUrl={PDF_URL}
      shareImagePath={SHARE_IMAGE}
    />
  )
}
