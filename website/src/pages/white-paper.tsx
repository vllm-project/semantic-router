import React from 'react'
import { translate } from '@docusaurus/Translate'
import PaperViewerPage from '@site/src/components/PaperViewerPage'

const PDF_URL = '/white-paper.pdf'

export default function WhitePaper(): JSX.Element {
  return (
    <PaperViewerPage
      title={translate({ id: 'whitePaper.title', message: 'White Paper' })}
      metaDescription={translate({
        id: 'whitePaper.metaDescription',
        message: 'Signal Driven Decision Routing for Mixture-of-Modality Models',
      })}
      heroDescription={translate({
        id: 'whitePaper.heroDescription',
        message: 'Signal-driven decision routing for mixture-of-modality models, presented as a full PDF reader inside the same website shell.',
      })}
      pdfUrl={PDF_URL}
    />
  )
}
