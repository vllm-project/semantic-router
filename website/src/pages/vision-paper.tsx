import React from 'react'
import PaperViewerPage from '@site/src/components/PaperViewerPage'

const PDF_URL = '/vision-paper.pdf'
const SHARE_IMAGE = '/img/vllm-sr-logo.light.png'

export default function VisionPaper(): JSX.Element {
  return (
    <PaperViewerPage
      title="Vision Paper"
      socialTitle="Our Vision — vLLM Semantic Router"
      metaDescription="The Workload-Router-Pool Architecture for LLM Inference Optimization: A Vision Paper from the vLLM Semantic Router Project"
      heroDescription="The Workload-Router-Pool Architecture for LLM Inference Optimization, presented as a full PDF reader for the vLLM Semantic Router vision paper."
      pdfUrl={PDF_URL}
      shareImagePath={SHARE_IMAGE}
    />
  )
}
