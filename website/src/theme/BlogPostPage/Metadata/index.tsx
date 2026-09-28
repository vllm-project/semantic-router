import React from 'react'
import Head from '@docusaurus/Head'
import { useBlogPost } from '@docusaurus/plugin-content-blog/client'
import BlogPostPageMetadata from '@theme-original/BlogPostPage/Metadata'

export default function BlogPostSocialMetadata(): React.ReactNode {
  const { metadata: { title } } = useBlogPost()

  return (
    <>
      <BlogPostPageMetadata />
      <Head>
        <meta property="og:title" content={title} />
        <meta name="twitter:title" content={title} />
      </Head>
    </>
  )
}
