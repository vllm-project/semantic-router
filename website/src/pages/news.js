import React from 'react'
import Layout from '@theme/Layout'
import styles from './news.module.css'

const newsArticles = [
  {
    title: 'vLLM Semantic Router: Improving Efficiency in AI Reasoning',
    date: 'September 11, 2025',
    source: 'Red Hat Developer',
    description: 'This article explores how the vLLM Semantic Router addresses challenges in AI reasoning by implementing dynamic, semantic-aware routing to optimize performance and cost.',
    url: 'https://developers.redhat.com/articles/2025/09/11/vllm-semantic-router-improving-efficiency-ai-reasoning',
    category: 'Technical Article',
  },
  {
    title: 'LLM Semantic Router: Intelligent Request Routing for Large Language Models',
    date: 'May 20, 2025',
    source: 'Red Hat Developer',
    description: 'This piece introduces the LLM Semantic Router, focusing on intelligent, cost-aware request routing to ensure efficient processing of queries by large language models.',
    url: 'https://developers.redhat.com/articles/2025/05/20/llm-semantic-router-intelligent-request-routing',
    category: 'Technical Article',
  },
  {
    title: 'Smarter LLMs: How the vLLM Semantic Router Delivers Fast, Efficient Inference',
    date: 'September 2025',
    source: 'Joshua Berkowitz Blog',
    description: 'This blog post highlights the vLLM Semantic Router\'s role in enhancing large language model inference by intelligently routing queries to balance speed, accuracy, and cost.',
    url: 'https://joshuaberkowitz.us/blog/news-1/smarter-llms-how-the-vllm-semantic-router-delivers-fast-efficient-inference-1133',
    category: 'Blog Post',
  },
  {
    title: 'vLLM Semantic Router',
    date: 'September 2025',
    source: 'Jimmy Song\'s Blog',
    description: 'This article provides an overview of the vLLM Semantic Router, detailing its features and applications in improving large language model inference efficiency.',
    url: 'https://jimmysong.io/ai/semantic-router/',
    category: 'Blog Post',
  },
]

function NewsCard({ article }) {
  return (
    <div className={`card ${styles.newsCard}`}>
      <div className="card__header">
        <div className={styles.cardHeader}>
          <h3 className={styles.articleTitle}>{article.title}</h3>
          <div className={styles.articleMeta}>
            <span className={`badge badge--primary ${styles.categoryBadge}`}>
              {article.category}
            </span>
            <span className={styles.source}>{article.source}</span>
            <span className={styles.date}>{article.date}</span>
          </div>
        </div>
      </div>
      <div className="card__body">
        <p className={styles.articleDescription}>{article.description}</p>
      </div>
      <div className="card__footer">
        <a
          href={article.url}
          target="_blank"
          rel="noopener noreferrer"
          className={`button button--primary button--outline ${styles.readMoreButton}`}
        >
          Read More →
        </a>
      </div>
    </div>
  )
}

export default function News() {
  return (
    <Layout
      title="News"
      description="Latest news, articles, and publications about vLLM Semantic Router"
    >
      <div className="container margin-vert--lg">
        <div className="row">
          <div className="col col--12">
            <div className={styles.heroSection}>
              <h1 className={styles.heroTitle}>News & Articles</h1>
              <p className={styles.heroDescription}>
                Stay updated with the latest news, research papers, blog posts, and articles
                about vLLM Semantic Router and its impact on LLM inference efficiency.
              </p>
            </div>
          </div>
        </div>

        <div className="row">
          <div className="col col--12">
            <div className={styles.newsGrid}>
              {newsArticles.map((article, index) => (
                <div key={index} className={styles.newsItem}>
                  <NewsCard article={article} />
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="row">
          <div className="col col--12">
            <div className={styles.contributeSection}>
              <h2>Contribute to News</h2>
              <p>
                Know of an article, blog post, or publication about vLLM Semantic Router
                that should be featured here?
              </p>
              <p className={styles.contributeActions}>
                <a
                  href="https://github.com/vllm-project/semantic-router/issues/new?template=feature_request.md"
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  Submit a suggestion
                </a>
                {' '}
                or
                {' '}
                <a
                  href="https://github.com/vllm-project/semantic-router/issues/new?template=feature_request.md"
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  contribute directly
                </a>
                {' '}
                to our repository.
              </p>
            </div>
          </div>
        </div>
      </div>
    </Layout>
  )
}
