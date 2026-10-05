export interface Article {
  id: string
  title: string
  body: string
  source_url: string | null
  language: 'en' | 'ko' | 'unknown'
  version: number
  created_at: string
  updated_at: string
}
export interface ArticleItem extends Omit<Article, 'body'> { preview: string }
export interface Page<T> { items: T[]; total: number; page: number; page_size: number }
export interface Stats { articles: number; analyses: number; completed: number; pending: number; failed: number }
export interface Capabilities {
  ai_enabled: boolean
  classifier_artifact_present: boolean
  summarizer_model: string
  message: string
}
export interface Analysis {
  id: string
  article_id: string
  article_version: number
  article_title: string
  mode: 'classify' | 'summarize' | 'both'
  status: 'queued' | 'running' | 'completed' | 'failed'
  language: string
  classification: { status: string; display_label?: string; fake_score?: number; message?: string; note?: string; uncertain?: boolean } | null
  summary: { summary: string; model: string; note: string; chunks: number } | null
  error: string | null
  created_at: string
}
export type ArticleInput = Pick<Article, 'title' | 'body' | 'source_url' | 'language'>

const BASE = (import.meta.env.VITE_API_BASE_URL ?? '').replace(/\/$/, '')

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${BASE}/api${path}`, {
    ...init,
    headers: { 'Content-Type': 'application/json', ...init?.headers },
  })
  if (!response.ok) {
    const error = await response.json().catch(() => null)
    const detail = error?.detail
    const message = Array.isArray(detail) ? '입력 형식과 길이를 확인해 주세요.'
      : typeof detail === 'object' ? detail?.message : detail
    throw new Error(message || '서버에 연결하지 못했습니다. 잠시 후 다시 시도해 주세요.')
  }
  return response.status === 204 ? undefined as T : response.json() as Promise<T>
}

export const api = {
  health: () => request<{ status: string; database: string }>('/health'),
  capabilities: () => request<Capabilities>('/capabilities'),
  stats: () => request<Stats>('/stats'),
  articles: (page = 1, q = '') => request<Page<ArticleItem>>(`/articles?page=${page}&q=${encodeURIComponent(q)}`),
  article: (id: string) => request<Article>(`/articles/${id}`),
  create: (article: ArticleInput) => request<Article>('/articles', { method: 'POST', body: JSON.stringify(article) }),
  update: (id: string, article: ArticleInput, version: number) => request<Article>(`/articles/${id}`, {
    method: 'PUT', body: JSON.stringify({ ...article, expected_version: version }),
  }),
  remove: (id: string) => request<void>(`/articles/${id}`, { method: 'DELETE' }),
  analyses: (articleId?: string, page = 1) => request<Page<Analysis>>(`/analyses?page=${page}${articleId ? `&article_id=${articleId}` : ''}`),
  analyze: (id: string, mode: Analysis['mode'], language: string) => request<Analysis>(`/articles/${id}/analyses`, {
    method: 'POST', body: JSON.stringify({ mode, language }),
  }),
}
