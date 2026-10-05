import { useEffect, useRef, useState } from 'react'
import type { FormEvent } from 'react'
import { api } from './api'
import type { Analysis, Article, ArticleInput, ArticleItem, Capabilities, Page, Stats } from './api'

const EMPTY_PAGE: Page<ArticleItem> = { items: [], total: 0, page: 1, page_size: 12 }
const MODE_LABELS = { classify: '패턴 분류', summarize: '내용 요약', both: '분류 + 요약' }
const STATUS_LABELS = { queued: '대기 중', running: '분석 중', completed: '완료', failed: '실패' }
const SAMPLE: ArticleInput = {
  title: '연결 테스트용 예시: 도서관 보수 계획',
  body: 'This is a fictional article for testing the news workspace. Cedar City Council approved a library renovation plan with a budget of two million dollars. Construction is scheduled to begin next September. The library will stay open during most of the work. This example is not a real news report.',
  source_url: null, language: 'en',
}

function Icon({ name, size = 20 }: { name: 'archive' | 'plus' | 'arrow' | 'spark' | 'history' | 'search' | 'trash'; size?: number }) {
  const paths = {
    archive: 'M3 4h18v4H3z M5 8v12h14V8 M10 12h4',
    plus: 'M12 5v14 M5 12h14',
    arrow: 'M5 12h14 M13 6l6 6-6 6',
    spark: 'M12 3l2.5 6.5L21 12l-6.5 2.5L12 21l-2.5-6.5L3 12l6.5-2.5z',
    history: 'M3 11a9 9 0 1 1 2.7 7 M3 4v7h7 M12 7v5l3 2',
    search: 'M10.5 3a7.5 7.5 0 1 0 0 15 7.5 7.5 0 0 0 0-15 M16 16l5 5',
    trash: 'M4 7h16 M9 7V4h6v3 M6 7l1 14h10l1-14 M10 10v7 M14 10v7',
  }
  return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d={paths[name]} /></svg>
}

function date(value: string) {
  return new Intl.DateTimeFormat('ko-KR', { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit' }).format(new Date(value))
}
function message(error: unknown) { return error instanceof Error ? error.message : '요청을 처리하지 못했습니다.' }

function Results({ items, version, onOpen }: { items: Analysis[]; version?: number; onOpen?: (id: string) => void }) {
  if (!items.length) return <div className="result-empty"><Icon name="spark" size={24} /><p>아직 분석 기록이 없습니다.</p><span>AI 모델 연결 후 분류와 요약 결과를 이곳에서 확인할 수 있습니다.</span></div>
  return <div className="results">{items.map(job => <article className="result-card" key={job.id}>
    <div className="result-top"><span className={`status ${job.status}`}>{STATUS_LABELS[job.status]}</span><span>{MODE_LABELS[job.mode]} · {date(job.created_at)}</span></div>
    {onOpen ? <button className="history-title" onClick={() => onOpen(job.article_id)}>{job.article_title}<Icon name="arrow" size={16} /></button> : null}
    {version !== undefined && job.article_version !== version ? <p className="muted">이 결과는 수정 전 기사 v{job.article_version} 기준입니다.</p> : null}
    {job.classification ? <div className="classification">
      <strong>{job.classification.display_label ?? job.classification.message}</strong>
      {job.classification.fake_score !== undefined ? <><div className="score-track"><span style={{ width: `${job.classification.fake_score * 100}%` }} /></div><p>가짜 뉴스 패턴 점수 {(job.classification.fake_score * 100).toFixed(1)}%{job.classification.uncertain ? ' · 판단 불확실' : ''}</p></> : null}
      <p className="muted">{job.classification.note}</p>
    </div> : null}
    {job.summary ? <div className="summary"><p>{job.summary.summary}</p><small>{job.summary.model} · {job.summary.note}</small></div> : null}
    {job.error ? <p className="inline-error">{job.error}</p> : null}
    {job.status === 'queued' || job.status === 'running' ? <p className="muted">완료되면 결과가 자동으로 표시됩니다. 다른 기사를 관리하며 기다릴 수 있습니다.</p> : null}
  </article>)}</div>
}

function Editor({ article, onSave, onDelete, busy, onDirty }: {
  article: Article | null; onSave: (input: ArticleInput) => Promise<void>;
  onDelete: () => Promise<void>; busy: boolean; onDirty: (dirty: boolean) => void;
}) {
  const initial: ArticleInput = article ? { title: article.title, body: article.body, source_url: article.source_url, language: article.language }
    : { title: '', body: '', source_url: null, language: 'en' }
  const [values, setValues] = useState<ArticleInput>(initial)
  function change(next: ArticleInput) {
    setValues(next)
    onDirty(next.title !== initial.title || next.body !== initial.body || next.source_url !== initial.source_url || next.language !== initial.language)
  }
  async function submit(event: FormEvent) { event.preventDefault(); await onSave({ ...values, title: values.title.trim(), body: values.body.trim() }) }
  return <form className="editor" onSubmit={submit}>
    <div className="panel-heading"><div><span className="eyebrow">{article ? `ARTICLE / V${article.version}` : 'NEW ARTICLE'}</span><h2>{article ? '기사 편집' : '새 기사 저장'}</h2></div><span className="round-icon"><Icon name="plus" /></span></div>
    <label htmlFor="title">기사 제목 <span>필수</span></label>
    <input id="title" name="title" value={values.title} onChange={event => change({ ...values, title: event.target.value })} maxLength={300} required placeholder="보관할 기사의 제목을 입력하세요" disabled={busy} />
    <div className="field-grid"><div><label htmlFor="source">원문 링크 <span>선택</span></label><input id="source" type="url" value={values.source_url ?? ''} onChange={event => change({ ...values, source_url: event.target.value || null })} maxLength={2048} placeholder="https://" disabled={busy} /></div><div><label htmlFor="language">원문 언어</label><select id="language" value={values.language} onChange={event => change({ ...values, language: event.target.value as ArticleInput['language'] })} disabled={busy}><option value="en">영어</option><option value="ko">한국어</option><option value="unknown">기타</option></select></div></div>
    <label htmlFor="body">기사 본문 <span>필수</span></label>
    <textarea id="body" name="body" value={values.body} onChange={event => change({ ...values, body: event.target.value })} required minLength={20} maxLength={59000} placeholder="원문 내용을 붙여 넣으세요. 기사와 분석 이력은 워크스페이스에 저장됩니다." disabled={busy} />
    <div className="editor-note"><span>본문 20자 이상</span><span>{values.body.length.toLocaleString()} / 59,000</span></div>
    <div className="editor-actions"><div>{article ? <button type="button" className="danger-button" onClick={() => void onDelete()} disabled={busy}><Icon name="trash" size={17} />삭제</button> : <button type="button" className="text-button" onClick={() => change(SAMPLE)} disabled={busy}>예시 기사 입력</button>}</div><button className="primary" type="submit" disabled={busy || !values.title.trim() || values.body.trim().length < 20}>{busy ? '저장 중…' : article ? '변경 내용 저장' : '기사 저장'}<Icon name="arrow" size={18} /></button></div>
  </form>
}

export default function App() {
  const [tab, setTab] = useState<'articles' | 'history'>('articles')
  const [health, setHealth] = useState<'loading' | 'online' | 'offline'>('loading')
  const [caps, setCaps] = useState<Capabilities | null>(null)
  const [stats, setStats] = useState<Stats | null>(null)
  const [articlePage, setArticlePage] = useState(EMPTY_PAGE)
  const [page, setPage] = useState(1)
  const [search, setSearch] = useState('')
  const [query, setQuery] = useState('')
  const [selected, setSelected] = useState<Article | null>(null)
  const [jobs, setJobs] = useState<Analysis[]>([])
  const [history, setHistory] = useState<Page<Analysis> | null>(null)
  const [historyPage, setHistoryPage] = useState(1)
  const [listLoading, setListLoading] = useState(true)
  const [detailLoading, setDetailLoading] = useState(false)
  const [busy, setBusy] = useState(false)
  const [dirty, setDirty] = useState(false)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [revision, setRevision] = useState(0)
  const [editorRevision, setEditorRevision] = useState(0)
  const [summaryLanguage, setSummaryLanguage] = useState('ko')
  const selectionRequest = useRef(0)
  const hasPending = jobs.some(job => job.status === 'queued' || job.status === 'running')
  const historyPending = history?.items.some(job => job.status === 'queued' || job.status === 'running') ?? false

  useEffect(() => {
    if (!dirty) return
    const warn = (event: BeforeUnloadEvent) => { event.preventDefault(); event.returnValue = '' }
    window.addEventListener('beforeunload', warn)
    return () => window.removeEventListener('beforeunload', warn)
  }, [dirty])

  useEffect(() => {
    let active = true
    Promise.all([api.health(), api.capabilities(), api.stats()]).then(([, capability, counts]) => {
      if (active) { setHealth('online'); setCaps(capability); setStats(counts) }
    }).catch(reason => { if (active) { setHealth('offline'); setError(message(reason)) } })
    return () => { active = false }
  }, [revision])

  useEffect(() => {
    let active = true
    setListLoading(true)
    api.articles(page, query).then(result => { if (active) setArticlePage(result) })
      .catch(reason => { if (active) setError(message(reason)) })
      .finally(() => { if (active) setListLoading(false) })
    return () => { active = false }
  }, [page, query, revision])

  useEffect(() => {
    if (tab !== 'history') return
    let active = true
    const refresh = () => api.analyses(undefined, historyPage).then(result => { if (active) setHistory(result) })
      .catch(reason => { if (active) setError(message(reason)) })
    void refresh()
    const timer = historyPending ? window.setInterval(() => void refresh(), 2500) : undefined
    return () => { active = false; window.clearInterval(timer) }
  }, [tab, historyPage, revision, historyPending])

  useEffect(() => {
    if (!selected || !hasPending) return
    let active = true
    const timer = window.setInterval(() => {
      Promise.all([api.analyses(selected.id), api.stats()]).then(([result, counts]) => {
        if (active) { setJobs(result.items); setStats(counts) }
      }).catch(reason => { if (active) setError(message(reason)) })
    }, 2500)
    return () => { active = false; window.clearInterval(timer) }
  }, [selected?.id, hasPending])

  function mayLeave() { return !dirty || window.confirm('저장하지 않은 변경 내용이 있습니다. 계속 이동할까요?') }
  function newArticle() {
    if (busy || !mayLeave()) return
    selectionRequest.current += 1
    setEditorRevision(value => value + 1)
    setDetailLoading(false); setSelected(null); setJobs([]); setDirty(false); setTab('articles'); setError(''); setNotice('')
  }
  function switchTab(next: 'articles' | 'history') {
    if (next === tab) return
    if (busy || !mayLeave()) return
    setDirty(false); setTab(next)
  }
  async function openArticle(id: string) {
    if (busy || !mayLeave()) return
    const current = ++selectionRequest.current
    setDirty(false); setDetailLoading(true); setError(''); setNotice(''); setTab('articles')
    try {
      const [article, records] = await Promise.all([api.article(id), api.analyses(id)])
      if (current === selectionRequest.current) { setSelected(article); setJobs(records.items) }
    } catch (reason) { if (current === selectionRequest.current) setError(message(reason)) }
    finally { if (current === selectionRequest.current) setDetailLoading(false) }
  }
  async function saveArticle(input: ArticleInput) {
    setBusy(true); setError(''); setNotice('')
    try {
      const article = selected ? await api.update(selected.id, input, selected.version) : await api.create(input)
      setSelected(article); setDirty(false); setNotice('기사가 저장됐습니다.'); setRevision(value => value + 1)
      const records = await api.analyses(article.id); setJobs(records.items)
    } catch (reason) { setError(message(reason)) }
    finally { setBusy(false) }
  }
  async function deleteArticle() {
    if (!selected || !window.confirm('이 기사와 연결된 분석 이력을 삭제할까요?')) return
    setBusy(true); setError('')
    try {
      await api.remove(selected.id)
      setSelected(null); setJobs([]); setDirty(false); setNotice('기사가 삭제됐습니다.')
      setPage(1); setRevision(value => value + 1)
    } catch (reason) { setError(message(reason)) }
    finally { setBusy(false) }
  }
  async function analyze(mode: Analysis['mode']) {
    if (!selected || busy || dirty || detailLoading) return
    setBusy(true); setError('')
    try {
      const job = await api.analyze(selected.id, mode, summaryLanguage)
      setJobs(previous => [job, ...previous]); setRevision(value => value + 1)
    } catch (reason) { setError(message(reason)) }
    finally { setBusy(false) }
  }

  const connectionText = health === 'online' ? '저장소 연결됨' : health === 'offline' ? '연결 확인 필요' : '연결 중'
  const totalPages = Math.max(1, Math.ceil(articlePage.total / articlePage.page_size))

  return <div className="app-shell">
    <aside className="sidebar">
      <a className="brand" href="/" aria-label="SAI 뉴스 워크스페이스"><span className="brand-symbol">s<span>ai</span><i /></span><span className="brand-label">NEWS WORKSPACE</span></a>
      <span className="sidebar-label">WORKSPACE</span>
      <nav aria-label="주 메뉴"><button className={tab === 'articles' ? 'nav-item active' : 'nav-item'} onClick={() => switchTab('articles')}><Icon name="archive" />뉴스 보관함<span>{stats?.articles ?? '–'}</span></button><button className={tab === 'history' ? 'nav-item active' : 'nav-item'} onClick={() => switchTab('history')}><Icon name="history" />분석 기록<span>{stats?.analyses ?? '–'}</span></button></nav>
      <div className="sidebar-project"><span className="eyebrow">SAI PROJECT / 01</span><h3>읽고, 비교하고,<br />근거를 남기다.</h3><p>뉴스와 분석 이력을<br />한 공간에서 관리하세요.</p><div className="project-lines"><span /><span /><span /><span /></div></div>
      <div className="sidebar-bottom"><span className={`connection-dot ${health}`} /><div><strong>{connectionText}</strong><span>로컬 워크스페이스</span></div><button className="refresh-button" onClick={() => { setError(''); setRevision(value => value + 1) }} aria-label="연결 다시 확인">↻</button></div>
    </aside>
    <main>
      <header className="topbar"><span>WORKSPACE <span className="slash">/</span> {tab === 'articles' ? '뉴스 보관함' : '분석 기록'}</span><span className="phase-badge"><span />{caps?.ai_enabled ? 'AI 분석 사용 가능' : 'AI 연결 준비'}</span></header>
      <div className="page-content">
        <section className="page-heading"><div><span className="eyebrow">YOUR NEWS, IN CONTEXT</span><h1>{tab === 'articles' ? <>뉴스를 모으고,<br /><span>맥락을 연결하세요.</span></> : <>읽은 뉴스의<br /><span>분석 기록을 한눈에.</span></>}</h1><p>{tab === 'articles' ? '관심 있는 기사를 저장하고, 분류와 요약의 출발점을 만들어 보세요.' : '저장된 기사 버전별로 분석 상태와 결과를 확인하세요.'}</p></div><button className="primary new-article-button" onClick={newArticle} disabled={busy}><Icon name="plus" size={18} />새 기사</button></section>
        {error ? <div className="banner error" role="alert"><span>{error}</span><button onClick={() => setError('')} aria-label="오류 메시지 닫기">×</button></div> : null}
        {notice ? <div className="banner success" role="status"><span>{notice}</span><button onClick={() => setNotice('')} aria-label="알림 닫기">×</button></div> : null}
        <section className="stats-row" aria-label="워크스페이스 현황"><div><span>보관한 기사</span><strong>{stats?.articles ?? '–'}<small>건</small></strong><Icon name="archive" size={22} /></div><div><span>완료한 분석</span><strong>{stats?.completed ?? '–'}<small>건</small></strong><Icon name="spark" size={22} /></div><div><span>대기·진행 중</span><strong>{stats?.pending ?? '–'}<small>건</small></strong><Icon name="history" size={22} /></div></section>
        {tab === 'articles' ? <div className="workspace-grid">
          <section className="article-library"><div className="library-heading"><h2>뉴스 보관함</h2><span>{articlePage.total} ARTICLES</span></div><form className="search-form" onSubmit={event => { event.preventDefault(); setQuery(search.trim()); setPage(1) }}><Icon name="search" size={18} /><input aria-label="기사 제목 검색" placeholder="제목으로 기사 찾기" value={search} onChange={event => setSearch(event.target.value)} /><button type="submit">검색</button></form>
            <div className="article-list" aria-busy={listLoading}>{listLoading ? <div className="library-empty">기사를 불러오는 중…</div> : articlePage.items.length ? articlePage.items.map((article, index) => <button className={selected?.id === article.id ? 'article-card selected' : 'article-card'} key={article.id} onClick={() => void openArticle(article.id)} disabled={busy}><div className="article-meta"><span>{String((page - 1) * 12 + index + 1).padStart(2, '0')}</span><span>{article.language === 'en' ? 'ENGLISH' : article.language === 'ko' ? 'KOREAN' : 'OTHER'} · {date(article.created_at)}</span></div><h3>{article.title}</h3><p>{article.preview}</p><div className="article-footer"><span>기사 열기</span><Icon name="arrow" size={17} /></div></button>) : <div className="library-empty"><span className="empty-number">01</span><h3>{query ? '검색 결과가 없습니다.' : '첫 번째 기사를 기다립니다.'}</h3><p>{query ? '다른 제목으로 검색해 보세요.' : '오른쪽에서 기사를 저장하면 이곳에 차곡차곡 쌓입니다.'}</p></div>}</div>
            <div className="pagination"><button disabled={page <= 1 || listLoading} onClick={() => setPage(value => value - 1)}>← 이전</button><span>{page} / {totalPages}</span><button disabled={page >= totalPages || listLoading} onClick={() => setPage(value => value + 1)}>다음 →</button></div>
          </section>
          <div className="detail-column">{detailLoading ? <div className="editor loading-panel">기사를 불러오는 중…</div> : <Editor key={selected ? `${selected.id}:${selected.version}` : `new:${editorRevision}`} article={selected} onSave={saveArticle} onDelete={deleteArticle} busy={busy} onDirty={setDirty} />}
            <section className="analysis-panel"><div className="analysis-heading"><div><span className="eyebrow">NEXT / INSIGHT</span><h2>기사 분석</h2></div><Icon name="spark" size={24} /></div><p className="muted">{caps?.ai_enabled ? '저장한 원문에서 영어 뉴스 패턴을 분류하고 핵심 내용을 요약합니다.' : '기사를 먼저 모아 두세요. AI 모델 연결 후 분류와 요약을 사용할 수 있습니다.'}</p><div className="analysis-controls"><select aria-label="요약 언어" value={summaryLanguage} onChange={event => setSummaryLanguage(event.target.value)} disabled={!caps?.ai_enabled || busy}><option value="ko">한국어 요약</option><option value="en">영어 요약</option></select><button className="secondary" disabled={!selected || !caps?.ai_enabled || busy || dirty || hasPending} onClick={() => void analyze('classify')}>패턴 분류</button><button className="secondary" disabled={!selected || !caps?.ai_enabled || busy || dirty || hasPending} onClick={() => void analyze('summarize')}>내용 요약</button></div>{dirty && caps?.ai_enabled ? <p className="muted">변경 내용을 저장한 뒤 분석해 주세요.</p> : null}<Results items={jobs} version={selected?.version} /></section>
          </div>
        </div> : <section className="history-panel"><div className="library-heading"><h2>분석 타임라인</h2><span>{history?.total ?? 0} RECORDS</span></div>{history ? <Results items={history.items} onOpen={id => void openArticle(id)} /> : <p className="muted">기록을 불러오는 중…</p>}<div className="pagination"><button disabled={historyPage <= 1} onClick={() => setHistoryPage(value => value - 1)}>← 이전</button><span>{historyPage} / {Math.max(1, Math.ceil((history?.total ?? 0) / 20))}</span><button disabled={historyPage * 20 >= (history?.total ?? 0)} onClick={() => setHistoryPage(value => value + 1)}>다음 →</button></div></section>}
        <footer className="page-footer"><span>SAI · SIGNAL, ARCHIVE, INSIGHT</span><span>뉴스 분류 점수와 요약은 사실 검증을 대신하지 않습니다.</span></footer>
      </div>
    </main>
  </div>
}
