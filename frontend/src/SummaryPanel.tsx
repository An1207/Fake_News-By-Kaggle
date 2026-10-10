import { useRef, useState } from 'react'
import { api } from './api'
import type { Analysis, Capabilities, SummaryResult } from './api'

function SummaryText({ result, label }: { result: SummaryResult; label: string }) {
  return <div className="summary"><strong>{label}</strong><p>{result.summary}</p>
    <small>{result.model} · {result.note}</small>
    {result.usage ? <small className="token-usage">입력 {result.usage.input_tokens.toLocaleString()} · 출력 {result.usage.output_tokens.toLocaleString()} · 총 {result.usage.total_tokens.toLocaleString()} 토큰
      {result.estimated_cost_usd !== undefined ? ` · 추정 $${result.estimated_cost_usd.toFixed(6)}` : ''}</small> : null}
  </div>
}

export default function SummaryPanel({ job, caps, onUpdate }: {
  job: Analysis; caps: Capabilities | null; onUpdate: (job: Analysis) => void
}) {
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const submitting = useRef(false)
  if (!job.summary) return null
  const extra = job.gpt_summary
  const pending = extra?.status === 'queued' || extra?.status === 'running'
  async function addGPT() {
    if (submitting.current) return
    submitting.current = true
    setBusy(true); setError('')
    try { onUpdate(await api.gptSummary(job.id, extra?.status === 'failed')) }
    catch (reason) { setError(reason instanceof Error ? reason.message : 'GPT 요약 요청에 실패했습니다.') }
    finally { submitting.current = false; setBusy(false) }
  }
  return <div className="summary-pair">
    <SummaryText result={job.summary} label="Ollama 기본 요약" />
    <div className="gpt-supplement">
      {extra?.status === 'completed' && extra.summary ? <SummaryText result={extra.summary} label="GPT 추가 요약" /> : <>
        <button type="button" className="gpt-token" onClick={() => void addGPT()}
          disabled={!caps?.gpt_summary_available || job.status !== 'completed' || busy || pending}>
          {pending || busy ? 'GPT 요약 진행 중…' : extra?.status === 'failed' ? 'GPT 추가 요약 다시 시도' : 'GPT로 추가 요약'}
          <span>{caps?.gpt_summary_model ?? 'gpt-4o-mini'}</span>
        </button>
        <p className="muted">{caps?.gpt_summary_available ? '클릭하면 같은 기사 원문을 OpenAI로 전송해 추가 요약합니다. API 토큰 비용이 발생합니다.' : 'GPT 추가 요약은 백엔드 API 키 등록 후 사용할 수 있습니다. 기본 요약은 Ollama입니다.'}</p>
        {extra?.error ? <p className="inline-error" role="alert">{extra.error} 재시도 시 추가 과금될 수 있습니다.</p> : null}
      </>}
      {error ? <p className="inline-error" role="alert">{error}</p> : null}
    </div>
  </div>
}
