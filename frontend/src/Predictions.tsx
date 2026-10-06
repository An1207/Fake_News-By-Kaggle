import type { Analysis, Prediction } from './api'

function signed(value: number) { return `${value >= 0 ? '+' : ''}${value.toFixed(3)}` }

function PredictionCard({ prediction, title }: { prediction: Prediction; title: string }) {
  const detail = prediction.explanation
  const parameters = detail?.parameters
  return <section className="prediction-card" aria-label={title}>
    <h3>{title}</h3>
    <strong className="prediction-label">{prediction.display_label ?? prediction.message}</strong>
    {prediction.fake_score !== undefined ? <>
      <div className="score-track"><span style={{ width: `${prediction.fake_score * 100}%` }} /></div>
      <p className="prediction-score">가짜 패턴 점수 <b>{(prediction.fake_score * 100).toFixed(1)}%</b></p>
      <p className="muted">진짜 패턴 점수 {((1 - prediction.fake_score) * 100).toFixed(1)}%{prediction.uncertain ? ' · 판단 불확실' : ''}</p>
    </> : null}
    {detail && parameters ? <>
      <dl className="prediction-parameters">
        <div><dt>가짜 판정 기준</dt><dd>점수 ≥ {parameters.decision_threshold * 100}%</dd></div>
        <div><dt>LR 정규화 C</dt><dd>{parameters.C} · {parameters.class_weight}</dd></div>
        <div><dt>TF-IDF 특징</dt><dd>{parameters.tfidf_features.toLocaleString()}개 · {parameters.ngram_range.join('–')} gram</dd></div>
        <div><dt>일치한 특징</dt><dd>{detail.input.matched_tfidf_features}개 · 입력 {detail.input.normalized_words}단어</dd></div>
        {parameters.encoder_dimensions ? <><div><dt>문맥 특징</dt><dd>{parameters.encoder_dimensions}차원 · 앞 {parameters.encoder_max_tokens}토큰</dd></div><div><dt>문맥 가중치</dt><dd>{parameters.embedding_weight} · EuroBERT 고정</dd></div></> : null}
      </dl>
      <div className="contribution-summary"><b>이 기사에 대한 점수 기여도</b>
        <p>기본값 {signed(detail.intercept)} + 어휘 {signed(detail.tfidf_contribution)}{parameters.encoder_dimensions ? ` + 문맥 ${signed(detail.encoder_contribution)}` : ''} = {signed(detail.logit)}</p>
        <span>log-odds 단위 · 양수: 가짜 패턴 방향 / 음수: 진짜 패턴 방향</span>
      </div>
      <details className="feature-evidence"><summary>주요 단어·표현 기여도 보기</summary>
        {(['toward_fake', 'toward_real'] as const).map(direction => <div key={direction}><h4>{direction === 'toward_fake' ? '가짜 패턴 쪽으로' : '진짜 패턴 쪽으로'}</h4>
          {detail[direction].length ? <ul>{detail[direction].map(item => <li key={item.term}><span>{item.term}</span><b>{signed(item.contribution)}</b></li>)}</ul> : <p className="muted">해당 방향의 일치 특징 없음</p>}
        </div>)}
        <p className="muted">상위 5개씩 표시합니다. 문맥 기여도는 벡터 전체의 합산값이며 특정 단어의 인과적 중요도를 뜻하지 않습니다.</p>
      </details>
    </> : null}
  </section>
}

export default function Predictions({ classification }: { classification: NonNullable<Analysis['classification']> }) {
  const models = classification.models
  const context = models?.transformer.explanation?.parameters
  return <div className="predictions">
    <div className="prediction-grid">{models ? <>
      <PredictionCard prediction={models.baseline} title="기존 · TF-IDF + LR" />
      <PredictionCard prediction={models.transformer} title="새 모델 · EuroBERT + TF-IDF + LR" />
    </> : <PredictionCard prediction={classification} title="저장된 분류 결과" />}</div>
    {models && classification.agreement !== null && classification.agreement !== undefined ? <p className="model-agreement">{classification.agreement ? '두 모델의 패턴 판정이 같습니다.' : '두 모델의 패턴 판정이 다릅니다. 각 모델의 점수와 기여도를 비교해 보세요.'}</p> : null}
    {classification.status === 'ok' && classification.explanation ? <aside className="prediction-process" aria-label="AI 예측 과정">
      <h3>AI는 이렇게 예측했습니다</h3>
      <p>① 제목·본문을 정리하고 영어·길이·학습 어휘를 확인합니다. ② 전체 원문의 TF-IDF 특징을 추출합니다.{context ? ` 새 모델은 앞 ${context.encoder_max_tokens}토큰에서 EuroBERT 문맥 특징 ${context.encoder_dimensions}개를 더합니다.` : ''} ③ 각 LR이 특징 × 학습 가중치 + 기본값을 합산해 sigmoid로 점수를 계산합니다. ④ 가짜 패턴 점수가 50% 이상이면 가짜 패턴 쪽으로 판정합니다.</p>
      <p className="muted">더 높은 패턴 점수도 65% 미만이면 불확실로 표시합니다. 기여도는 실제 선형 계산값이며, 외부 사실 확인이나 LLM이 만든 설명이 아닙니다.</p>
    </aside> : null}
    <p className="muted prediction-note">{classification.note ?? '두 모델은 영어 뉴스 학습 데이터의 패턴을 분류합니다. 사실 검증 결과가 아닙니다.'}</p>
  </div>
}
