"""Run with: .venv\\Scripts\\python.exe -m streamlit run app.py"""

import json
import os
from pathlib import Path

import streamlit as st

from news_ai.model import load_classifier, predict
from news_ai.summarizer import OllamaError, OllamaSummarizer

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "artifacts" / "classifier.joblib"

st.set_page_config(page_title="SAI 뉴스 분석", page_icon="📰", layout="wide")
st.title("SAI 뉴스 분석")
st.caption("영어 뉴스의 가짜 뉴스 패턴을 분류하고, 로컬 LLM으로 핵심 내용을 요약합니다.")


@st.cache_resource
def classifier(path: str, modified_ns: int):
    return load_classifier(Path(path))


with st.sidebar:
    st.header("요약 설정")
    llm_model = st.text_input("Ollama 모델", os.getenv("OLLAMA_MODEL", "exaone3.5:2.4b"))
    language = st.selectbox("요약 언어", ["한국어", "English"])
    if st.button("Ollama 연결 확인"):
        try:
            available = OllamaSummarizer(model=llm_model).list_models()
            if available:
                st.success("사용 가능: " + ", ".join(available))
                if llm_model not in available:
                    st.info(f"선택한 모델이 목록에 없으면 먼저 ollama pull {llm_model} 을 실행하세요.")
            else:
                st.info("설치된 모델이 없습니다. 아래 준비 명령을 실행하세요.")
        except (OllamaError, ValueError) as error:
            st.warning(str(error))
    st.markdown("[Ollama 설치](https://ollama.com/download/windows)")
    st.code("ollama pull exaone3.5:2.4b", language="powershell")
    st.caption("요약은 Ollama 로컬 서버를 사용합니다. 긴 기사는 여러 번 나누어 요약합니다.")

analysis_tab, evaluation_tab = st.tabs(["기사 분석", "학습 결과"])

with analysis_tab:
    title = st.text_input("기사 제목 (선택)", key="article_title")
    body = st.text_area("기사 본문", height=300, placeholder="영어 뉴스 본문을 붙여 넣으세요.", key="article_body")
    input_identity = (title, body, llm_model, language)
    if st.session_state.get("result_identity") != input_identity:
        st.session_state.pop("classification", None)
        st.session_state.pop("summary", None)
        st.session_state["result_identity"] = input_identity
    column_classify, column_summary = st.columns(2)
    classify_clicked = column_classify.button("가짜 뉴스 패턴 분류", type="primary", disabled=not body.strip(), use_container_width=True)
    summary_clicked = column_summary.button("핵심 내용 요약", disabled=not body.strip(), use_container_width=True)
    if classify_clicked:
        try:
            with st.spinner("분류 중..."):
                bundle = classifier(str(MODEL), MODEL.stat().st_mtime_ns)
                st.session_state["classification"] = predict(bundle, title, body)
        except (OSError, ValueError) as error:
            st.error(f"분류 모델을 불러오지 못했습니다: {error}")
            st.code(".venv\\Scripts\\python.exe -m news_ai train", language="powershell")
    if summary_clicked:
        try:
            with st.spinner("로컬 LLM으로 요약 중... 긴 기사는 시간이 더 걸립니다."):
                st.session_state["summary"] = OllamaSummarizer(model=llm_model).summarize(
                    title, body, "ko" if language == "한국어" else "en",
                ).to_dict()
        except (OllamaError, ValueError) as error:
            st.error(str(error))
    result = st.session_state.get("classification")
    if result:
        st.subheader("분류 결과")
        if result["status"] == "ok":
            st.write(result["display_label"])
            left, right = st.columns(2)
            left.metric("가짜 뉴스 패턴 점수", f"{result['fake_score']:.1%}")
            right.metric("진짜 뉴스 패턴 점수", f"{result['real_score']:.1%}")
            if result["uncertain"]:
                st.info("두 패턴의 점수가 가까워 판단이 불확실합니다.")
            st.caption(result["note"])
        else:
            st.info(result["message"])
    summary_result = st.session_state.get("summary")
    if summary_result:
        st.subheader("기사 요약")
        st.markdown(summary_result["summary"])
        st.caption(summary_result["note"])
        st.caption(f"모델 {summary_result['model']} · 본문 조각 {summary_result['chunks']}개 · LLM 호출 {summary_result['llm_calls']}회")
        st.download_button("요약 저장", summary_result["summary"], "news_summary.txt", mime="text/plain")
    st.info("분류 모델은 과거 영어 뉴스의 문체·어휘 패턴을 학습했습니다. 한국어 뉴스와 최신 뉴스의 분류 성능은 검증되지 않았습니다.")

with evaluation_tab:
    metrics_file = ROOT / "artifacts" / "metrics.json"
    if not metrics_file.exists():
        st.info("모델을 학습하면 평가 결과가 표시됩니다.")
    else:
        report = json.loads(metrics_file.read_text(encoding="utf-8"))
        accuracy, macro_f1, count = st.columns(3)
        accuracy.metric("테스트 정확도", f"{report['test']['accuracy']:.2%}")
        macro_f1.metric("테스트 Macro F1", f"{report['test']['macro_f1']:.4f}")
        count.metric("평가 기사", f"{report['test']['rows']:,}")
        st.caption("같은 기사·제목·본문 시작 구간은 한 분할에 묶었습니다. 날짜·언론사별 외부 검증은 별도로 필요합니다.")
        st.subheader("데이터 처리")
        st.json(report["data_audit"])
        st.subheader("혼동 행렬: 행은 실제, 열은 예측")
        st.dataframe({
            "실제 라벨": ["real", "fake"],
            "예측 real": [row[0] for row in report["test"]["confusion_matrix_rows_actual_columns_predicted"]],
            "예측 fake": [row[1] for row in report["test"]["confusion_matrix_rows_actual_columns_predicted"]],
        }, hide_index=True)
        with st.expander("전체 평가 기록"):
            st.json(report)
