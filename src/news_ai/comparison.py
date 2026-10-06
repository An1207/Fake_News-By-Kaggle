"""Readable comparison from saved aggregate evaluations; no model loading."""
import json
from pathlib import Path


def comparison_text(artifact_dir: Path) -> str:
    def read(path):
        return json.loads(path.read_text(encoding="utf-8"))

    report = read(artifact_dir / "model_comparison.json")
    old_path = artifact_dir / "baseline" / "metrics.json"
    old = read(old_path if old_path.is_file() else artifact_dir / "metrics.json")
    new = read((artifact_dir / report["transformer_artifact"]).parent / "metrics.json")
    baseline, transformer = report["baseline"], report["transformer"]
    rows = baseline["test"]["rows"]
    if not report.get("identical_partitions") or rows != transformer["test"]["rows"]:
        raise ValueError("Comparison requires identical held-out partitions and row counts.")
    if old["test"] != baseline["test"] or new["test"] != transformer["test"]:
        raise ValueError("Model metrics differ from the completed comparison report.")
    paired = report["paired_test"]
    if sum(paired.values()) != rows:
        raise ValueError("Paired prediction counts do not match the test rows.")
    old_errors = paired["both_wrong"] + paired["transformer_only_correct"]
    new_errors = paired["both_wrong"] + paired["baseline_only_correct"]
    if abs(1 - old_errors / rows - baseline["test"]["accuracy"]) > 1e-9 or abs(1 - new_errors / rows - transformer["test"]["accuracy"]) > 1e-9:
        raise ValueError("Paired prediction counts disagree with test accuracy.")
    difference = 100 * (transformer["test"]["accuracy"] - baseline["test"]["accuracy"])
    reduced = old_errors - new_errors
    relative = f"{100 * reduced / old_errors:+.2f}%" if old_errors else "해당 없음"
    encoder = new["encoder"]
    lexical_count = old["actual_features"]
    dimensions = new["actual_features"] - lexical_count
    lines = [
        "=== 두 가짜 뉴스 분류 모델 비교 ===",
        "",
        "[구조와 입력]",
        f"기존: 전체 전처리 기사 → TF-IDF {lexical_count:,}개 → 로지스틱 회귀",
        f"신규: 같은 TF-IDF + EuroBERT 문맥 {dimensions:,}차원 → 로지스틱 회귀",
        f"      문맥은 제목·본문 앞 {encoder['max_length']}토큰, TF-IDF는 전체 원문 사용",
        f"      합계 {new['actual_features']:,}개 특징 / 문맥 가중치 {new['embedding_weight']}",
        "차이: 단어·표현의 빈도 특징에 주변 토큰 관계를 반영한 문맥 벡터를 추가했습니다.",
        "",
        "[학습과 판정 파라미터]",
        "기존 모델을 복사해 보존했고 TF-IDF 어휘·IDF와 원래 기사 분할을 재사용했습니다.",
        "두 모델 모두 최종 분류기는 LR입니다. 새 모델의 EuroBERT는 고정되어 있으며 전체 미세조정은 하지 않았습니다.",
        f"LR C: 기존 {old['selected_C']} / 신규 {new['selected_C']} (클래스 가중치 balanced)",
        f"가짜 판정 임계값: 기존 {old['decision_threshold']} / 신규 {new['decision_threshold']}",
        "최대 클래스 점수가 0.65 미만이면 불확실로 표시합니다.",
        f"EuroBERT 공개: {new['encoder_release_date']} / 사용 커밋 날짜: {new['encoder_revision_date'][:10]}",
        f"고정 revision: {encoder['revision']}",
        "",
        f"[동일 테스트 기사 {rows:,}건의 실측 성능]",
        "지표                  기존 TF-IDF + LR     EuroBERT + TF-IDF + LR",
    ]
    for key, title, percentage in [("accuracy", "정확도", True), ("macro_f1", "Macro F1", False),
                                    ("fake_f1", "가짜 클래스 F1", False), ("roc_auc", "ROC-AUC", False)]:
        values = [model["test"][key] for model in [baseline, transformer]]
        formatted = [f"{value:.2%}" if percentage else f"{value:.4f}" for value in values]
        lines.append(f"{title:<18} {formatted[0]:>12}         {formatted[1]:>12}")
    lines += [
        f"정확도 변화: {difference:+.2f}%p / Macro F1 변화: {transformer['test']['macro_f1'] - baseline['test']['macro_f1']:+.4f}",
        f"오답: {old_errors:,} → {new_errors:,}건 / 오답 감소 {reduced:+,}건 ({relative})",
        f"기존 오답을 신규가 교정: {paired['transformer_only_correct']:,}건",
        f"기존 정답이 신규에서 오답: {paired['baseline_only_correct']:,}건",
        f"둘 다 정답: {paired['both_correct']:,}건 / 둘 다 오답: {paired['both_wrong']:,}건",
        "",
        "[선택 기준과 해석]",
        f"검증 Macro F1: 기존 {baseline['validation']['macro_f1']:.4f} / 신규 {transformer['validation']['macro_f1']:.4f}",
        f"검증 기준 권장 모델: {'EuroBERT 파생 LR' if report['recommended_model'] == 'transformer' else '기존 TF-IDF LR'}",
        "특징 가중치·C·권장 모델은 검증 데이터로 선택했고 테스트 점수로 조정하지 않았습니다.",
        "문맥 특징 추가 후 이 내부 평가의 성능이 개선됐으나, 모든 기사에서 개선된 것은 아닙니다.",
        "",
        "[실행 비용과 판정 근거]",
        "기존은 희소 TF-IDF 계산만 필요합니다. 신규는 encoder 실행에 추가 메모리와 시간이 필요합니다.",
        f"신규 encoder 실행: {encoder.get('backend', 'torch')} / {encoder.get('device', 'CPU')} / {encoder.get('inference_precision', encoder.get('quantization', 'none'))}",
        "두 점수는 sigmoid(절편 + 특징값 × LR 계수의 합)로 계산합니다.",
        "웹은 단어별 기여도와 신규 문맥 벡터 전체의 합산 기여도를 표시합니다.",
        "문맥 합산값은 attention 중요도나 특정 문장의 인과적 설명이 아닙니다.",
        "",
        "[한계]",
        "과거 영어 Kaggle 데이터의 내부 그룹 분할 결과이며 최신·한국어·다른 언론사 성능은 별도 검증이 필요합니다.",
        "EuroBERT 사전학습 자료와 공개 벤치마크의 중복은 독립적으로 감사하지 않았습니다.",
        "점수는 학습 데이터의 패턴 점수이며 기사 내용에 대한 외부 사실 검증 결과가 아닙니다.",
    ]
    return "\n".join(lines)
