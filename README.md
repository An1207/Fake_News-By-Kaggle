# Fake_News(By-Kaggle)

두 Kaggle 데이터셋으로 학습한 영어 뉴스 분류기와 Ollama 로컬 LLM 요약기를 연결한 **SAI 뉴스 워크스페이스**입니다. React에서 기사를 관리하고, FastAPI가 분류·요약을 실행하며, MySQL에 기사와 분석 이력을 저장합니다.

> 분류 점수는 학습 데이터의 문체·어휘 패턴에 따른 모델 점수입니다. 사실의 참·거짓 확률이 아니며, 요약도 외부 사실 검증을 수행하지 않습니다.

![실제 분류·요약 이력 화면](artifacts/web-preview.png)

## 구현한 기능

- 기사 등록·조회·제목 검색·페이지 구분·수정·삭제, 원문 링크와 언어 표시
- 기사 버전 확인과 오래된 수정 요청 거부(HTTP 409)
- TF-IDF + 로지스틱 회귀 영어 뉴스 분류, 부적절한 입력 보류, 불확실한 결과 표시
- 기존 모델을 복사해 보존하고 EuroBERT 문맥 특징을 추가한 파생 로지스틱 회귀 학습, 동일 분할의 성능 비교
- 같은 기사에서 두 모델의 점수·판정·단어별 기여도·실제 파라미터를 비교하고 한 칸에서 예측 과정 설명
- Ollama `exaone3.5:2.4b` 뉴스 요약, 한국어·영어 출력, 긴 기사 분할·통합
- 요청 시점의 기사 원문·버전 스냅샷과 분석 결과 JSON 저장, 상태 자동 갱신
- MySQL 컨테이너, Alembic 마이그레이션, 설치·실행 스크립트, 단위·통합 검증
- 기존 Streamlit 프로토타입과 다운로드·학습·분류·요약 CLI 유지

개발 환경에서 실제 AI 연결을 검증했습니다. 새 설치의 기본값은 `AI_ENABLED=false`입니다. 데이터를 다운로드해 분류기를 학습하고 Ollama를 준비한 뒤 활성화합니다.

## 기술과 구조

| 영역 | 기술 |
| --- | --- |
| 프론트엔드 | React 19, TypeScript, Vite |
| API | FastAPI, Pydantic, Uvicorn |
| DB | MySQL 8.4, SQLAlchemy, PyMySQL, Alembic |
| 분류 | scikit-learn, TF-IDF, LogisticRegression, EuroBERT, OpenVINO/ONNX |
| 요약 | Ollama, EXAONE 3.5 2.4B |
| 개발·검증 | uv, npm, Docker Compose, pytest, HTTPX |

```mermaid
flowchart LR
    UI[React 뉴스 워크스페이스] -->|/api| API[FastAPI]
    API --> DB[(MySQL: 기사 · 분석 이력)]
    API --> Jobs[분석 작업 실행기]
    Jobs --> Classifier[두 모델: TF-IDF LR + EuroBERT 파생 LR]
    Classifier --> Evidence[각 점수 · 기여도 · 판정 기준]
    Evidence --> DB
    Jobs --> Ollama[Ollama / EXAONE]
    Jobs --> DB
    Kaggle[Kaggle CSV] --> Prepare[라벨 확인 · 중복 제거 · 그룹 분할]
    Prepare --> Train[학습 · 검증 · 테스트]
    Train --> Classifier
```

```text
frontend/src/                React 화면, 타입이 있는 API 호출, 반응형 CSS
src/news_api/
  main.py                    API 라우트, CORS, DB 오류 처리
  models.py / schemas.py     DB 모델, 입력 검증, 응답 형식
  config.py / db.py          환경 설정, DB 엔진·세션
  jobs.py                    작업 상태, 추론 직렬화, 재시작 처리
  ai.py                      기존 분류·요약 모델 연결 어댑터
src/news_ai/
  data.py                    KaggleHub 다운로드, 라벨·중복·그룹 점검
  text.py                    학습·예측 공통 전처리
  model.py                   학습, 평가, 모델 저장, 분류 추론
  transformer.py             원본 모델 복사, 특징 캐시, EuroBERT 파생 LR 학습·비교
  onnx_encoder.py            원본 특징 검증, ONNX 내보내기, Intel GPU/CPU 실행
  selection.py               검증 성능에 따른 기본 분류기 선택
  summarizer.py              Ollama 요약, 긴 기사 분할·통합
  cli.py                     다운로드·학습·분류·요약 명령
backend/migrations/          Alembic 마이그레이션
scripts/                     설정·실행·실제 연동 검증
tests/                       데이터·분류·요약·웹 API 테스트
artifacts/metrics.json       최초 학습의 집계 평가 결과
artifacts/data_audit.json    최초 데이터 점검의 집계 결과
compose.yaml                 프로젝트 전용 MySQL
app.py                       Streamlit 프로토타입
pyproject.toml / uv.lock     Python 의존성과 고정 버전
frontend/package-lock.json   프론트엔드 고정 버전
```

## 설치와 실행

### 1. 환경 준비

아래는 **Windows PowerShell** 기준입니다. Python 3.11–3.13, [uv](https://docs.astral.sh/uv/), Node.js 22.12 이상 또는 24 계열, Docker Desktop과 Git이 필요합니다. Docker Desktop을 먼저 실행합니다.

```powershell
git clone https://github.com/An1207/Fake_News-By-Kaggle.git
cd Fake_News-By-Kaggle
.\scripts\setup_web.ps1
```

설정 스크립트는 고정 버전의 Python·npm 의존성을 설치하고, `.env` 생성, MySQL 실행·상태 확인, DB 마이그레이션, 프론트엔드 빌드를 수행합니다. 실행 정책이 로컬 스크립트를 막으면 현재 터미널에만 `Set-ExecutionPolicy -Scope Process Bypass`를 적용할 수 있습니다.

`.env`가 없으면 `scripts/init_web_env.py`가 프로젝트 전용 DB 비밀번호를 생성합니다. 기존 파일을 덮어쓰지 않고 비밀번호를 출력하지 않습니다. `.env.example`은 변수 참고용입니다.

### 2. 웹 서비스 시작

프로젝트 폴더에서 두 터미널을 사용합니다.

```powershell
# 터미널 1: FastAPI (단일 worker)
.\scripts\run_api.ps1

# 터미널 2: React
.\scripts\run_frontend.ps1
```

| 서비스 | 주소 |
| --- | --- |
| React 화면 | http://127.0.0.1:5173 |
| FastAPI 문서 | http://127.0.0.1:8001/docs |
| API 상태 | http://127.0.0.1:8001/api/health |
| MySQL | `127.0.0.1:13306`, DB `sai_news` |

React 개발 서버가 `/api`를 FastAPI로 프록시합니다. DB·API 포트는 최초 환경의 기존 서비스와 충돌하지 않도록 13306·8001을 선택했습니다.

컴퓨터 재시작 후에는 `docker compose -p sai-news up -d mysql`로 DB를 실행한 뒤 두 실행 스크립트를 사용합니다. 데이터는 named volume `sai-news_mysql_data`에 유지되며 `docker compose -p sai-news down`으로 컨테이너를 종료해도 보존됩니다.

DB 포트는 `.env`의 `MYSQL_PORT`로 바꾸고 Compose를 다시 적용합니다. API 포트 변경 시 `scripts/run_api.ps1`과 `frontend/vite.config.ts`의 프록시를 함께 수정합니다. 서비스는 로그인 없는 **로컬 개발용**으로 `127.0.0.1`에 바인딩합니다. 외부 배포에는 인증·권한·배포 구성이 필요합니다.

### 3. 데이터 다운로드와 분류기 학습

원본 CSV와 모델 바이너리는 Git에 포함하지 않습니다. Kaggle 이용 조건과 필요한 인증을 준비한 뒤 실행합니다.

```powershell
.venv\Scripts\python.exe -m news_ai download --data-dir data/raw
.venv\Scripts\python.exe -m news_ai train --data-dir data/raw --output-dir artifacts
```

다운로드 명령은 제공된 API를 호출합니다.

```python
import kagglehub

isot_path = kagglehub.dataset_download("clmentbisaillon/fake-and-real-news-dataset")
welfake_path = kagglehub.dataset_download("saurabhshahane/fake-news-classification")
```

CLI는 `Fake.csv`, `True.csv`, `WELFake_Dataset.csv`를 지정한 폴더로 복사하고 기존 대상 파일을 보존합니다. CSV가 프로젝트 루트에 있으면 `--data-dir .`으로 학습합니다. 원본은 이동·수정하지 않습니다.

학습 산출물은 `classifier.joblib`, `metrics.json`, `data_audit.json`, `split_manifest.csv.gz`, `test_errors.csv`입니다. 비교 실험은 `--output-dir artifacts/experiment-02`처럼 별도 폴더를 사용합니다. joblib은 pickle 기반이므로 신뢰하는 파일만 불러오세요.

### 4. Ollama와 AI 활성화

[Ollama](https://ollama.com/download/windows)를 설치하고 실행한 뒤 모델을 준비합니다.

```powershell
ollama pull exaone3.5:2.4b
.venv\Scripts\python.exe -m news_ai ollama-status
```

`.env`를 다음처럼 설정하고 FastAPI를 재시작합니다.

```dotenv
AI_ENABLED=true
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=exaone3.5:2.4b
OLLAMA_TIMEOUT=180
```

기사를 저장하고 **패턴 분류** 또는 **내용 요약** 버튼을 사용합니다. 분류는 영어 기사 대상, 요약은 한국어·영어 출력입니다. 다른 모델은 `OLLAMA_MODEL`로 지정합니다. 로컬 HTTP Ollama 주소만 허용합니다.

EXAONE 2.4B 다운로드 크기는 약 1.6GB이며 실행에는 추가 메모리가 필요합니다. 이용 조건은 [공식 모델 페이지](https://ollama.com/library/exaone3.5:2.4b)와 [EXAONE 라이선스](https://github.com/LG-AI-EXAONE/EXAONE-3.5/blob/main/LICENSE)를 확인하세요.

## 데이터 점검과 학습

| 데이터셋 | 라벨 확인 |
| --- | --- |
| [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) | `Fake.csv`=fake, `True.csv`=real |
| [Fake News Classification / WELFake](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification) | 동일 본문을 Fake/True와 대조해 label 방향 확인 |

공통 라벨은 **0=real, 1=fake**입니다. 최초 CSV의 WELFake는 **1=fake, 0=real**로 확인됐고, 최소 본문 길이를 만족한 43,775행 대조에서 100% 일치했습니다. [저자 데이터 설명](https://zenodo.org/records/4561253)의 라벨 설명과 반대여서 문구만으로 라벨을 정하지 않습니다.

기본 `auto`는 동일 본문 50건 이상에서 95% 이상 일치하는 방향을 채택합니다. 근거가 부족하거나 명시적인 `--welfake-fake-label 0/1` 설정이 충분한 대조 근거와 충돌하면 중단합니다.

1. 결측·HTML·유니코드·공백을 정리하고, 본문 영어 단어가 20개 미만인 행을 제외합니다.
2. 동일 본문의 라벨 충돌 그룹 전체를 제외하고, 본문 중복을 제거하며 데이터셋 포함 여부를 보존합니다.
3. 상당한 길이의 제목 또는 첫 80단어가 같은 기사를 연결 그룹으로 묶습니다.
4. 그룹을 유지하며 약 80/10/10 학습·검증·테스트로 나눕니다. 그룹 크기에 따라 실제 비율은 달라집니다.
5. Reuters 출처 토큰·앞 dateline과 URL을 제거합니다. `subject`, `date`, CSV 인덱스, 데이터셋 이름은 특징으로 사용하지 않습니다.
6. 학습 데이터에서만 TF-IDF 어휘·IDF를 계산하고 검증 Macro F1으로 `C ∈ {0.5, 1, 2}`를 선택한 뒤 테스트에서 평가합니다. 선택 모델은 학습 세트로 학습한 상태로 저장합니다.

본문 해시와 연결 그룹의 분할 간 중복을 검사합니다. 모든 의역·사건 단위 중복을 제거하지는 않습니다. 두 데이터셋에는 같은 기사가 많아 데이터셋별 점수도 독립적인 교차 데이터셋 평가가 아닙니다.

## EuroBERT 파생 모델과 비교

[EuroBERT](https://manuelfay.github.io/news/release_eurobert/)는 2025년 3월 10일 공개됐으며, [논문](https://arxiv.org/abs/2503.05500)은 3월 7일 공개됐습니다. 이 프로젝트는 **EuroBERT-210M**의 **2025년 4월 17일 커밋 `6720a49834876b4ec5afcf6ef0d149a2c6501cce`**를 사용합니다. 사전학습 모델 공개본이 2025년 5월 이전 조건을 충족하며, 실행 라이브러리는 현재의 고정 버전을 사용합니다.

기존 `artifacts/classifier.joblib`은 유지하고 `artifacts/baseline/`에 모델·평가·분할 목록을 복사한 뒤 SHA-256을 대조합니다. 기존 TF-IDF 어휘와 IDF를 그대로 가져오고, 제목과 본문 앞 **128토큰**에서 추출한 EuroBERT 특징 768개를 추가합니다. TF-IDF는 전체 전처리 기사를 읽습니다. **EuroBERT 가중치는 고정하고 최종 로지스틱 회귀를 새로 학습**하는 특징 전이 방식이며, Transformer 전체 미세조정은 하지 않습니다.

### Transformer가 추가하는 정보

TF-IDF는 단어와 두 단어 표현의 등장 빈도·희소성을 수치화합니다. Transformer encoder는 self-attention으로 입력 토큰 사이의 관계를 반영해 문맥 벡터를 만듭니다. 같은 단어도 주변 표현에 따라 다른 벡터를 만들 수 있어 기존 빈도 특징을 보완합니다. EuroBERT는 생성형 요약 모델이 아니라 문서 분류·검색·임베딩에 사용하는 양방향 encoder이며, 뉴스 요약에는 기존 Ollama EXAONE을 계속 사용합니다.

이 실험에서는 마지막 토큰 표현을 attention mask로 평균 내고 L2 정규화한 **768차원 벡터**를 만듭니다. 기존 **60,000차원 TF-IDF**에 검증으로 선택한 가중치 **0.5**를 곱한 문맥 벡터를 붙여 **60,768개 특징**을 LR에 입력합니다. EuroBERT는 약 2.1억 파라미터의 공개 사전학습 모델이고, 프로젝트에서 새로 학습한 부분은 최종 LR의 계수와 절편입니다. 사전학습 전체를 다시 수행하지 않았습니다. 128토큰 제한은 현재 PC의 메모리·실행 시간에 맞춘 선택으로, 모델 자체의 최대 문맥 길이를 의미하지 않습니다.

새 분할을 만들지 않고 원래 `split_manifest.csv.gz`의 기사·본문 해시·그룹·라벨을 대조합니다. 학습 49,236건, 검증 6,155건, 테스트 6,154건을 공유합니다. 특징 가중치 `{0.5, 1, 2}`와 LR `C ∈ {0.5, 1, 2}`는 **검증 Macro F1**으로만 선택합니다. 기본 분류기 추천도 검증 성능으로 결정한 뒤 테스트를 평가합니다.

<!-- model-comparison:start -->
두 모델은 **동일한 테스트 기사 6,154건**으로 비교했습니다.

| 모델 | 테스트 정확도 | Macro F1 | fake F1 | ROC-AUC |
| --- | --- | --- | --- | --- |
| 기존 TF-IDF + LR | **96.34%** | **0.9629** | 0.9587 | 0.9938 |
| EuroBERT + TF-IDF + LR | **98.16%** | **0.9814** | 0.9792 | 0.9984 |

새 모델의 정확도 차이는 **+1.82%p**이고 Macro F1 차이는 **+0.0184**입니다. 기존 모델의 오답 중 153건을 바로잡았고, 기존 모델이 맞힌 기사 중 41건은 새 모델에서 오답이 됐습니다.

검증 Macro F1은 기존 **0.9659**, 새 모델 **0.9832**입니다. 검증 기준 권장 모델은 **EuroBERT 파생 LR**입니다. 새 모델은 특징 가중치 **0.5**, **C=2.0**를 선택했습니다.

집계 결과: [모델 비교](artifacts/model_comparison.json), [새 모델 평가](artifacts/eurobert/metrics.json), [원본 보존 평가](artifacts/baseline/metrics.json). 본문·그룹이 겹치지 않는 동일 분할의 내부 평가이며, 최신 기사나 다른 언론사의 성능을 보장하지 않습니다.
<!-- model-comparison:end -->

```powershell
# 기존 baseline 학습이 완료된 프로젝트에서 실행; 중단 후 같은 명령으로 재개
powershell -ExecutionPolicy Bypass -File scripts/train_transformer.ps1
# 다른 데이터 폴더를 사용하는 경우: ... -File scripts/train_transformer.ps1 -DataDir data/raw

# 원본 모델 명시적 선택
.venv\Scripts\python.exe -m news_ai predict --model artifacts/classifier.joblib --file article.txt

# 파생 모델 명시적 선택
.venv\Scripts\python.exe -m news_ai predict --model artifacts/eurobert/classifier.joblib --file article.txt

# 구조·학습·파라미터·정확도·오답 변화·한계를 CLI로 출력 (모델/LLM 실행 없음)
.venv\Scripts\python.exe -X utf8 -m news_ai compare
# 다른 평가 폴더: ... -m news_ai compare --artifact-dir artifacts
```

새 모델은 `artifacts/eurobert/classifier.joblib`, 집계 평가는 `artifacts/eurobert/metrics.json`에 저장합니다. `.hf-cache/`와 ONNX 파일, 특징 캐시는 로컬에 보관하고 Git에는 올리지 않습니다. 새 환경에서는 데이터와 공개 가중치를 내려받고 다시 학습해야 합니다.

원본 PyTorch FP32와 ONNX의 입력 토큰·특징을 대조한 후 실행합니다. Intel GPU에서는 OpenVINO의 혼합 FP16 실행을 사용하고, GPU가 없는 환경에서는 CPU를 선택합니다. 실제 GPU 특징 추출 검증은 원본과 코사인 유사도 0.999 이상을 확인했습니다. INT8 활성값 양자화는 이 모델에서 특징 일치도 검증을 통과하지 못해 사용하지 않습니다. 장치·배치에 따른 작은 수치 차이와 기존보다 긴 추론 시간이 발생할 수 있습니다.

`news-ai predict`와 Streamlit은 완료된 비교 기록을 읽어 권장 모델을 기본으로 사용합니다. FastAPI의 분류 요청은 **기존 LR와 EuroBERT 파생 LR를 모두 실행**하고 같은 기사 버전의 두 결과를 MySQL JSON 이력에 저장합니다. `.env`의 `CLASSIFIER_PATH`는 기존 API 최상위 결과에 사용할 기본 분류기 경로를 우선하며, 두 결과는 `classification.models.baseline` / `classification.models.transformer`에 별도로 포함됩니다. FastAPI는 학습 완료 후 재시작해야 새 모델이 적용됩니다. Streamlit 사이드바에서는 두 모델을 직접 선택할 수 있습니다.

### 웹에서 두 모델의 판정과 예측 과정 확인

React에서 기사를 저장한 뒤 **두 모델 비교**를 누르면 각 모델의 판정, 가짜·진짜 패턴 점수, 판정 일치 여부를 함께 표시합니다. 이전 단일 모델 이력도 계속 볼 수 있습니다. 모델 파일이 없는 경우 해당 결과 카드에 준비되지 않았음을 표시하며 한국어·짧은 원문·어휘 불일치 입력은 분류를 보류합니다.

각 결과에는 판정 임계값 **0.5**, 불확실 기준 **최대 클래스 점수 0.65 미만**, 정규화 **C=2**, 클래스 가중치 **balanced**, 어휘 특징 수와 일치 수를 표시합니다. 새 모델에는 문맥 **768차원**, **128토큰 제한**, 특징 가중치 **0.5**, encoder 고정 여부를 추가합니다. `C`는 정규화 강도의 역수이며 커질수록 정규화가 약해집니다.

판정 근거는 해당 기사에서 실제 계산한 **특징값 × LR 계수**입니다. 가짜 방향·진짜 방향의 주요 단어/표현을 각각 최대 5개 표시하고, 전체 TF-IDF 기여도·문맥 벡터의 합산 기여도·절편을 함께 보여줍니다. 합은 `z = intercept + TF-IDF 기여도 + EuroBERT 기여도`, 가짜 패턴 점수는 `sigmoid(z) = 1 / (1 + exp(-z))`입니다. 기여도 단위는 log-odds이며 양수는 가짜 패턴 쪽, 음수는 진짜 패턴 쪽으로 움직입니다. 상위 단어 목록만으로 전체 합을 재구성할 수는 없습니다.

결과 아래 **AI는 이렇게 예측했습니다** 한 칸에서 입력 점검 → 특징 추출 → 가중합·sigmoid → 50% 기준 판정을 설명합니다. 이 설명은 실제 계산에 기반하며 LLM이 임의로 만든 판단 사유가 아닙니다. 문맥 기여도는 encoder 출력 전체의 선형 합산값으로, attention 중요도나 특정 문장의 인과적 설명이 아닙니다. 두 모델 모두 외부 근거를 찾아 사실 확인하는 모델은 아닙니다.

주요 코드: [예측·기여도 계산](src/news_ai/model.py), [두 모델 API 연결](src/news_api/ai.py), [결과 UI](frontend/src/Predictions.tsx), [EuroBERT 학습](src/news_ai/transformer.py). 실제 저장 모델의 원본 보존·추론 검증 결과는 [inference_verification.json](artifacts/eurobert/inference_verification.json)에 기록했습니다.

실제 로컬 웹에서 가상 기사로 확인한 두 모델 결과와 예측 과정:

![두 모델 예측 결과와 파라미터·예측 과정](docs/screenshots/dual-model-results.jpg)

## 최초 학습 결과

집계 평가와 데이터 점검은 [metrics.json](artifacts/metrics.json), [data_audit.json](artifacts/data_audit.json)에 포함했습니다. 다운로드 버전·환경·seed가 달라지면 재학습 결과도 달라질 수 있습니다.

| 항목 | 결과 |
| --- | --- |
| 원본 행 | 117,032 |
| 본문 결측·길이 부족 제외 | 3,131 |
| 중복 본문 제외 | 52,356 |
| 학습 가능한 고유 기사 | 61,545 |
| 두 데이터셋에 함께 있는 고유 기사 | 38,017 |
| 학습 / 검증 / 테스트 | 49,236 / 6,155 / 6,154 |
| TF-IDF 특징 / 선택 C | 60,000 / 2.0 |
| 테스트 정확도 | **96.34%** |
| 테스트 Macro F1 | **0.9629** |
| 테스트 fake F1 | 0.9587 |
| 테스트 ROC-AUC | 0.9938 |
| 분할 간 본문·연결 그룹 중복 | 0 |

이 결과는 **과거 영어 데이터 내부 그룹 분할 성능**입니다. 한국어·최신 뉴스·언론사·시간별 외부 일반화 성능으로 해석할 수 없습니다. 정리된 ISOT 기사는 모두 WELFake에도 포함돼 원본 117,032행을 고유 기사 수로 볼 수 없습니다.

## API와 분석 이력

| 메서드·경로 | 기능 |
| --- | --- |
| `GET /api/health`, `/api/capabilities`, `/api/stats` | DB·AI 설정·집계 현황 |
| `POST /api/articles` | 기사 등록 |
| `GET /api/articles?page=1&q=제목` | 검색·페이지 구분 |
| `GET /api/articles/{id}` | 원문 조회 |
| `PUT /api/articles/{id}` | `expected_version` 확인 후 수정 |
| `DELETE /api/articles/{id}` | 기사·연결 이력 삭제 |
| `POST /api/articles/{id}/analyses` | `classify`, `summarize`, `both` 분석 |
| `GET /api/analyses`, `/api/analyses/{id}` | 분석 이력·상태 조회 |

기사에는 제목, 본문, URL, 언어, 버전, UTC 시간을 저장합니다. 본문은 20–59,000자이고 LONGTEXT로 한글·이모지·긴 원문을 지원합니다. URL은 참고값이며 자동 수집하지 않습니다.

분석 요청은 제목·본문·버전 스냅샷을 저장해 이후 기사 수정이 이전 분석 원문을 바꾸지 않습니다. 상태는 `queued → running → completed/failed`이고, 재시작으로 남은 작업은 실패 처리해 사용자가 다시 실행할 수 있게 합니다. 진행 중인 분석이 있으면 기사 삭제를 거부합니다.

로컬 AI 호출은 한 번에 하나, 대기·진행 작업은 최대 10개입니다. 현재 실행기는 **단일 FastAPI 프로세스용**이며, 여러 worker·서버에서는 별도 작업 큐와 복구 정책이 필요합니다.

## 검증

```powershell
# 단위·API 테스트 (실제 LLM 대신 모의 응답)
.venv\Scripts\python.exe -m pytest -q

# 파생 모델 학습 완료 후 원본 보존·어휘/IDF 복사·실제 추론 확인
.venv\Scripts\python.exe scripts\verify_transformer.py

# 실행 중인 웹 서비스와 실제 MySQL 검증
.venv\Scripts\python.exe -X utf8 scripts\verify_web.py

# 실제 두 모델 예측·근거 계산·MySQL JSON 재조회 검증 (가상 예시 기사를 남김)
.venv\Scripts\python.exe -X utf8 scripts\verify_dual_web.py

# React에서 예시 기사를 저장한 뒤 실제 분류기·Ollama 검증
.venv\Scripts\python.exe -X utf8 scripts\verify_ai_web.py

# TypeScript·프로덕션 빌드
cd frontend
npm.cmd run build
```

- **Python 테스트 41개 통과**: 기존 28개 + 파생 모델·선택·특징 캐시·토크나이저 6개 + 비교 API 1개 + 선형 근거·두 모델 연결·파일 부재·JSON 저장 4개 + 비교 CLI 2개. 라벨·그룹 누수·입력 보류·요약 요청·실패 처리·수정 충돌·스냅샷·재시작 복구·평가 파일 불일치 거부 등을 확인했습니다.
- **TypeScript·React 빌드 통과**, 브라우저 등록·수정·검색·분류 요청·결과 표시 확인, 데스크톱·모바일 화면 확인
- **실제 MySQL 통합 검증 통과**: 한글·이모지·64KiB 초과 UTF-8 본문, 별도 연결의 저장 유지, JSON 이력, 버전 충돌, 기사 삭제 시 이력 cascade
- **실제 분류기 + EXAONE 웹 연결 통과**: 가상 도서관 기사에서 분류·한국어 요약, DB 저장·재조회 확인. 첫 호출 약 34.5초
- **실제 두 모델 웹 연결 통과**: 같은 가상 기사에서 두 분류기 실행, 각 logit 기여도 합·sigmoid 점수 일치, MySQL JSON 저장·재조회 확인. 12.56초에 완료했고 데스크톱에서 두 카드와 설명, 모바일 390px에서 세로 배치·가로 넘침 없음·콘솔 오류 없음을 확인했습니다. 이 가상 기사의 점수는 모델 정확도 평가에 사용하지 않았습니다.

`verify_web.py`는 자체 테스트 기사만 생성·검증·제거합니다. `verify_ai_web.py`는 “예시 기사 입력”으로 저장한 가상 기사에 실제 분석 이력을 추가합니다. 로컬 결과 `artifacts/web-smoke*.json`은 공개 Git에서 제외합니다.

`verify_dual_web.py`는 “두 모델 비교 테스트: 가상 도서관 보수 계획” 기사를 남겨 UI에서 결과를 확인할 수 있게 합니다. 기존 사용자 기사는 수정·삭제하지 않으며 요약 LLM은 호출하지 않습니다.

## 해석 범위와 관찰된 한계

- 분류 점수는 보정되지 않은 패턴 점수입니다. 한국어 비중이 높은 입력, 짧은 영어 입력, 학습 어휘가 전혀 겹치지 않는 입력은 보류합니다. 문자 비율 검사는 정교한 언어 판별기가 아닙니다.
- 요약에 분류 결과를 전달하지 않습니다. 원문의 주장·인물·숫자·예정 상태 보존을 지시하지만 환각·누락·번역 오류를 막는다는 보장은 없습니다.
- 기본 분할 크기는 영어 비중이 높은 기사 6,000자, 비ASCII 비중이 높은 기사 3,000자입니다. 최대 60,000자까지 처리하고 초과하면 오류를 반환합니다. 웹 입력 한도는 59,000자입니다.
- 데이터셋에 정답 요약문이 없어 요약 모델 미세 조정·ROUGE 평가를 수행하지 않았습니다.
- 초기 Qwen 2.5 3B 비교에서 한국어 인명·정당 오류를 관찰해 EXAONE을 기본값으로 선택했습니다. EXAONE에서도 고유명사·발언 주체·원문에 없는 해석 오류를 관찰했습니다.
- 웹 연결 샘플은 원문 “next September”에 없는 “초”를 추가하고 예산을 반복했습니다. 실행 성공과 요약 충실도는 별도로 평가해야 합니다.
- 약 8GB RAM·CPU 환경의 EXAONE 표본에서 짧은 기사 약 25초, 약 4,700자 단일 요약 약 101초, 2조각·통합 약 136초를 측정했습니다. 환경·기사 길이에 따라 달라집니다.

다음 실험은 최신 뉴스·다른 언론사·시기별 외부 평가, 오분류 분석, 같은 분할의 임베딩/신경망 비교, 한국어 라벨 데이터, 요약 사실 일치·누락 평가입니다.

## CLI와 Streamlit

```powershell
.venv\Scripts\python.exe -m news_ai prepare --data-dir data/raw --output-dir artifacts
.venv\Scripts\python.exe -m news_ai predict --title "Article title" --file article.txt
.venv\Scripts\python.exe -m news_ai summarize --title "Article title" --file article.txt --language ko
.\scripts\run_app.ps1
```

Streamlit 프로토타입은 http://127.0.0.1:8501 에서, 통합 React 서비스는 5173에서 실행합니다. CLI와 Streamlit은 프로세스 환경 변수 또는 화면 설정을 사용하고 FastAPI는 프로젝트 `.env`를 읽습니다.

## 공개 범위와 이용 조건

소스·테스트·마이그레이션·스크립트·의존성 잠금 파일·집계 평가 JSON·가상 기사의 웹 화면을 포함합니다. **원본 CSV, 모델 바이너리, 원문이 포함된 추론 기록, `.env`, DB 데이터, 캐시·설치 파일은 제외**합니다. 재현에는 다운로드·학습·설정이 필요합니다.

데이터와 LLM 이용 조건은 각 Kaggle 데이터셋 페이지와 EXAONE 라이선스를 따로 확인하세요. 공개 저장소라는 사실만으로 해당 데이터·모델의 이용 조건이 바뀌지는 않습니다.
