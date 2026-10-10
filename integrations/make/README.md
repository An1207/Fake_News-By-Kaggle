# Make 연동 파이프라인 — 비활성

현재 웹은 FastAPI를 직접 호출합니다. 기본 요약은 Ollama이며 GPT 추가 요약은 사용자가 버튼을 누를 때만 호출합니다. Make는 실행하지 않으며 계정 연결·시나리오 배포·스케줄 등록도 수행하지 않았습니다.

`pipeline.json`은 HTTP 경로·시트 열·처리 상태·두 시나리오를 정의한 **워크플로 계약 파일**입니다. Make에 바로 가져오는 blueprint가 아닙니다. FastAPI 처리 코드와 로컬 실행 클라이언트는 구현했고, 실제 Make 시나리오는 아래 계약에 맞게 계정에서 구성해야 합니다.

## 데이터 흐름

1. **intake 시나리오**: Google Sheets에서 `status=ready`인 행을 읽고 빈 본문을 제외합니다. `news_id`는 행 번호 대신 고유 값으로 관리합니다. HTTP 모듈이 `POST /api/automation/news`를 호출하고 반환된 `id`를 `analysis_id`에 저장합니다.
2. **results 시나리오**: `queued/running`인 행의 `analysis_id`로 `GET /api/automation/jobs/{analysis_id}`를 호출합니다. 완료되면 `summary.summary`를 `ollama_summary`, 처리 상태와 오류를 해당 열에 기록합니다. 반복 원문 제출 대신 결과를 조회합니다.
3. **선택적 GPT**: Ollama 완료 이후에만 `POST /api/automation/jobs/{analysis_id}/gpt-summary`를 호출합니다. 이후 같은 조회 API의 `gpt_summary.status`가 완료될 때까지 확인하고 `gpt_summary.summary.summary`, `usage`, `estimated_cost_usd`를 기록합니다. 이 단계도 기본 비활성입니다. 자동화와 웹은 같은 추가 요약 저장소를 사용하므로 같은 분석에 중복 GPT 호출을 하지 않습니다.
4. 사람은 원문과 요약의 인물·수치·날짜·부정·예정 상태를 대조해 `review_status`, `review_note`를 작성합니다. `completed`는 실행 완료이며 검수 통과를 뜻하지 않습니다.

HTTP Authorization 헤더에는 `Bearer <AUTOMATION_API_KEY>`를 설정합니다. 이 키는 OpenAI 키와 별개입니다. OpenAI 키는 백엔드에만 저장합니다. Make에서 OpenAI를 별도로 호출하면 웹과 토큰 비용·중복 제어가 분리되므로 이 계약에서는 FastAPI를 호출합니다.

## 나중에 가동할 때 필요한 설정

- 백엔드 `.env`에 `AUTOMATION_ENABLED=true`, 최소 32자 인증 키와 `AI_ENABLED=true`를 설정하고 서버를 재시작합니다. GPT 단계에는 별도로 `OPENAI_API_KEY`가 필요합니다. 현재 설정은 변경하지 않았습니다.
- Make의 Google Sheets OAuth 연결, 시트 열, HTTP 모듈 인증 정보와 요청 필드 매핑을 구성합니다. `request.example.json`은 가상 기사 요청 예입니다.
- `request_id`는 `<sheet 식별자>:<news_id>:<원문 버전>`처럼 안정적으로 지정합니다. 같은 ID·같은 요청은 기존 작업을 반환합니다. 같은 ID에 다른 본문/설정을 보내면 HTTP 409입니다. 기사를 삭제하면 연결된 중복 제어 기록도 삭제되므로 이후 같은 ID가 새 작업으로 처리됩니다.
- 실패한 요청을 새 ID로 무조건 재전송하지 않습니다. GPT 실패는 자동 재시도하지 않으며 사용량 확인 후 `?retry=true`로 명시적 재시도합니다. 429·503 오류는 시트에 기록하고 반복 호출을 제한합니다.
- Make는 클라우드 서비스여서 PC의 `127.0.0.1`에 접속할 수 없습니다. 실제 가동에는 인증을 갖춘 HTTPS 배포/접근 경로가 필요합니다. 기존 로컬 앱의 관리 API는 로그인 없이 동작하므로 전체 앱을 그대로 인터넷에 공개하면 안 됩니다. 대안은 Windows 동기화 작업이 시트를 읽고 로컬 API를 호출하는 방식이며 별도 구현 대상입니다.
- 위 설정과 연결 검증을 마친 뒤 두 Make 시나리오의 스케줄을 켜야 실제 자동화가 시작됩니다. 플래그 하나만 바꾸면 SaaS 연결이 완료되는 구조는 아닙니다.

## 로컬 클라이언트

```powershell
# 기본: 입력·계약 확인만 수행. 네트워크 요청/AI 호출 없음.
.venv\Scripts\python.exe -X utf8 scripts\automation_client.py

# 향후 활성화된 로컬 서버에만 명시적으로 제출
.venv\Scripts\python.exe -X utf8 scripts\automation_client.py --execute --input integrations\make\request.example.json
# --gpt 추가 시 Ollama 완료 후 유료 GPT 요약도 요청
```

실행 클라이언트는 인증 키를 프로세스 환경에서 읽습니다. 셸 명령 인자·시트 셀·Git에 키를 기록하지 않습니다. 기본 실행은 미리보기이며 스케줄러를 설치하지 않습니다.

연동 API의 비활성화·인증·입력 검증·중복 처리와 모의 GPT 저장/조회는 자동 테스트했습니다. 실제 Make 계정·Google Sheets·유료 GPT를 함께 연결한 운영 테스트는 수행하지 않았습니다.

공식 문서: [Make Google Sheets·OpenAI 연동](https://www.make.com/en/integrations/google-sheets/openai-gpt-3), [Make 실행 횟수](https://help.make.com/operations), [OpenAI Responses API](https://developers.openai.com/api/docs/guides/text).
