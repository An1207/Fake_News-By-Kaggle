from contextlib import asynccontextmanager
import json
from hashlib import sha256
from hmac import compare_digest
from threading import Lock
from typing import Annotated

from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import func, select, text, update
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from .ai import LocalAIService
from .config import ROOT, Settings
from .db import make_engine, make_sessions
from .jobs import JobRunner
from .models import Analysis, Article, AutomationRequest, GPTSummary, utc_now
from .schemas import AnalysisInput, AnalysisOut, AnalysisPage, ArticleInput, ArticleListItem, ArticleOut, ArticlePage, ArticleUpdate, AutomationInput


def failure(code: str, message: str, status: int = 409):
    return HTTPException(status, detail={"code": code, "message": message})


def db_session(request: Request):
    with request.app.state.sessions() as session:
        yield session


DB = Annotated[Session, Depends(db_session)]


def require_automation(request: Request, authorization: Annotated[str | None, Header()] = None):
    settings = request.app.state.settings
    if not settings.automation_enabled:
        raise failure("AUTOMATION_DISABLED", "자동화 파이프라인은 현재 비활성 상태입니다.", 503)
    key = settings.automation_api_key.get_secret_value()
    if len(key) < 32:
        raise failure("AUTOMATION_NOT_CONFIGURED", "자동화 인증 키 설정을 확인하세요.", 503)
    if not authorization or not compare_digest(authorization.encode(), ("Bearer " + key).encode()):
        raise failure("AUTOMATION_UNAUTHORIZED", "자동화 인증에 실패했습니다.", 401)


def ensure_queue_space(session: Session):
    pending = session.scalar(select(func.count()).select_from(Analysis).where(Analysis.status.in_(["queued", "running"])))
    pending += session.scalar(select(func.count()).select_from(GPTSummary).where(GPTSummary.status.in_(["queued", "running"])))
    if pending >= 10:
        raise failure("QUEUE_FULL", "분석 대기열이 가득 찼습니다. 잠시 후 다시 시도해 주세요.", 429)


def article_or_404(session: Session, article_id: str, *, lock: bool = False) -> Article:
    query = select(Article).where(Article.id == article_id)
    if lock:
        query = query.with_for_update()
    article = session.scalar(query)
    if article is None:
        raise failure("ARTICLE_NOT_FOUND", "기사를 찾을 수 없습니다.", 404)
    return article


def create_app(settings: Settings | None = None, engine=None, service=None) -> FastAPI:
    settings = settings or Settings()
    engine = engine if engine is not None else make_engine(settings)
    sessions = make_sessions(engine)
    runner = JobRunner(sessions, service if service is not None else LocalAIService(settings))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Apply Alembic migrations before starting. A missing database is never
        # silently replaced with an in-memory database in the running service.
        runner.recover_interrupted()
        yield
        engine.dispose()

    app = FastAPI(title="SAI News API", version="0.1.0", lifespan=lifespan)
    app.state.sessions = sessions
    app.state.settings = settings
    app.state.runner = runner
    app.state.enqueue_lock = Lock()
    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins,
                       allow_methods=["GET", "POST", "PUT", "DELETE"], allow_headers=["Content-Type"])

    @app.exception_handler(SQLAlchemyError)
    async def database_error(request: Request, error: SQLAlchemyError):
        return JSONResponse(status_code=503, content={"detail": {
            "code": "DATABASE_UNAVAILABLE", "message": "MySQL 연결을 확인해 주세요. 요청을 저장하지 못했습니다.",
        }})

    @app.get("/api/health")
    def health(session: DB):
        session.execute(text("SELECT 1"))
        return {"status": "ok", "database": "connected", "ai_enabled": settings.ai_enabled}

    @app.get("/api/capabilities")
    def capabilities():
        comparison_path = ROOT / "artifacts" / "model_comparison.json"
        comparison = json.loads(comparison_path.read_text(encoding="utf-8")) if comparison_path.is_file() else None
        metrics_path = settings.classifier_path.parent / "metrics.json"
        model_name = json.loads(metrics_path.read_text(encoding="utf-8"))["model"] if metrics_path.is_file() else "분류 모델"
        return {
            "ai_enabled": settings.ai_enabled,
            "classifier_artifact_present": settings.classifier_path.is_file(),
            "summarizer_model": settings.ollama_model,
            "default_summary_provider": "ollama",
            "gpt_summary_available": settings.gpt_available,
            "gpt_summary_model": "gpt-4o-mini",
            "automation_enabled": settings.automation_enabled,
            "supported_classification_language": "en",
            "summary_languages": ["ko", "en"],
            "classifier_model": model_name,
            "classification_models_present": {
                "baseline": settings.baseline_classifier_path.is_file(),
                "transformer": settings.transformer_classifier_path.is_file(),
            },
            "model_comparison": comparison,
            "message": "AI 분석 사용 가능" if settings.ai_enabled else "AI 모델 연결 준비 단계입니다. 기사 저장과 관리는 사용할 수 있습니다.",
        }

    @app.get("/api/stats")
    def stats(session: DB):
        counts = dict(session.execute(select(Analysis.status, func.count()).group_by(Analysis.status)).all())
        extras = dict(session.execute(select(GPTSummary.status, func.count()).group_by(GPTSummary.status)).all())
        return {"articles": session.scalar(select(func.count()).select_from(Article)),
                "analyses": sum(counts.values()), "completed": counts.get("completed", 0),
                "pending": counts.get("queued", 0) + counts.get("running", 0) + extras.get("queued", 0) + extras.get("running", 0),
                "failed": counts.get("failed", 0), "gpt_completed": extras.get("completed", 0)}

    @app.post("/api/articles", response_model=ArticleOut, status_code=201)
    def create_article(payload: ArticleInput, session: DB):
        values = payload.model_dump()
        values["source_url"] = str(payload.source_url) if payload.source_url else None
        article = Article(**values)
        session.add(article)
        session.commit()
        session.refresh(article)
        return article

    @app.get("/api/articles", response_model=ArticlePage)
    def list_articles(session: DB, page: int = Query(1, ge=1), page_size: int = Query(12, ge=1, le=100),
                      q: str = Query("", max_length=100)):
        query = select(Article)
        if q.strip():
            query = query.where(Article.title.contains(q.strip(), autoescape=True))
        total = session.scalar(select(func.count()).select_from(query.subquery()))
        articles = session.scalars(query.order_by(Article.created_at.desc(), Article.id.desc())
                                  .offset((page - 1) * page_size).limit(page_size)).all()
        items = [ArticleListItem(
            id=article.id, title=article.title, preview=article.body[:180], source_url=article.source_url,
            language=article.language, version=article.version, created_at=article.created_at,
            updated_at=article.updated_at,
        ) for article in articles]
        return {"items": items, "total": total, "page": page, "page_size": page_size}

    @app.get("/api/articles/{article_id}", response_model=ArticleOut)
    def get_article(article_id: str, session: DB):
        return article_or_404(session, article_id)

    @app.put("/api/articles/{article_id}", response_model=ArticleOut)
    def edit_article(article_id: str, payload: ArticleUpdate, session: DB):
        article_or_404(session, article_id)
        values = payload.model_dump(exclude={"expected_version"})
        values["source_url"] = str(payload.source_url) if payload.source_url else None
        result = session.execute(update(Article).where(
            Article.id == article_id, Article.version == payload.expected_version,
        ).values(**values, version=Article.version + 1, updated_at=utc_now()))
        if result.rowcount != 1:
            session.rollback()
            raise failure("VERSION_CONFLICT", "다른 요청에서 기사를 수정했습니다. 최신 내용을 다시 불러와 주세요.")
        session.commit()
        session.expire_all()
        return article_or_404(session, article_id)

    @app.delete("/api/articles/{article_id}", status_code=204)
    def delete_article(article_id: str, session: DB):
        article = article_or_404(session, article_id, lock=True)
        active = session.scalar(select(Analysis.id).where(
            Analysis.article_id == article_id, Analysis.status.in_(["queued", "running"]),
        ).limit(1))
        if active:
            raise failure("ANALYSIS_ACTIVE", "분석이 끝난 뒤 기사를 삭제할 수 있습니다.")
        if session.scalar(select(GPTSummary.analysis_id).join(Analysis).where(
            Analysis.article_id == article_id, GPTSummary.status.in_(["queued", "running"])).limit(1)):
            raise failure("ANALYSIS_ACTIVE", "GPT 요약이 끝난 뒤 기사를 삭제할 수 있습니다.")
        session.delete(article)
        session.commit()
        return Response(status_code=204)

    @app.post("/api/articles/{article_id}/analyses", response_model=AnalysisOut, status_code=202)
    def start_analysis(article_id: str, payload: AnalysisInput, background: BackgroundTasks, session: DB):
        if not settings.ai_enabled:
            raise failure("AI_NOT_ENABLED", "AI 모델은 다음 연결 단계에서 활성화합니다. 현재 기사 저장·관리 기능을 사용할 수 있습니다.")
        with app.state.enqueue_lock:
            article = article_or_404(session, article_id, lock=True)
            if session.scalar(select(Analysis.id).where(
                Analysis.article_id == article_id, Analysis.status.in_(["queued", "running"]),
            ).limit(1)):
                raise failure("ANALYSIS_ACTIVE", "이 기사의 분석이 이미 진행 중입니다.")
            ensure_queue_space(session)
            job = Analysis(article_id=article.id, article_version=article.version,
                           article_title=article.title, article_body=article.body,
                           mode=payload.mode, language=payload.language)
            session.add(job)
            session.commit()
            session.refresh(job)
            output = AnalysisOut.model_validate(job)
            background.add_task(runner.run, job.id)
        return output

    @app.get("/api/analyses", response_model=AnalysisPage)
    def list_analyses(session: DB, article_id: str | None = None, page: int = Query(1, ge=1),
                      page_size: int = Query(20, ge=1, le=100)):
        query = select(Analysis)
        if article_id is not None:
            article_or_404(session, article_id)
            query = query.where(Analysis.article_id == article_id)
        total = session.scalar(select(func.count()).select_from(query.subquery()))
        items = session.scalars(query.order_by(Analysis.created_at.desc(), Analysis.id.desc())
                               .offset((page - 1) * page_size).limit(page_size)).all()
        return {"items": items, "total": total, "page": page, "page_size": page_size}

    @app.get("/api/analyses/{analysis_id}", response_model=AnalysisOut)
    def get_analysis(analysis_id: str, session: DB):
        job = session.get(Analysis, analysis_id)
        if job is None:
            raise failure("ANALYSIS_NOT_FOUND", "분석 기록을 찾을 수 없습니다.", 404)
        return job

    @app.post("/api/analyses/{analysis_id}/gpt-summary", response_model=AnalysisOut, status_code=202)
    def start_gpt_summary(analysis_id: str, background: BackgroundTasks, session: DB,
                          response: Response, retry: bool = False):
        with app.state.enqueue_lock:
            parent = get_analysis(analysis_id, session)
            article_or_404(session, parent.article_id, lock=True)
            if parent.status != "completed" or parent.mode not in {"summarize", "both"} or not parent.summary:
                raise failure("OLLAMA_SUMMARY_REQUIRED", "Ollama 요약이 완료된 뒤 GPT 추가 요약을 사용할 수 있습니다.")
            extra = session.get(GPTSummary, analysis_id)
            if extra and extra.status in {"queued", "running", "completed"}:
                response.status_code = 200
                return parent
            if not settings.gpt_available:
                raise failure("GPT_NOT_CONFIGURED", "AI 활성화와 백엔드 OPENAI_API_KEY 등록 후 서버를 재시작하세요.")
            if extra and not retry:
                raise failure("GPT_RETRY_REQUIRED", "실패한 요청입니다. API 사용량 확인 후 명시적으로 재시도하세요.")
            ensure_queue_space(session)
            if extra:
                extra.status, extra.error, extra.summary = "queued", None, None
                extra.started_at = extra.completed_at = None
            else:
                session.add(GPTSummary(analysis_id=analysis_id))
            session.commit()
            session.expire_all()
            output = AnalysisOut.model_validate(get_analysis(analysis_id, session))
            background.add_task(runner.run_gpt, analysis_id)
            return output

    @app.post("/api/automation/news", response_model=AnalysisOut, status_code=202,
              dependencies=[Depends(require_automation)])
    def automation_news(payload: AutomationInput, background: BackgroundTasks, session: DB, response: Response):
        request_hash = sha256(payload.request_id.encode()).hexdigest()
        values = payload.model_dump(mode="json", exclude={"request_id"})
        payload_hash = sha256(json.dumps(values, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

        def existing_result(existing):
            if existing.payload_hash != payload_hash:
                raise failure("IDEMPOTENCY_CONFLICT", "동일 request_id에 다른 내용을 전송했습니다.")
            response.status_code = 200
            return get_analysis(existing.analysis_id, session)

        with app.state.enqueue_lock:
            existing = session.get(AutomationRequest, request_hash)
            if existing:
                return existing_result(existing)
            if not settings.ai_enabled:
                raise failure("AI_NOT_ENABLED", "자동화 처리에는 AI_ENABLED=true가 필요합니다.")
            ensure_queue_space(session)
            article = Article(**{key: values[key] for key in ("title", "body", "source_url", "language")})
            session.add(article)
            session.flush()
            job = Analysis(article_id=article.id, article_version=article.version,
                           article_title=article.title, article_body=article.body,
                           mode=payload.mode, language=payload.summary_language)
            session.add(job)
            session.flush()
            session.add(AutomationRequest(request_hash=request_hash, payload_hash=payload_hash, analysis_id=job.id))
            try:
                session.commit()
            except IntegrityError:
                session.rollback()
                existing = session.get(AutomationRequest, request_hash)
                if existing is None:
                    raise
                return existing_result(existing)
            output = AnalysisOut.model_validate(job)
            background.add_task(runner.run, job.id)
            return output

    @app.get("/api/automation/jobs/{analysis_id}", response_model=AnalysisOut,
             dependencies=[Depends(require_automation)])
    def automation_job(analysis_id: str, session: DB):
        return get_analysis(analysis_id, session)

    @app.post("/api/automation/jobs/{analysis_id}/gpt-summary", response_model=AnalysisOut, status_code=202,
              dependencies=[Depends(require_automation)])
    def automation_gpt(analysis_id: str, background: BackgroundTasks, session: DB, response: Response, retry: bool = False):
        return start_gpt_summary(analysis_id, background, session, response, retry)

    return app


app = create_app()
