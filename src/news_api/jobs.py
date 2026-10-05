from threading import BoundedSemaphore

from sqlalchemy import update

from .models import Analysis, utc_now


class JobRunner:
    """Single local inference at a time; each job owns its database sessions."""

    def __init__(self, sessions, service):
        self.sessions = sessions
        self.service = service
        self.gate = BoundedSemaphore(1)

    def recover_interrupted(self):
        with self.sessions() as session:
            session.execute(update(Analysis).where(Analysis.status.in_(["queued", "running"])).values(
                status="failed", error="서버 재시작으로 분석이 중단됐습니다. 다시 실행해 주세요.",
                completed_at=utc_now(),
            ))
            session.commit()

    def run(self, analysis_id: str):
        with self.gate:
            with self.sessions() as session:
                job = session.get(Analysis, analysis_id)
                if job is None or job.status != "queued":
                    return
                job.status = "running"
                job.started_at = utc_now()
                arguments = (job.article_title, job.article_body, job.mode, job.language)
                session.commit()
            # Return the MySQL connection to the pool during slow model inference.
            try:
                result = self.service.analyze(*arguments)
                error = None
            except (OSError, ValueError, RuntimeError) as exception:
                result = None
                error = str(exception)[:2000]
            except Exception:
                result = None
                error = "분석 중 예상하지 못한 오류가 발생했습니다. 서버 로그와 모델 설정을 확인해 주세요."
            with self.sessions() as session:
                job = session.get(Analysis, analysis_id)
                if job is None:
                    return
                job.completed_at = utc_now()
                if error:
                    job.status = "failed"
                    job.error = error
                else:
                    job.status = "completed"
                    job.classification = result.get("classification")
                    job.summary = result.get("summary")
                session.commit()
