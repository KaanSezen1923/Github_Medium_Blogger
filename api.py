import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import asynccontextmanager
import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import time
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import text
from sqlalchemy.orm import Session
import uvicorn

from agents.analyzer_agent import AnalyzerAgent
from agents.planner_agent import DEFAULT_AUDIENCE, DEFAULT_TONE, PlannerAgent
from agents.writer_agent import WriterAgent
from core.github_fetcher import GitHubFetcher, parse_repo_url
from core.utils import SNIPPET_TOKEN, final_checks, partition_repo_files
from database import engine, get_db
import models

load_dotenv()

# --- Logging Yapılandırması ---
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("github_medium_api")

# --- Sabitler ve Güvenlik Ayarları ---
SECRET_KEY = os.getenv("JWT_SECRET_KEY") or os.getenv("SECRET_KEY") or "prod-change-me-github-medium-secret-key"
TOKEN_EXPIRE_SECONDS = int(os.getenv("TOKEN_EXPIRE_SECONDS", str(60 * 60 * 24 * 7)))  # 7 gün
PBKDF2_ITERATIONS = 100_000
MAX_OUTLINE_ATTEMPTS = 3
EMAIL_REGEX = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

security_scheme = HTTPBearer(auto_error=False)


# --- Lifespan (Başlangıç / Kapanış Yönetimi) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("API başlatılıyor, veritabanı tabloları kontrol ediliyor...")
    try:
        models.Base.metadata.create_all(bind=engine)
        logger.info("Veritabanı tabloları hazır.")
    except Exception as e:
        logger.error(f"Veritabanı başlangıç kontrolünde hata: {e}")
    yield
    logger.info("API kapatılıyor...")


app = FastAPI(
    title="GitHub to Medium API",
    description="AI Agent tabanlı kod analizi ve blog üretim backend'i",
    version="1.0.0",
    lifespan=lifespan,
)

# --- CORS Middleware ---
cors_origins_env = os.getenv("CORS_ORIGINS", "*")
allowed_origins = [o.strip() for o in cors_origins_env.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- İstek Süresi ve Loglama Middleware ---
@app.middleware("http")
async def log_requests_middleware(request: Request, call_next):
    start_time = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception as exc:
        logger.exception(f"İşlenmeyen sunucu hatası: {request.method} {request.url.path} -> {exc}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Sunucu tarafında beklenmeyen bir hata oluştu."},
        )
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"
    logger.info(f"{request.method} {request.url.path} - {response.status_code} ({elapsed_ms:.1f} ms)")
    return response


# --- Şifreleme ve JWT Yardımcıları ---
def hash_password(password: str) -> str:
    """PBKDF2-HMAC-SHA256 ile tuzlanmış (salted) güvenli şifre özeti üretir."""
    salt = secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), PBKDF2_ITERATIONS)
    return f"pbkdf2_sha256${PBKDF2_ITERATIONS}${salt}${dk.hex()}"


def verify_password(plain_password: str, stored_password: str) -> bool:
    """PBKDF2, eski SHA-256 ve düz metin şifrelerle geriye dönük uyumlu doğrulama yapar."""
    if not stored_password:
        return False
    if stored_password.startswith("pbkdf2_sha256$"):
        try:
            _, iter_str, salt, hash_hex = stored_password.split("$", 3)
            dk = hashlib.pbkdf2_hmac(
                "sha256",
                plain_password.encode("utf-8"),
                salt.encode("utf-8"),
                int(iter_str),
            )
            return hmac.compare_digest(dk.hex(), hash_hex)
        except Exception:
            return False

    legacy_sha256 = hashlib.sha256(plain_password.encode("utf-8")).hexdigest()
    if hmac.compare_digest(stored_password, legacy_sha256):
        return True
    return hmac.compare_digest(stored_password, plain_password)


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(data: str) -> bytes:
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + padding)


def create_access_token(user_id: str, email: str) -> str:
    header = {"alg": "HS256", "typ": "JWT"}
    payload = {
        "sub": user_id,
        "email": email,
        "iat": int(time.time()),
        "exp": int(time.time()) + TOKEN_EXPIRE_SECONDS,
    }
    signing_input = f"{_b64url_encode(json.dumps(header).encode())}.{_b64url_encode(json.dumps(payload).encode())}"
    signature = hmac.new(SECRET_KEY.encode("utf-8"), signing_input.encode("ascii"), hashlib.sha256).digest()
    return f"{signing_input}.{_b64url_encode(signature)}"


def decode_access_token(token: str) -> Optional[dict]:
    try:
        header_b64, payload_b64, sig_b64 = token.split(".")
        signing_input = f"{header_b64}.{payload_b64}"
        expected_sig = hmac.new(SECRET_KEY.encode("utf-8"), signing_input.encode("ascii"), hashlib.sha256).digest()
        if not hmac.compare_digest(_b64url_encode(expected_sig), sig_b64):
            return None
        payload = json.loads(_b64url_decode(payload_b64).decode("utf-8"))
        if payload.get("exp", 0) < int(time.time()):
            return None
        return payload
    except Exception:
        return None


def get_optional_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security_scheme),
    db: Session = Depends(get_db),
) -> Optional[models.User]:
    if not credentials or not credentials.credentials:
        return None
    payload = decode_access_token(credentials.credentials)
    if not payload or not payload.get("sub"):
        return None
    return db.query(models.User).filter(models.User.user_id == payload["sub"]).first()


# --- Pydantic Şemaları ---
class SignupRequest(BaseModel):
    username: str = Field(..., min_length=2, max_length=100)
    email: str = Field(..., max_length=255)
    password: str = Field(..., min_length=6, max_length=128)

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        email_clean = v.strip().lower()
        if not EMAIL_REGEX.match(email_clean):
            raise ValueError("Geçerli bir e-posta adresi girin.")
        return email_clean

    @field_validator("username")
    @classmethod
    def validate_username(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Kullanıcı adı boş olamaz.")
        return cleaned


class LoginRequest(BaseModel):
    email: str = Field(..., max_length=255)
    password: str = Field(..., min_length=1, max_length=128)

    @field_validator("email")
    @classmethod
    def normalize_email(cls, v: str) -> str:
        return v.strip().lower()


class WriteRequest(BaseModel):
    repo_url: str = Field(..., min_length=5, max_length=500)
    user_id: Optional[str] = None
    focus_area: Optional[str] = Field(default=None, max_length=500)
    audience: Optional[str] = Field(default=DEFAULT_AUDIENCE, max_length=255)
    tone: Optional[str] = Field(default=DEFAULT_TONE, max_length=255)
    workers: int = Field(default=3, ge=1, le=10)


def _outline_quality_ok(outline: dict) -> bool:
    secs = outline.get("sections", [])
    return len(secs) >= 3 and all(s.get("source_files") for s in secs)


# --- Endpoint'ler ---
@app.get("/")
def read_root():
    return {"message": "GitHub to Medium API çalışıyor 🚀", "version": "1.0.0"}


@app.get("/health")
def check_health(db: Session = Depends(get_db)):
    """Veritabanı bağlantısını test eden health check endpoint'i"""
    try:
        db.execute(text("SELECT 1"))
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        logger.error(f"Health check başarısız: {e}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "unhealthy", "database": "disconnected", "error": str(e)},
        )


@app.post("/signup", status_code=status.HTTP_201_CREATED)
def signup(user_data: SignupRequest, db: Session = Depends(get_db)):
    """Yeni kullanıcı kaydı oluşturur ve JWT erişim token'ı döner"""
    existing_user = db.query(models.User).filter(models.User.email == user_data.email).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Bu e-posta adresi zaten kullanımda.",
        )

    try:
        new_user = models.User(
            username=user_data.username,
            email=user_data.email,
            password=hash_password(user_data.password),
        )
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
    except Exception as e:
        db.rollback()
        logger.error(f"Kayıt sırasında veritabanı hatası: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Kullanıcı kaydedilirken bir hata oluştu.",
        )

    token = create_access_token(new_user.user_id, new_user.email)
    logger.info(f"Yeni kullanıcı kaydoldu: {new_user.email} ({new_user.user_id})")

    return {
        "message": "Kayıt başarılı",
        "access_token": token,
        "token_type": "bearer",
        "user": {
            "user_id": new_user.user_id,
            "username": new_user.username,
            "email": new_user.email,
            "created_at": new_user.created_at,
        },
    }


@app.post("/login")
def login(user_data: LoginRequest, db: Session = Depends(get_db)):
    """Kullanıcı girişi yapar, eski şifre formatlarını otomatik günceller ve JWT döner"""
    user = db.query(models.User).filter(models.User.email == user_data.email).first()
    if not user or not verify_password(user_data.password, user.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="E-posta veya şifre hatalı.",
        )

    # Eski formatta (SHA-256 veya düz metin) saklanan şifreyi otomatik olarak PBKDF2'ye yükselt
    if not user.password.startswith("pbkdf2_sha256$"):
        try:
            user.password = hash_password(user_data.password)
            db.commit()
        except Exception:
            db.rollback()

    token = create_access_token(user.user_id, user.email)
    logger.info(f"Kullanıcı girişi başarılı: {user.email}")

    return {
        "message": "Giriş başarılı",
        "access_token": token,
        "token_type": "bearer",
        "user": {
            "user_id": user.user_id,
            "username": user.username,
            "email": user.email,
        },
    }


@app.post("/write")
def write_blog(
    req: WriteRequest,
    db: Session = Depends(get_db),
    current_user: Optional[models.User] = Depends(get_optional_current_user),
):
    """GitHub repo URL'sini alır, analiz/planlama/yazım aşamalarını çalıştırır ve veritabanına kaydeder"""
    try:
        owner, repo = parse_repo_url(req.repo_url)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Geçersiz GitHub URL'si: {e}",
        )

    effective_user_id = current_user.user_id if current_user else req.user_id
    if effective_user_id and not current_user:
        user_exists = db.query(models.User).filter(models.User.user_id == effective_user_id).first()
        if not user_exists:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Belirtilen user_id ile kullanıcı bulunamadı.",
            )

    audience = (req.audience or DEFAULT_AUDIENCE).strip()
    tone = (req.tone or DEFAULT_TONE).strip()
    focus_area = req.focus_area.strip() if req.focus_area and req.focus_area.strip() else None

    try:
        repo_record = models.Repository(
            user_id=effective_user_id,
            owner=owner,
            repo_name=repo,
            repo_url=req.repo_url.strip(),
            status="processing",
            focus_area=focus_area,
            audience=audience,
            tone=tone,
        )
        db.add(repo_record)
        db.commit()
        db.refresh(repo_record)
    except Exception as e:
        db.rollback()
        logger.error(f"Repository kaydı oluşturulamadı: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Veritabanı kaydı başlatılamadı.",
        )

    logger.info(f"[Repo #{repo_record.id}] '{owner}/{repo}' için blog üretimi başlatıldı.")

    try:
        # 1. AŞAMA: GitHub'dan repoyu çek
        fetcher = GitHubFetcher()
        repo_data = fetcher.fetch_repository(owner, repo)
        files = repo_data.get("files", [])
        directory_structure = repo_data.get("directory_structure") or [f["path"] for f in files]

        # 2. AŞAMA: Dosyaları analiz et
        dependency_manifests, markdown_docs, analyzable, _ = partition_repo_files(files)
        if not analyzable:
            raise RuntimeError("Analiz edilecek kod dosyası bulunamadı.")

        analyzer = AnalyzerAgent()
        analyzed: List[dict] = []
        failed: List[dict] = []

        def analyze_task(f_info: dict):
            try:
                result_schema = analyzer.run(
                    repo_name=repo_data.get("repository", repo),
                    file_path=f_info["path"],
                    file_content=f_info["content"],
                )
                return f_info, result_schema.model_dump(), None
            except Exception as e:
                return f_info, None, str(e)

        with ThreadPoolExecutor(max_workers=max(1, req.workers)) as executor:
            futures = [executor.submit(analyze_task, f) for f in analyzable]
            for fut in as_completed(futures):
                f_info, res_dict, err = fut.result()
                if err:
                    failed.append({"path": f_info["path"], "error": err[:500]})
                else:
                    analyzed.append({
                        "path": f_info["path"],
                        "size": f_info.get("size", 0),
                        **res_dict,
                    })

        if not analyzed:
            raise RuntimeError("Hiçbir dosya başarıyla analiz edilemedi.")

        analyzed.sort(key=lambda x: x.get("blog_worthiness_score", 0), reverse=True)
        highlights: List[str] = []
        for item in analyzed:
            for h in item.get("engineering_highlights", []):
                if h not in highlights:
                    highlights.append(h)

        analysis_data = {
            "repository": repo_data.get("repository", f"{owner}/{repo}"),
            "description": repo_data.get("description"),
            "stargazers_count": repo_data.get("stargazers_count", 0),
            "languages": repo_data.get("languages", []),
            "directory_structure": directory_structure,
            "dependency_manifests": dependency_manifests,
            "markdown_docs": markdown_docs,
            "project_wide_highlights": highlights,
            "top_architectural_files": [
                {
                    "path": i["path"],
                    "architectural_domain": i.get("architectural_domain", "Genel Mimari"),
                    "score": i["blog_worthiness_score"],
                    "responsibility": i["responsibility"],
                    "engineering_highlights": i["engineering_highlights"],
                }
                for i in analyzed
                if i.get("blog_worthiness_score", 0) >= 7
            ],
            "analyzed_files_count": len(analyzed),
            "failed_files": failed,
            "file_analyses": analyzed,
        }

        # 3. AŞAMA: Blog iskeletini (Outline) oluştur
        planner = PlannerAgent()
        outline = None
        for _ in range(MAX_OUTLINE_ATTEMPTS):
            outline_model = planner.run(
                analysis=analysis_data,
                audience=audience,
                tone=tone,
                focus_area=focus_area,
            )
            outline = outline_model.model_dump()
            if _outline_quality_ok(outline):
                break
        else:
            kept = [s for s in outline.get("sections", []) if s.get("source_files")]
            if len(kept) < 2:
                raise RuntimeError("Geçerli bir blog iskeleti (outline) üretilemedi.")
            outline["sections"] = kept

        selected_focus = outline.get("selected_focus_area") or focus_area
        outline_data = {
            "repository": analysis_data["repository"],
            "audience": audience,
            "tone": tone,
            "focus_area": selected_focus,
            "outline": outline,
        }

        # 4. AŞAMA: Blog yazısını (Markdown) üret
        code_contents = {f["path"]: f.get("content", "") for f in files}
        writer = WriterAgent(
            outline_data=outline_data,
            analysis=analysis_data,
            code_contents=code_contents,
        )
        write_result = writer.run()
        markdown = write_result["markdown"]
        warnings = list(write_result.get("warnings", []))

        issues = final_checks(markdown)
        if SNIPPET_TOKEN in markdown:
            markdown = markdown.replace(SNIPPET_TOKEN, "")
            issues = [i for i in issues if "yer tutucu" not in i.lower()] + [
                "Kullanılmamış yer tutucu otomatik temizlendi."
            ]
        all_warnings = warnings + issues

        # 5. AŞAMA: Sonuçları veritabanına kaydet
        repo_record.status = "completed"
        repo_record.focus_area = selected_focus

        blog_output = models.BlogOutput(
            repo_id=repo_record.id,
            analysis_data=analysis_data,
            outline_data=outline_data,
            markdown_content=markdown,
            warnings=all_warnings,
        )
        db.add(blog_output)
        db.commit()
        db.refresh(blog_output)

        logger.info(f"[Repo #{repo_record.id}] Blog üretimi başarıyla tamamlandı (Blog #{blog_output.id}).")

        return {
            "message": "Blog yazısı başarıyla üretildi",
            "repo_id": repo_record.id,
            "blog_id": blog_output.id,
            "status": repo_record.status,
            "owner": owner,
            "repo_name": repo,
            "focus_area": selected_focus,
            "title": outline.get("chosen_title"),
            "subtitle": outline.get("subtitle"),
            "tags": outline.get("tags", []),
            "estimated_read_minutes": outline.get("estimated_read_minutes"),
            "warnings": all_warnings,
            "analysis": analysis_data,
            "outline": outline,
            "markdown_content": markdown,
        }

    except Exception as e:
        db.rollback()
        try:
            repo_record.status = "failed"
            db.add(repo_record)
            db.commit()
        except Exception:
            db.rollback()
        logger.exception(f"[Repo #{repo_record.id}] Blog üretim hatası: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Blog üretim sürecinde hata oluştu: {str(e)}",
        )


@app.get("/repositories")
def list_repositories(
    user_id: Optional[str] = Query(default=None),
    limit: int = Query(default=20, ge=1, le=100),
    db: Session = Depends(get_db),
    current_user: Optional[models.User] = Depends(get_optional_current_user),
):
    """Kullanıcıya ait veya genel üretilen repoların listesini döner"""
    effective_user_id = current_user.user_id if current_user else user_id
    query = db.query(models.Repository)
    if effective_user_id:
        query = query.filter(models.Repository.user_id == effective_user_id)

    repos = query.order_by(models.Repository.created_at.desc()).limit(limit).all()
    return {
        "count": len(repos),
        "items": [
            {
                "repo_id": r.id,
                "user_id": r.user_id,
                "owner": r.owner,
                "repo_name": r.repo_name,
                "repo_url": r.repo_url,
                "status": r.status,
                "focus_area": r.focus_area,
                "created_at": r.created_at,
                "has_blog": r.blog_output is not None,
            }
            for r in repos
        ],
    }


@app.get("/repositories/{repo_id}")
def get_repository_blog(repo_id: int, db: Session = Depends(get_db)):
    """Belirli bir repo kaydını ve üretilmiş blog içeriğini döner"""
    repo_record = db.query(models.Repository).filter(models.Repository.id == repo_id).first()
    if not repo_record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Repo kaydı bulunamadı.",
        )

    blog = repo_record.blog_output
    outline = (blog.outline_data or {}).get("outline", {}) if blog else {}

    return {
        "repo_id": repo_record.id,
        "blog_id": blog.id if blog else None,
        "status": repo_record.status,
        "owner": repo_record.owner,
        "repo_name": repo_record.repo_name,
        "repo_url": repo_record.repo_url,
        "focus_area": repo_record.focus_area,
        "created_at": repo_record.created_at,
        "title": outline.get("chosen_title"),
        "subtitle": outline.get("subtitle"),
        "tags": outline.get("tags", []),
        "estimated_read_minutes": outline.get("estimated_read_minutes"),
        "warnings": blog.warnings if blog else [],
        "analysis": blog.analysis_data if blog else None,
        "outline": outline,
        "markdown_content": blog.markdown_content if blog else None,
    }


if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "3000"))
    uvicorn.run("api:app", host=host, port=port, reload=os.getenv("ENV", "development") == "development")
