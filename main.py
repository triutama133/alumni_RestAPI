from fastapi import FastAPI, HTTPException, Security, status
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from typing import Optional, List
import json
import logging
import os
import re
import httpx
import asyncpg
import traceback
from dotenv import load_dotenv

# --- KONFIGURASI DAN SETUP ---

# Muat variabel lingkungan dari file .env
load_dotenv()
SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
INTERNAL_API_KEY = os.getenv("INTERNAL_API_KEY")

# Inisialisasi aplikasi FastAPI
app = FastAPI()

# Definisi skema header API Key untuk keamanan
api_key_header = APIKeyHeader(name="X-API-KEY", auto_error=False)

# Dependency untuk memeriksa API Key pada setiap request
async def get_api_key(api_key: str = Security(api_key_header)):
    if not api_key or api_key != INTERNAL_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or Missing API Key"
        )
    return api_key

@app.get("/")
def root():
    return {"message": "HubTalent AI Engine is running!"}

# --- MODEL DATA (PYDANTIC) ---

class RekomendasiInput(BaseModel):
    user_id: Optional[int] = None
    nama_lengkap: Optional[str] = None
    cohort_id: Optional[int] = None
    language: str = "id"
    prompt_tambahan: Optional[str] = None

class WawasanInput(BaseModel):
    user_id: Optional[int] = None
    nama_lengkap: Optional[str] = None
    cohort_id: Optional[int] = None
    language: str = "id"

class ProyekInput(BaseModel):
    ide_proyek: str
    cohort_id: Optional[int] = None
    language: str = "id"

class LearningPathInput(BaseModel):
    user_id: int
    target_role: str
    language: str = "id"

class InterviewTurn(BaseModel):
    role: str  # "ai" atau "user"
    content: str

class InterviewSimulationInput(BaseModel):
    user_id: int
    target_role: str
    conversation_history: List[InterviewTurn] = []
    language: str = "id"

def sanitize_ai_input(text: str, max_length: int = 500) -> str:
    """Membersihkan input pengguna sebelum disertakan ke prompt LLM."""
    if not text:
        return ""
    text = re.sub(r"[\x00-\x1f\x7f]", "", text)
    text = text.replace("SYSTEM:", "").replace("IGNORE PREVIOUS", "").replace("---", "")
    return text[:max_length].strip()

def normalize_aktivitas_values(value):
    if value is None:
        return []
    if isinstance(value, list):
        raw_values = value
    else:
        raw_values = re.split(r"[,;]", str(value))
    normalized = []
    for item in raw_values:
        cleaned = str(item).strip()
        if cleaned:
            normalized.append(cleaned)
    return normalized

def tokenize_text(text: str):
    if not text:
        return set()
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return {t for t in tokens if len(t) > 2}

def compute_weighted_match_score(source_tokens, target_text: str, priority_keywords=None):
    if not source_tokens or not target_text:
        return 0.0
    target_tokens = tokenize_text(target_text)
    if not target_tokens:
        return 0.0

    priority_set = {str(x).lower() for x in (priority_keywords or []) if x}
    score = 0.0
    for token in source_tokens:
        if token not in target_tokens:
            continue
        if token in priority_set:
            score += 2.5
        elif len(token) >= 8:
            score += 1.6
        elif len(token) >= 5:
            score += 1.3
        else:
            score += 1.0
    return score

# Common Indonesian/English connector words that pass the length filter in
# tokenize_text but aren't meaningful "skills" — excluded so matched-skill badges
# shown to users stay credible instead of surfacing words like "untuk" or "dengan".
STOPWORDS = {
    "untuk", "dengan", "yang", "dari", "pada", "adalah", "akan", "atau", "juga",
    "dapat", "serta", "dalam", "oleh", "para", "tersebut", "secara", "sudah",
    "harus", "bisa", "saja", "seperti", "maka", "lain", "lainnya", "antara",
    "about", "with", "from", "that", "this", "have", "will", "your", "their",
}

def extract_matched_terms(source_tokens, target_text: str, limit: int = 8):
    """
    Returns the actual overlapping keywords between a source (a project's description,
    or a profile's skills) and a candidate's profile text — the same token overlap
    compute_weighted_match_score already uses to produce a number, just surfaced as
    concrete words instead. This is what powers "matched skills" in the talent preview,
    so it's a factual overlap, not something re-guessed by the LLM.
    """
    if not source_tokens or not target_text:
        return []
    target_tokens = tokenize_text(target_text)
    matched = sorted(
        {t for t in source_tokens if t in target_tokens and len(t) >= 4 and t not in STOPWORDS},
        key=len,
        reverse=True,
    )
    return matched[:limit]

def build_candidate_cards(raw_candidates, *, name_key, skills_key, id_key="id", aktivitas_key="aktivitas", score_key="match_score", matched_key="matched_terms"):
    """
    Turns a raw scored-candidate list (already computed for the LLM prompt) into the
    structured shape the frontend needs for clickable, inviteable talent cards: an id
    to link/invite by, and a 'tier' (kuat/sedang/lemah) computed relative to the top
    score in this batch, for the orbit-style strength visualization. Previously these
    candidates only ever reached the client as names embedded in prose.
    """
    if not raw_candidates:
        return []

    max_score = max((c.get(score_key) or 0) for c in raw_candidates) or 1
    cards = []
    for c in raw_candidates:
        score = c.get(score_key) or 0
        strength_pct = round((score / max_score) * 100, 1)
        if strength_pct >= 70:
            tier = "kuat"
        elif strength_pct >= 40:
            tier = "sedang"
        else:
            tier = "lemah"
        cards.append({
            "id": c.get(id_key),
            "nama_lengkap": c.get(name_key),
            "aktivitas": c.get(aktivitas_key),
            "skills": c.get(skills_key) or "",
            "matched_skills": c.get(matched_key) or [],
            "match_score": score,
            "match_strength": strength_pct,
            "tier": tier,
        })
    return cards

async def call_gemini(prompt: str, temperature: float = 0.7) -> str:
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY belum dikonfigurasi.")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    body = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": temperature, "maxOutputTokens": 2500},
    }

    async with httpx.AsyncClient(timeout=90.0) as client:
        res = await client.post(url, headers={"Content-Type": "application/json"}, json=body)
        res.raise_for_status()
        return res.json()["candidates"][0]["content"]["parts"][0]["text"].strip()


async def call_llm_service(prompt: str, temperature: float = 0.7) -> str:
    """Wrapper tunggal untuk memanggil Google Gemini."""
    try:
        return await call_gemini(prompt, temperature)
    except Exception as err:
        raise HTTPException(
            status_code=500,
            detail=f"Panggilan ke Gemini gagal: {err}",
        )


def parse_learning_path_json(raw_text: str) -> dict:
    """Ekstrak objek JSON dari respons model agar frontend selalu menerima struktur stabil."""
    text = (raw_text or "").strip()
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError("Respons AI tidak berisi JSON valid.")

    payload = json.loads(match.group(0))
    gap_analysis = payload.get("gap_analysis")
    learning_path = payload.get("learning_path")
    checklist = payload.get("checklist")

    if not isinstance(gap_analysis, list) or not isinstance(learning_path, list) or not isinstance(checklist, list):
        raise ValueError("Struktur JSON AI tidak sesuai format yang dibutuhkan.")

    return {
        "gap_analysis": gap_analysis,
        "learning_path": learning_path,
        "checklist": checklist,
    }


def parse_interview_feedback_json(raw_text: str) -> dict:
    """Ekstrak objek JSON feedback wawancara dari respons model."""
    text = (raw_text or "").strip()
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError("Respons AI tidak berisi JSON valid.")

    payload = json.loads(match.group(0))
    strengths = payload.get("strengths")
    improvements = payload.get("improvements")
    summary = payload.get("summary")
    overall_score = payload.get("overall_score")

    if (
        not isinstance(strengths, list)
        or not isinstance(improvements, list)
        or not isinstance(summary, str)
        or not isinstance(overall_score, (int, float))
    ):
        raise ValueError("Struktur JSON feedback AI tidak sesuai format yang dibutuhkan.")

    return {
        "overall_score": max(1, min(10, round(overall_score))),
        "summary": summary,
        "strengths": strengths,
        "improvements": improvements,
        "closing_tip": payload.get("closing_tip") or "",
    }

# --- FUNGSI LOGIKA INTI ---

async def cari_top_alumni_kolaborasi(current_alumni_id: int, current_alumni_full_profile_text: str, current_aktivitas_values=None, cohort_id: int = None):
    """
    OPTIMAL: Mencari 5 alumni relevan hanya dengan 1 query ke alumni_db.
    Mendukung penyaringan berdasarkan cohort_id jika diisi.
    """
    conn = await asyncpg.connect(SUPABASE_DB_URL, statement_cache_size=0)
    try:
        # Mengambil semua data yang dibutuhkan dalam satu query
        if cohort_id:
            all_alumni_general = await conn.fetch("""
                SELECT ad.id, ad.nama_lengkap, ad.aktivitas, ad.skill_gabungan, ad.gabungan_data 
                FROM alumni_db ad
                JOIN cohort_members cm ON cm.user_id = ad.id
                WHERE ad.id != $1 AND cm.cohort_id = $2
            """, current_alumni_id, cohort_id)
        else:
            all_alumni_general = await conn.fetch(
                "SELECT id, nama_lengkap, aktivitas, skill_gabungan, gabungan_data FROM alumni_db WHERE id != $1",
                current_alumni_id
            )

        all_relevant_alumni = []
        current_alumni_tokens = tokenize_text(current_alumni_full_profile_text)
        current_aktivitas_values = current_aktivitas_values or []

        for alumni_gen in all_alumni_general:
            # Profil lengkap alumni lain kini hanya dari skill_gabungan dan gabungan_data
            other_alumni_skills = alumni_gen["skill_gabungan"] or ""
            other_alumni_details = alumni_gen["gabungan_data"] or ""
            other_alumni_full_profile_text = f"{other_alumni_skills} {other_alumni_details}".lower().strip()

            # Tidak ada lagi query tambahan ke tabel sekunder

            match_score = compute_weighted_match_score(
                current_alumni_tokens,
                other_alumni_full_profile_text,
                priority_keywords=current_aktivitas_values,
            )

            other_aktivitas_values = normalize_aktivitas_values(alumni_gen["aktivitas"])
            if set(a.lower() for a in current_aktivitas_values).intersection(a.lower() for a in other_aktivitas_values):
                match_score += 1.5

            if match_score > 0:
                matched_terms = extract_matched_terms(current_alumni_tokens, other_alumni_full_profile_text)
                all_relevant_alumni.append({
                    "id": alumni_gen["id"],
                    "nama_alumni_kolaborasi": alumni_gen["nama_lengkap"],
                    "aktivitas": alumni_gen["aktivitas"],
                    "relevance_skills": other_alumni_skills,
                    "relevance_detail_summary": other_alumni_full_profile_text,
                    "match_score": round(match_score, 2),
                    "matched_terms": matched_terms,
                })
        
        all_relevant_alumni.sort(key=lambda x: x['match_score'], reverse=True)
        return all_relevant_alumni[:5]
    finally:
        await conn.close()

async def ambil_profil_alumni(user_id: int = None, nama_lengkap: str = None, cohort_id: int = None):
    """
    OPTIMAL: Mengambil profil lengkap alumni hanya dengan 1 query ke alumni_db.
    Juga mengambil data peluang dari tabel lain untuk konteks tambahan.
    """
    conn = await asyncpg.connect(SUPABASE_DB_URL, statement_cache_size=0)
    try:
        # Mengambil data utama alumni
        if user_id is not None:
            row = await conn.fetchrow("""
                SELECT id, nama_lengkap, nama_panggilan, aktivitas, skill_gabungan, gabungan_data
                FROM alumni_db WHERE id = $1
            """, user_id)
        elif nama_lengkap:
            row = await conn.fetchrow("""
                SELECT id, nama_lengkap, nama_panggilan, aktivitas, skill_gabungan, gabungan_data
                FROM alumni_db WHERE LOWER(TRIM(nama_lengkap)) = LOWER(TRIM($1))
            """, nama_lengkap)
        else:
            raise HTTPException(status_code=400, detail="user_id atau nama_lengkap wajib diisi")

        if not row:
            raise HTTPException(status_code=404, detail="Alumni tidak ditemukan")

        alumni_id = row["id"]
        alumni_skills = row["skill_gabungan"] or ""
        alumni_details_text = row["gabungan_data"] or ""
        alumni_aktivitas_values = normalize_aktivitas_values(row["aktivitas"])
        alumni_aktivitas_primary = alumni_aktivitas_values[0] if alumni_aktivitas_values else "tidak diketahui"

        # Profil lengkap untuk pencocokan kini hanya dari gabungan_data dan skill_gabungan
        current_alumni_full_profile_text = f"{alumni_skills} {alumni_details_text}".strip()

        # Mendukung split koma (,) dan titik koma (;) dari skill_gabungan
        skills_for_cocok = []
        for s in alumni_skills.replace(',', ';').split(';'):
            if s.strip():
                skills_for_cocok.append(s.strip())

        def cocok(row_val):
            if not row_val: return False
            return any(skill.lower() in str(row_val).lower() for skill in skills_for_cocok)

        bisnis_rows = await conn.fetch("SELECT nama_usaha, produk_layanan_utama, skala_usaha, target_pasar, kendala_bisnis, keahlian_wirausahaan FROM alumni_bisnis")
        pekerja_rows = await conn.fetch("SELECT nama_instansi, posisi, keahlian_pekerja, pengalaman_proyek FROM alumni_pekerja")
        sosial_rows = await conn.fetch("SELECT nama_organisasi, isu_fokus, keahlian_sosial, pengalaman_proyek_sosial FROM alumni_sosial")
        kreatif_rows = await conn.fetch("SELECT keahlian_kreatif, platform_digital_utama, jenis_konten, total_jangkauan FROM alumni_kreatif")
        irt_rows = await conn.fetch("SELECT keahlian_irt, kegiatan_organisasi_irt FROM alumni_rumah_tangga")
        mahasiswa_rows = await conn.fetch("SELECT keahlian_mahasiswa, kegiatan_organisasi_mahasiswa, pengalaman_magang FROM alumni_mahasiswa")
        informal_rows = await conn.fetch("SELECT keahlian_informal FROM alumni_informal")
        agri_rows = await conn.fetch("SELECT keahlian_agri, komoditas_utama, skala_usaha_agri, kendala_dihadapi_agri FROM alumni_agri")
        pendidik_rows = await conn.fetch("SELECT keahlian_pendidik, jenjang_pendidikan, mata_pelajaran, inovasi_pembelajaran FROM alumni_pendidik")
        
        peluang_bisnis = []
        for r in bisnis_rows:
            desc = f"Produk: {r['produk_layanan_utama'] or ''}. Kendala: {r['kendala_bisnis'] or ''}"
            if cocok(r["produk_layanan_utama"]) or cocok(r["kendala_bisnis"]) or cocok(r["keahlian_wirausahaan"]):
                peluang_bisnis.append({
                    "nama_usaha": r["nama_usaha"],
                    "dukungan": desc,
                    "kolaborasi": r["keahlian_wirausahaan"],
                    "butuh_sdm": r["skala_usaha"]
                })

        peluang_pekerja = []
        for r in pekerja_rows:
            if cocok(r["keahlian_pekerja"]) or cocok(r["pengalaman_proyek"]) or cocok(r["posisi"]):
                peluang_pekerja.append({
                    "nama_instansi": r["nama_instansi"],
                    "skill": r["keahlian_pekerja"],
                    "deskripsi_skill": r["pengalaman_proyek"],
                    "sertifikasi": r["posisi"],
                    "dukungan": r["pengalaman_proyek"]
                })

        peluang_irt = []
        for r in irt_rows:
            if cocok(r["keahlian_irt"]) or cocok(r["kegiatan_organisasi_irt"]):
                peluang_irt.append({
                    "bidang_minat": r["keahlian_irt"],
                    "spesifik_bidang": r["keahlian_irt"],
                    "pengalaman_kelas": "",
                    "bentuk_kolaborasi": r["kegiatan_organisasi_irt"],
                    "perlu_grup": ""
                })

        peluang_sosial = []
        for r in sosial_rows:
            if cocok(r["keahlian_sosial"]) or cocok(r["isu_fokus"]) or cocok(r["pengalaman_proyek_sosial"]):
                peluang_sosial.append({
                    "nama_organisasi": r["nama_organisasi"],
                    "isu_fokus": r["isu_fokus"],
                    "peluang": r["pengalaman_proyek_sosial"] or r["keahlian_sosial"],
                })

        peluang_kreatif = []
        for r in kreatif_rows:
            if cocok(r["keahlian_kreatif"]) or cocok(r["platform_digital_utama"]) or cocok(r["jenis_konten"]):
                peluang_kreatif.append({
                    "keahlian": r["keahlian_kreatif"],
                    "platform": r["platform_digital_utama"],
                    "jenis_konten": r["jenis_konten"],
                    "jangkauan": r["total_jangkauan"],
                })

        peluang_mahasiswa = []
        for r in mahasiswa_rows:
            if cocok(r["keahlian_mahasiswa"]) or cocok(r["kegiatan_organisasi_mahasiswa"]) or cocok(r["pengalaman_magang"]):
                peluang_mahasiswa.append({
                    "keahlian": r["keahlian_mahasiswa"],
                    "organisasi": r["kegiatan_organisasi_mahasiswa"],
                    "magang": r["pengalaman_magang"],
                })

        peluang_informal = []
        for r in informal_rows:
            if cocok(r["keahlian_informal"]):
                peluang_informal.append({
                    "keahlian": r["keahlian_informal"],
                })

        peluang_agri = []
        for r in agri_rows:
            if cocok(r["keahlian_agri"]) or cocok(r["komoditas_utama"]) or cocok(r["kendala_dihadapi_agri"]):
                peluang_agri.append({
                    "keahlian": r["keahlian_agri"],
                    "komoditas": r["komoditas_utama"],
                    "skala": r["skala_usaha_agri"],
                    "kendala": r["kendala_dihadapi_agri"],
                })

        peluang_pendidik = []
        for r in pendidik_rows:
            if cocok(r["keahlian_pendidik"]) or cocok(r["mata_pelajaran"]) or cocok(r["inovasi_pembelajaran"]):
                peluang_pendidik.append({
                    "keahlian": r["keahlian_pendidik"],
                    "jenjang": r["jenjang_pendidikan"],
                    "mapel": r["mata_pelajaran"],
                    "inovasi": r["inovasi_pembelajaran"],
                })

        education_histories = await conn.fetch(
            """
            SELECT level, institution_name, major_program, start_year, end_year, is_current
            FROM alumni_education_histories
            WHERE alumni_id = $1
            ORDER BY is_current DESC, COALESCE(end_year, 0) DESC, COALESCE(start_year, 0) DESC
            """,
            alumni_id,
        )
        education_summaries = [
            f"{e['level']} {e['major_program']} - {e['institution_name']}"
            for e in education_histories
        ]

        # Hitung statistik jejaring/kelompok
        if cohort_id:
            total_cohort = await conn.fetchval("""
                SELECT COUNT(*) FROM cohort_members WHERE cohort_id = $1
            """, cohort_id)
            activity_rows = await conn.fetch("""
                SELECT ad.aktivitas
                FROM alumni_db ad
                JOIN cohort_members cm ON cm.user_id = ad.id
                WHERE cm.cohort_id = $1
            """, cohort_id)
        else:
            total_cohort = await conn.fetchval("SELECT COUNT(*) FROM alumni_db")
            activity_rows = await conn.fetch("SELECT aktivitas FROM alumni_db")

        activity_dist = {}
        for row_act in activity_rows:
            aktivitas_values = normalize_aktivitas_values(row_act["aktivitas"])
            if not aktivitas_values:
                aktivitas_values = ["tidak diketahui"]
            for act_name in aktivitas_values:
                activity_dist[act_name] = activity_dist.get(act_name, 0) + 1

        same_activity_count = max([activity_dist.get(act, 0) for act in alumni_aktivitas_values], default=0)
        same_activity_percentage = round((same_activity_count / total_cohort) * 100, 1) if total_cohort > 0 else 0

        # Mencari top 5 alumni untuk kolaborasi
        top_alumni_kolaborasi = await cari_top_alumni_kolaborasi(
            alumni_id,
            current_alumni_full_profile_text,
            current_aktivitas_values=alumni_aktivitas_values,
            cohort_id=cohort_id,
        )

        return {
            "nama": row["nama_lengkap"],
            "nama_panggilan": row["nama_panggilan"],
            "aktivitas": row["aktivitas"],
            "aktivitas_primary": alumni_aktivitas_primary,
            "aktivitas_list": alumni_aktivitas_values,
            "skills": alumni_skills,
            "gabungan_data": alumni_details_text,
            "peluang_bisnis": peluang_bisnis,
            "peluang_pekerja": peluang_pekerja,
            "peluang_sosial": peluang_sosial,
            "peluang_kreatif": peluang_kreatif,
            "peluang_irt": peluang_irt,
            "peluang_mahasiswa": peluang_mahasiswa,
            "peluang_informal": peluang_informal,
            "peluang_agri": peluang_agri,
            "peluang_pendidik": peluang_pendidik,
            "education_summaries": education_summaries,
            "top_alumni_kolaborasi": top_alumni_kolaborasi,
            "total_cohort": total_cohort,
            "activity_dist": activity_dist,
            "same_activity_count": same_activity_count,
            "same_activity_percentage": same_activity_percentage
        }
    finally:
        await conn.close()

def build_prompt(data, language, source="profile"):
    """
    Membangun prompt untuk LLM, kini menggunakan gabungan_data secara langsung.
    Mendukung format ringkas untuk beranda (source='home') atau detail (source='profile').
    """
    aktivitas_list_display = ", ".join(data.get("aktivitas_list", [])) or str(data.get("aktivitas") or "tidak diketahui")

    # Free-text context from the user (e.g. "Pencarian Cerdas AI" on the Hub Proyek page
    # lets someone describe their availability/needs in natural language) — previously
    # accepted by the Next.js route but never actually reached this prompt.
    prompt_tambahan_content = ""
    if data.get("prompt_tambahan"):
        if language.lower() == "id":
            prompt_tambahan_content = f"\n--- KONTEKS TAMBAHAN DARI PENGGUNA ---\n{data['prompt_tambahan']}\n---------------------------------------\n"
        else:
            prompt_tambahan_content = f"\n--- ADDITIONAL CONTEXT FROM USER ---\n{data['prompt_tambahan']}\n-------------------------------------\n"

    top_alumni_kolaborasi_content = ""
    if data['top_alumni_kolaborasi']:
        temp_alumni_str_list = []
        if language.lower() == "id":
            for alumni in data['top_alumni_kolaborasi']:
                summary = alumni['relevance_detail_summary'] if alumni['relevance_detail_summary'] else alumni['relevance_skills']
                temp_alumni_str_list.append(f"- Nama: {alumni['nama_alumni_kolaborasi']} (Aktivitas: {alumni['aktivitas']}). Keahlian/Detail Relevan: {summary}")
            top_alumni_kolaborasi_content = "Konteks Tambahan - Profil alumni lain yang paling cocok untuk kolaborasi:\n" + "\n".join(temp_alumni_str_list)
        else: # en
            for alumni in data['top_alumni_kolaborasi']:
                summary = alumni['relevance_detail_summary'] if alumni['relevance_detail_summary'] else alumni['relevance_skills']
                temp_alumni_str_list.append(f"- Name: {alumni['nama_alumni_kolaborasi']} (Activity: {alumni['aktivitas']}). Skills/Relevant Details: {summary}")
            top_alumni_kolaborasi_content = "Additional Context - Other most suitable alumni profiles for collaboration:\n" + "\n".join(temp_alumni_str_list)

    peluang_bisnis_str = "\n".join([f"- Bisnis '{pb.get('nama_usaha', 'N/A')}' butuh: {pb.get('dukungan') or pb.get('kolaborasi') or pb.get('butuh_sdm')}" for pb in data['peluang_bisnis']])
    peluang_pekerja_str = "\n".join([f"- Profesional di '{p.get('nama_instansi', 'N/A')}' butuh dukungan di bidang: {p.get('dukungan') or p.get('skill')}" for p in data['peluang_pekerja']])
    peluang_irt_str = "\n".join([f"- Ibu Rumah Tangga dengan minat '{i.get('bidang_minat', 'N/A')}' butuh/ingin: {i.get('bentuk_kolaborasi') or i.get('perlu_grup')}" for i in data['peluang_irt']])
    peluang_sosial_str = "\n".join([f"- Komunitas '{s.get('nama_organisasi', 'N/A')}' fokus '{s.get('isu_fokus', 'N/A')}' membuka peluang: {s.get('peluang') or 'N/A'}" for s in data.get('peluang_sosial', [])])
    peluang_kreatif_str = "\n".join([f"- Kreator dengan keahlian '{k.get('keahlian', 'N/A')}' di platform '{k.get('platform', 'N/A')}' ({k.get('jenis_konten', 'N/A')})" for k in data.get('peluang_kreatif', [])])
    peluang_mahasiswa_str = "\n".join([f"- Mahasiswa/FG dengan keahlian '{m.get('keahlian', 'N/A')}' dan pengalaman magang: {m.get('magang') or 'N/A'}" for m in data.get('peluang_mahasiswa', [])])
    peluang_informal_str = "\n".join([f"- Pekerja informal dengan keahlian: {inf.get('keahlian') or 'N/A'}" for inf in data.get('peluang_informal', [])])
    peluang_agri_str = "\n".join([f"- Pelaku agri komoditas '{a.get('komoditas', 'N/A')}' dengan keahlian '{a.get('keahlian', 'N/A')}', kendala: {a.get('kendala') or 'N/A'}" for a in data.get('peluang_agri', [])])
    peluang_pendidik_str = "\n".join([f"- Pendidik bidang '{p.get('mapel', 'N/A')}' jenjang '{p.get('jenjang', 'N/A')}' dengan inovasi: {p.get('inovasi') or 'N/A'}" for p in data.get('peluang_pendidik', [])])
    education_str = "\n".join([f"- {edu}" for edu in data.get('education_summaries', [])])
    
    peluang_str = (
        f"Konteks Peluang dari Database:\n"
        f"Peluang Bisnis:\n{peluang_bisnis_str or 'Tidak ada data.'}\n"
        f"Peluang dari Pekerja:\n{peluang_pekerja_str or 'Tidak ada data.'}\n"
        f"Peluang dari Ibu Rumah Tangga:\n{peluang_irt_str or 'Tidak ada data.'}\n"
        f"Peluang dari Aktivis Sosial:\n{peluang_sosial_str or 'Tidak ada data.'}\n"
        f"Peluang dari Kreator Konten:\n{peluang_kreatif_str or 'Tidak ada data.'}\n"
        f"Peluang dari Mahasiswa/Fresh Graduate:\n{peluang_mahasiswa_str or 'Tidak ada data.'}\n"
        f"Peluang dari Pekerja Informal:\n{peluang_informal_str or 'Tidak ada data.'}\n"
        f"Peluang dari Sektor Agri:\n{peluang_agri_str or 'Tidak ada data.'}\n"
        f"Peluang dari Pendidik:\n{peluang_pendidik_str or 'Tidak ada data.'}\n"
        f"Riwayat Pendidikan Profil:\n{education_str or 'Tidak ada data.'}\n"
    )

    if source == "home":
        # Let's generate a statistics string
        stats_str = ""
        if language.lower() == "id":
            stats_str = (
                f"- Anda berada di kategori '{data['aktivitas']}' bersama {data['same_activity_count']} alumni lainnya "
                f"({data['same_activity_percentage']}% dari total {data['total_cohort']} alumni di jejaring).\n"
            )
            breakdown_parts = []
            for act, count in data['activity_dist'].items():
                breakdown_parts.append(f"{act}: {count}")
            stats_str += f"- Distribusi aktivitas keseluruhan: {', '.join(breakdown_parts)}.\n"
        else:
            stats_str = (
                f"- You are in the '{data['aktivitas']}' category with {data['same_activity_count']} other alumni "
                f"({data['same_activity_percentage']}% of the total {data['total_cohort']} alumni in the network).\n"
            )
            breakdown_parts = []
            for act, count in data['activity_dist'].items():
                breakdown_parts.append(f"{act}: {count}")
            stats_str += f"- Overall activity distribution: {', '.join(breakdown_parts)}.\n"

        prompt_template = {
            "id": (
                "Profil Anda:\n"
                f"Nama: {data['nama']} (Panggilan: {data['nama_panggilan']})\n"
                f"Aktivitas: {aktivitas_list_display}\n"
                f"Keahlian Utama: {data['skills']}\n\n"
                f"Konteks Statistik Komunitas:\n"
                f"{stats_str}\n"
                f"{top_alumni_kolaborasi_content}\n\n"
                f"TUGAS ANDA:\n"
                f"Berikan ringkasan wawasan partner kolaborasi AI yang sangat padat dan menarik untuk Widget Beranda. Anda HARUS menganalisis dan memaparkan posisi unik profil alumni tersebut terhadap statistik keseluruhan data (misalnya: keunikan keahlian mereka dibanding mayoritas kategori di komunitas).\n\n"
                f"Gunakan format markdown berikut secara persis (ganti dengan analisis nyata):\n\n"
                f"📊 **Posisi Jejaring**: [1-2 kalimat padat yang menjelaskan posisi si profil terhadap keseluruhan data komunitas. Contoh: 'Di tengah komunitas yang didominasi oleh 63.4% Pekerja, keunikan keahlian IT Anda menempatkan Anda sebagai jembatan digital utama untuk 19 wirausahawan (16.5%) di jejaring.']\n\n"
                f"🤝 **Partner Kolaborasi Teratas**:\n"
                f"[Tampilkan maksimal 3 rekomendasi partner dari alumni lain yang paling cocok untuk kolaborasi. Gunakan bullet list, format nama partner ditebalkan, contoh: '- **Nama Partner** (Pekerjaan/Bisnis): Justifikasi singkat 1 kalimat kenapa cocok'].\n\n"
                f"💡 **Peluang Kolaborasi**: 1 ide aksi kolaborasi kilat/proyek sampingan bersama partner tersebut (1 kalimat)."
            ),
            "en": (
                "Your Profile:\n"
                f"Name: {data['nama']} (Nickname: {data['nama_panggilan']})\n"
                f"Activity: {aktivitas_list_display}\n"
                f"Skills: {data['skills']}\n\n"
                f"Community Statistics Context:\n"
                f"{stats_str}\n"
                f"{top_alumni_kolaborasi_content}\n\n"
                f"YOUR TASK:\n"
                f"Provide a very concise and engaging AI collaboration partner summary for the Home Feed Widget. You MUST analyze and present the unique positioning of this alumni profile relative to the overall network statistics (e.g. how their skills stand out compared to the majority categories in the community).\n\n"
                f"Use the following markdown format exactly:\n\n"
                f"📊 **Network Position**: [1-2 concise sentences analyzing the profile's position relative to the overall community data. Example: 'In a community dominated by 63.4% Workers, your unique IT expertise positions you as a key digital bridge for 19 entrepreneurs (16.5%) in the network.']\n\n"
                f"🤝 **Top Partners**:\n"
                f"[Provide up to 3 recommended alumni partners. Use bullet list, format partner name in bold, example: '- **Partner Name** (Job/Business): 1 brief sentence justification'].\n\n"
                f"💡 **Quick Opportunity**: 1 quick tactical collaboration idea (1 sentence)."
            )
        }
    elif source == "karir":
        prompt_template = {
            "id": (
                "Profil Alumni Utama:\n"
                f"Nama: {data['nama']} (Panggilan: {data['nama_panggilan']})\n"
                f"Aktivitas: {aktivitas_list_display}\n"
                f"Ringkasan Keahlian: {data['skills']}\n\n"
                f"--- PROFIL DETAIL (dari gabungan_data) ---\n{data['gabungan_data']}\n"
                f"----------------------------------------\n\n"
                f"{peluang_str}\n"
                f"{top_alumni_kolaborasi_content}\n\n"
                f"TUGAS ANDA:\n"
                f"Berdasarkan profil dan data di atas, berikan analisis rekomendasi pengembangan karir komprehensif bagi {data['nama_panggilan']} dalam format berikut:\n"
                f"1. **Analisis Kekuatan & Potensi Karir Saat Ini**: Ringkasan mendalam tentang kekuatan utama {data['nama_panggilan']} dan bagaimana keahlian saat ini memposisikan dirinya di industri/pasar.\n"
                f"2. **Rekomendasi Peningkatan Keahlian & Sertifikasi (Upskilling & Certifications)**: Sebutkan secara spesifik 3-4 keterampilan baru (hard/soft skills) atau sertifikasi profesional yang wajib diambil untuk meningkatkan daya saing atau memperluas bisnisnya.\n"
                f"3. **Rencana Aksi Karir Strategis**: 4-5 langkah konkret, realistis, dan berurutan (jangka pendek hingga menengah) yang perlu dilakukan {data['nama_panggilan']} (misalnya pembuatan portofolio khusus, optimalisasi LinkedIn, pendaftaran ke program tertentu, atau pivot bisnis).\n"
                f"4. **Potensi Jalur Karir Masa Depan**: Proyeksikan 2-3 pilihan jalur karir atau model bisnis baru yang sangat potensial untuk dirambah dalam 2-5 tahun ke depan.\n\n"
                f"Gunakan bahasa Indonesia yang profesional, inspiratif, dan jelas. Fokus pada nama panggilan ({data['nama_panggilan']}). Hindari kata-kata menggantung atau kalimat yang tidak padu."
            ),
            "en": (
                "Main Alumni Profile:\n"
                f"Name: {data['nama']} (Nickname: {data['nama_panggilan']})\n"
                f"Activity: {aktivitas_list_display}\n"
                f"Skills Summary: {data['skills']}\n\n"
                f"--- DETAILED PROFILE (from gabungan_data) ---\n{data['gabungan_data']}\n"
                f"-------------------------------------------\n\n"
                f"{peluang_str.replace('Konteks Peluang dari Database', 'Opportunity Context from Database').replace('Peluang Bisnis', 'Business Opportunities').replace('Peluang dari Pekerja', 'Opportunities from Professionals').replace('Peluang dari Ibu Rumah Tangga', 'Opportunities from Homemakers').replace('Tidak ada data.', 'No data.')}\n"
                f"{top_alumni_kolaborasi_content}\n\n"
                f"YOUR TASK:\n"
                f"Based on the profile and data above, provide a comprehensive career development recommendation for {data['nama_panggilan']} in the following format:\n"
                f"1. **Current Career Strength & Potential Analysis**: A deep summary of {data['nama_panggilan']}'s core strengths and how their current skills position them in the industry/market.\n"
                f"2. **Upskilling & Certification Recommendations**: Specify 3-4 new skills (hard/soft skills) or professional certifications they should acquire to boost their competitiveness or expand their business.\n"
                f"3. **Strategic Career Action Plan**: 4-5 concrete, realistic, and sequential steps (short-to-medium term) {data['nama_panggilan']} needs to take (e.g., building a specific portfolio, optimizing LinkedIn, applying for programs, or pivoting business).\n"
                f"4. **Future Career Path Potential**: Project 2-3 potential career paths or new business models that are highly relevant to pursue in the next 2-5 years.\n\n"
                f"Use professional, inspiring, and clear English. Focus on the nickname ({data['nama_panggilan']}). Avoid dangling or disconnected phrasing."
            )
        }
    else:
        prompt_template = {
            "id": (
                "Profil Alumni Utama:\n"
                f"Nama: {data['nama']} (Panggilan: {data['nama_panggilan']})\n"
                f"Aktivitas: {aktivitas_list_display}\n"
                f"Ringkasan Keahlian: {data['skills']}\n\n"
                f"--- PROFIL DETAIL (dari gabungan_data) ---\n{data['gabungan_data']}\n"
                f"----------------------------------------\n\n"
                f"{peluang_str}\n"
                f"{top_alumni_kolaborasi_content}\n"
                f"{prompt_tambahan_content}\n"
                f"TUGAS ANDA:\n"
                f"Berdasarkan semua informasi di atas (termasuk konteks tambahan dari pengguna jika ada), berikan analisis komprehensif dalam format berikut:\n"
                f"1. **Ringkasan Profil {data['nama_panggilan']}**: Buat ringkasan naratif yang menyoroti kekuatan utama, keahlian, dan potensi dari {data['nama_panggilan']}.\n"
                f"2. **Analisis Peluang Kolaborasi**: Identifikasi 4-5 peluang paling relevan dari 'Konteks Peluang dari Database'. Jelaskan secara spesifik bagaimana {data['nama_panggilan']} bisa berkolaborasi atau mengisi kebutuhan tersebut. Jika relevan, sebutkan nama alumni dari 'Konteks Tambahan' yang bisa menjadi partner dalam kolaborasi ini.\n"
                f"3. **Rekomendasi Aksi Konkret**: Berikan 4-5 rekomendasi langkah nyata yang bisa diambil {data['nama_panggilan']} untuk pengembangan karir atau proyeknya, berdasarkan profil dan peluang yang ada.\n"
                f"4. **Ide Proyek Kolaborasi**: Usulkan 5 **judul proyek** yang konkret dan kreatif. Setiap ide harus melibatkan {data['nama_panggilan']} dan minimal satu alumni lain dari 'Konteks Tambahan'. Contoh: 'Workshop Digital Marketing untuk UMKM Kuliner oleh {data['nama_panggilan']} dan [Nama Alumni Lain]'.\n\n"
                f"Gunakan bahasa Indonesia yang profesional dan positif. Fokus pada nama panggilan ({data['nama_panggilan']})."
            ),
            "en": (
                "Main Alumni Profile:\n"
                f"Name: {data['nama']} (Nickname: {data['nama_panggilan']})\n"
                f"Activity: {aktivitas_list_display}\n"
                f"Skills Summary: {data['skills']}\n\n"
                f"--- DETAILED PROFILE (from gabungan_data) ---\n{data['gabungan_data']}\n"
                f"-------------------------------------------\n\n"
                f"{peluang_str.replace('Konteks Peluang dari Database', 'Opportunity Context from Database').replace('Peluang Bisnis', 'Business Opportunities').replace('Peluang dari Pekerja', 'Opportunities from Professionals').replace('Peluang dari Ibu Rumah Tangga', 'Opportunities from Homemakers').replace('Tidak ada data.', 'No data.')}\n"
                f"{top_alumni_kolaborasi_content}\n"
                f"{prompt_tambahan_content}\n"
                f"YOUR TASK:\n"
                f"Based on all the information above (including any additional user context), provide a comprehensive analysis in the following format:\n"
                f"1. **{data['nama_panggilan']}'s Profile Summary**: Create a narrative summary highlighting {data['nama_panggilan']}'s key strengths, skills, and potential.\n"
                f"2. **Collaboration Opportunity Analysis**: Identify the 4-5 most relevant opportunities from the 'Opportunity Context from Database'. Specifically explain how {data['nama_panggilan']} can collaborate or fill those needs. If relevant, mention names from the 'Additional Context' who could be partners in this collaboration.\n"
                f"3. **Concrete Action Recommendations**: Provide 4-5 tangible steps {data['nama_panggilan']} can take for career or project development based on the available profile and opportunities.\n"
                f"4. **Collaboration Project Ideas**: Propose 5 concrete and creative **project titles**. Each idea must involve {data['nama_panggilan']} and at least one other alumnus from the 'Additional Context'. Example: 'Digital Marketing Workshop for Culinary SMEs by {data['nama_panggilan']} and [Other Alumnus Name]'.\n\n"
                f"Use professional and positive English. Focus on the nickname ({data['nama_panggilan']})."
            )
        }
    return prompt_template.get(language.lower(), prompt_template["id"])

async def cari_alumni_untuk_proyek(project_text: str, cohort_id: int = None):
    """
    OPTIMAL: Mencari alumni untuk proyek hanya dengan 1 query ke alumni_db.
    Mendukung penyaringan berdasarkan cohort_id jika diisi.
    """
    conn = await asyncpg.connect(SUPABASE_DB_URL, statement_cache_size=0)
    try:
        if cohort_id:
            all_alumni_general = await conn.fetch("""
                SELECT ad.id, ad.nama_lengkap, ad.aktivitas, ad.skill_gabungan, ad.gabungan_data 
                FROM alumni_db ad
                JOIN cohort_members cm ON cm.user_id = ad.id
                WHERE cm.cohort_id = $1
            """, cohort_id)
        else:
            all_alumni_general = await conn.fetch(
                "SELECT id, nama_lengkap, aktivitas, skill_gabungan, gabungan_data FROM alumni_db"
            )

        project_tokens = tokenize_text(project_text)
        alumni_candidates = []

        for alumni_gen in all_alumni_general:
            alumni_skills = alumni_gen["skill_gabungan"] or ""
            alumni_details = alumni_gen["gabungan_data"] or ""
            alumni_full_profile_text = f"{alumni_skills} {alumni_details}".lower().strip()

            match_score = compute_weighted_match_score(project_tokens, alumni_full_profile_text)
            other_aktivitas_values = normalize_aktivitas_values(alumni_gen["aktivitas"])
            if any(a.lower() in project_text.lower() for a in other_aktivitas_values):
                match_score += 1.0
            
            if match_score > 0:
                matched_terms = extract_matched_terms(project_tokens, alumni_full_profile_text)
                alumni_candidates.append({
                    "id": alumni_gen["id"],
                    "nama_lengkap": alumni_gen["nama_lengkap"],
                    "aktivitas": alumni_gen["aktivitas"],
                    "skills_gabungan": alumni_skills,
                    "full_profile_text": f"{alumni_skills}. {alumni_details}",
                    "match_score": round(match_score, 2),
                    "matched_terms": matched_terms,
                })
        
        alumni_candidates.sort(key=lambda x: x['match_score'], reverse=True)
        return alumni_candidates[:10]
    finally:
        await conn.close()

def build_proyek_prompt(proyek_input_data, recommended_alumni, language):
    """
    Membangun prompt untuk LLM berdasarkan ide proyek dan alumni yang direkomendasikan.
    Versi ini memiliki instruksi yang lebih jelas untuk overview dan jumlah talent.
    """
    proyek_info = f"Ide Proyek: {proyek_input_data.ide_proyek}\n"

    alumni_list_content = ""
    if recommended_alumni:
        alumni_list_parts = []
        # Mengambil maksimal 10 alumni, sesuai dengan permintaan
        for alumni in recommended_alumni[:10]:
            summary = alumni['full_profile_text']
            alumni_list_parts.append(f"- Nama: {alumni['nama_lengkap']} (Aktivitas: {alumni['aktivitas']}, Keahlian Utama: {alumni['skills_gabungan']}).\n  Profil Detail: {summary}")
        
        if language.lower() == 'id':
            alumni_list_content = "Berikut adalah daftar alumni paling relevan yang ditemukan di database (maksimal 10):\n" + "\n".join(alumni_list_parts)
        else:
            alumni_list_content = "The following are the most relevant alumni found in the database (up to 10):\n" + "\n".join(alumni_list_parts)
    else:
        alumni_list_content = "Tidak ada alumni yang relevan ditemukan di database untuk proyek ini." if language.lower() == "id" else "No relevant alumni found in the database for this project."

    prompt_template = {
        "id": (
            f"Anda adalah seorang Talent Scout AI yang cerdas dan strategis.\n\n"
            f"**PROYEK YANG DIAJUKAN:**\n{proyek_info}\n"
            f"**KANDIDAT ALUMNI TERATAS (berdasarkan relevansi kata kunci):**\n{alumni_list_content}\n\n"
            f"**TUGAS ANDA:**\n"
            f"Daftar kandidat di atas HANYA konteks — jangan sebutkan atau daftar nama kandidat satu per satu dalam jawaban Anda, karena aplikasi sudah menampilkan setiap kandidat secara terpisah dalam bentuk visual. Berdasarkan deskripsi proyek dan gambaran umum kandidat yang tersedia, berikan analisis mendalam dengan bagian-bagian berikut (tanpa menyebut nama individu kandidat di bagian manapun):\n\n"
            f"**1. Gambaran Umum Proyek & Kebutuhan Talenta**\n"
            f"Berikan *overview* singkat (2-3 kalimat) yang merangkum tujuan utama proyek. Setelah itu, jabarkan jenis keahlian dan peran kunci yang dibutuhkan untuk menyukseskan proyek ini (contoh: Manajer Proyek, Ahli Pemasaran Digital, Desainer Grafis, dll.).\n\n"
            f"**2. Analisis Kesenjangan Keahlian (Skill Gap)**\n"
            f"Berdasarkan pola keahlian yang terlihat di antara para kandidat, identifikasi 2-3 area keahlian yang KUAT tersedia di antara kandidat, dan 1-2 area yang masih KURANG/langka dibanding kebutuhan proyek. Jelaskan implikasinya bagi keberhasilan proyek.\n\n"
            f"**3. Strategi Menyusun Tim**\n"
            f"Dalam 3-4 kalimat, berikan saran konkret bagaimana sebaiknya peran-peran dibagi/disusun berdasarkan pola keahlian yang tersedia (misalnya, seberapa banyak yang cenderung generalis vs spesialis, dan bagaimana mengombinasikan mereka menjadi tim yang efektif).\n\n"
            f"**4. Rekomendasi Langkah Selanjutnya**\n"
            f"Berikan 2-3 langkah aksi konkret bagi pemilik proyek — misalnya area mana yang perlu dicari talentanya lebih lanjut di luar kandidat yang ada, atau pertanyaan penting yang perlu ditanyakan saat wawancara kandidat."
        ),
        "en": (
            f"You are an intelligent and strategic AI Talent Scout.\n\n"
            f"**SUBMITTED PROJECT:**\n{proyek_info}\n"
            f"**TOP ALUMNI CANDIDATES (based on keyword relevance):**\n{alumni_list_content}\n\n"
            f"**YOUR TASK:**\n"
            f"The candidate list above is context ONLY — do not name or list individual candidates anywhere in your answer, since the app already displays each one separately in a visual format. Based on the project description and the general shape of the available candidates, provide an in-depth analysis with the following sections (without naming any individual candidate anywhere):\n\n"
            f"**1. Project Overview & Talent Needs**\n"
            f"Provide a brief overview (2-3 sentences) summarizing the project's main goal. Afterward, list the key skills and roles needed to make this project successful (e.g., Project Manager, Digital Marketing Specialist, Graphic Designer, etc.).\n\n"
            f"**2. Skill Gap Analysis**\n"
            f"Based on the skill patterns visible among the candidates, identify 2-3 skill areas that are STRONGLY represented among candidates, and 1-2 areas that are LACKING relative to the project's needs. Explain the implications for the project's success.\n\n"
            f"**3. Team-Building Strategy**\n"
            f"In 3-4 sentences, give concrete advice on how the roles should be divided based on the available skill patterns (e.g. how many lean generalist vs. specialist, and how to combine them into an effective team).\n\n"
            f"**4. Next-Step Recommendations**\n"
            f"Give 2-3 concrete action steps for the project owner — e.g. which skill areas still need sourcing beyond the current candidates, or important questions to ask when interviewing candidates."
        )
    }
    return prompt_template.get(language.lower(), prompt_template["id"])

# --- ENDPOINTS API (CONTROLLERS) ---

@app.post("/rekomendasi", dependencies=[Security(get_api_key)])
async def rekomendasi(input: RekomendasiInput):
    try:
        if input.user_id is None and not input.nama_lengkap:
            raise HTTPException(status_code=400, detail="user_id atau nama_lengkap wajib diisi")
        data = await ambil_profil_alumni(user_id=input.user_id, nama_lengkap=input.nama_lengkap, cohort_id=input.cohort_id)
        data["prompt_tambahan"] = input.prompt_tambahan
        prompt = build_prompt(data, input.language, source="profile")

        content = await call_llm_service(prompt)
        candidates = build_candidate_cards(
            data.get("top_alumni_kolaborasi", []),
            name_key="nama_alumni_kolaborasi",
            skills_key="relevance_skills",
        )
        return {"rekomendasi": content.strip(), "candidates": candidates}

    except HTTPException as e:
        raise e
    except Exception as e:
        error_traceback = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}\n\nTraceback:\n{error_traceback}")

@app.post("/wawasan", dependencies=[Security(get_api_key)])
async def wawasan(input: WawasanInput):
    try:
        if input.user_id is None and not input.nama_lengkap:
            raise HTTPException(status_code=400, detail="user_id atau nama_lengkap wajib diisi")
        data = await ambil_profil_alumni(user_id=input.user_id, nama_lengkap=input.nama_lengkap, cohort_id=input.cohort_id)
        prompt = build_prompt(data, input.language, source="home")

        content = await call_llm_service(prompt)
        return {"wawasan": content.strip()}

    except HTTPException as e:
        raise e
    except Exception as e:
        error_traceback = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}\n\nTraceback:\n{error_traceback}")


@app.post("/karir", dependencies=[Security(get_api_key)])
async def karir(input: RekomendasiInput):
    try:
        if input.user_id is None and not input.nama_lengkap:
            raise HTTPException(status_code=400, detail="user_id atau nama_lengkap wajib diisi")
        data = await ambil_profil_alumni(user_id=input.user_id, nama_lengkap=input.nama_lengkap, cohort_id=input.cohort_id)
        prompt = build_prompt(data, input.language, source="karir")

        content = await call_llm_service(prompt)
        return {"karir": content.strip()}

    except HTTPException as e:
        raise e
    except Exception as e:
        error_traceback = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}\n\nTraceback:\n{error_traceback}")


@app.post("/proyek_rekomendasi", dependencies=[Security(get_api_key)])
async def proyek_rekomendasi(input: ProyekInput):
    try:
        project_text = input.ide_proyek.strip()
        if not project_text:
            raise HTTPException(status_code=400, detail="Ide proyek tidak boleh kosong.")

        recommended_alumni_data = await cari_alumni_untuk_proyek(project_text, cohort_id=input.cohort_id)
        prompt = build_proyek_prompt(input, recommended_alumni_data, input.language)

        content = await call_llm_service(prompt)
        candidates = build_candidate_cards(
            recommended_alumni_data,
            name_key="nama_lengkap",
            skills_key="skills_gabungan",
        )
        return {"rekomendasi_proyek": content.strip(), "candidates": candidates}

    except HTTPException as e:
        raise e
    except Exception as e:
        error_traceback = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}\n\nTraceback:\n{error_traceback}")


@app.post("/learning_path", dependencies=[Security(get_api_key)])
async def learning_path(input: LearningPathInput):
    try:
        target_role = sanitize_ai_input(input.target_role)
        if not target_role:
            raise HTTPException(status_code=400, detail="target_role wajib diisi.")

        data = await ambil_profil_alumni(user_id=input.user_id)
        profile_summary = sanitize_ai_input(f"{data.get('skills', '')}. {data.get('gabungan_data', '')}", max_length=1800)
        education_summary = sanitize_ai_input("; ".join(data.get("education_summaries", [])), max_length=600)

        # Local RAG / Semantic Matchmaking against ALL active jobs in database
        jobs_context = ""
        conn = await asyncpg.connect(SUPABASE_DB_URL, statement_cache_size=0)
        try:
            # Fetch all active jobs from the database for ranking
            all_active_jobs = await conn.fetch("""
                SELECT job_title, company, description, job_desk, requirements, category
                FROM jobs
                WHERE is_active = true
            """)
            
            if all_active_jobs:
                role_tokens = tokenize_text(target_role)
                matched_jobs = []
                
                for job in all_active_jobs:
                    # Combine texts for semantic match score
                    job_text = f"{job['job_title']} {job['category']} {job['description']}"
                    if job['requirements']:
                        job_text += " " + " ".join(job['requirements'])
                    if job['job_desk']:
                        job_text += " " + " ".join(job['job_desk'])
                        
                    score = compute_weighted_match_score(role_tokens, job_text)
                    
                    # Boost score if target role is in job title
                    if target_role.lower() in job['job_title'].lower():
                        score += 5.0
                        
                    if score > 0:
                        matched_jobs.append({
                            "job": job,
                            "score": score
                        })
                
                # Sort by matching score descending
                matched_jobs.sort(key=lambda x: x['score'], reverse=True)
                
                if matched_jobs:
                    # Get top 5 matched jobs as representative examples
                    top_matches = matched_jobs[:5]
                    
                    # Aggregate requirements and job desk responsibilities from all matched jobs (baseline)
                    all_requirements = []
                    all_tasks = []
                    for mj in matched_jobs:
                        j = mj['job']
                        if j['requirements']:
                            all_requirements.extend([r.strip() for r in j['requirements'] if r.strip()])
                        if j['job_desk']:
                            all_tasks.extend([t.strip() for t in j['job_desk'] if t.strip()])
                    
                    # Deduplicate and limit key requirements and tasks
                    unique_reqs = list(dict.fromkeys(all_requirements))[:15]
                    unique_tasks = list(dict.fromkeys(all_tasks))[:15]
                    
                    jobs_list_str = []
                    for mj in top_matches:
                        j = mj['job']
                        jobs_list_str.append(f"- {j['job_title']} di {j['company']} (Skor Match: {mj['score']})")
                        
                    jobs_context = (
                        f"Ditemukan {len(matched_jobs)} lowongan kerja aktif yang cocok di database lokal.\n"
                        f"5 Contoh Lowongan Teratas:\n" + "\n".join(jobs_list_str) + "\n\n"
                        f"Persyaratan Utama Teragregasi (Baseline Kebutuhan dari seluruh lowongan cocok):\n" + "\n".join([f"- {req}" for req in unique_reqs]) + "\n\n"
                        f"Tanggung Jawab Teragregasi (Baseline Tugas dari seluruh lowongan cocok):\n" + "\n".join([f"- {task}" for task in unique_tasks])
                    )
        except Exception as e:
            print(f"Error executing jobs RAG: {e}")
        finally:
            await conn.close()

        prompt = (
            "Anda adalah mentor karir AI & Talent Scout untuk HubTalent Indonesia. "
            "Buat analisis gap skill dan roadmap belajar sekaligus roadmap pencarian pengalaman (learning & experience path) yang spesifik, realistis, dan bisa dieksekusi.\n\n"
            f"Target role: {target_role}\n"
            f"Aktivitas pengguna saat ini: {data.get('aktivitas_primary', 'tidak diketahui')}\n"
            f"Ringkasan skill dan profil: {profile_summary}\n"
            f"Riwayat pendidikan: {education_summary or 'Tidak ada data'}\n\n"
        )

        if jobs_context:
            prompt += (
                "BERIKUT ADALAH LOWONGAN KERJA NYATA YANG SEDANG AKTIF DI DATABASE HUBTALENT UNTUK PERAN INI:\n"
                f"{jobs_context}\n\n"
                "Instruksi Khusus: Analisis gap skill dan kurikulum harus disesuaikan secara langsung untuk melatih "
                "keahlian yang diminta oleh lowongan nyata di atas (terutama bagian Persyaratan Utama dan Tugas/Job Desk).\n\n"
            )
        else:
            prompt += (
                "Catatan: Saat ini tidak ditemukan lowongan kerja spesifik yang aktif di database lokal kami untuk peran ini. "
                "Oleh karena itu, buatlah rekomendasi berdasarkan standar umum industri nasional/global untuk peran target tersebut.\n\n"
            )

        prompt += (
            "Kembalikan HANYA JSON valid, tanpa teks lain, dengan skema berikut:\n"
            "{\n"
            "  \"gap_analysis\": [\n"
            "    {\"skill\": \"...\", \"description\": \"...\"}\n"
            "  ],\n"
            "  \"learning_path\": [\n"
            "    {\"step\": 1, \"topic\": \"...\", \"courses_certs\": [\"...\"], \"action_plan\": \"...\"}\n"
            "  ],\n"
            "  \"checklist\": [\"...\"]\n"
            "}\n"
            "Aturan penting: Hasil dalam Bahasa Indonesia. Maksimal 5 item gap_analysis, 6 step learning_path, 12 checklist item. "
            "Pastikan setidaknya 2 dari langkah learning_path dan 4 dari checklist item berfokus pada kegiatan praktis "
            "seperti membuat portofolio, mengerjakan proyek riil/freelance, kontribusi open source, atau magang untuk mencari pengalaman."
        )

        raw_response = await call_llm_service(prompt, temperature=0.55)
        parsed = parse_learning_path_json(raw_response)
        return parsed

    except HTTPException as e:
        raise e
    except ValueError as e:
        raise HTTPException(status_code=500, detail=f"Format respons AI tidak valid: {str(e)}")
    except Exception as e:
        error_traceback = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}\n\nTraceback:\n{error_traceback}")


# --- SIMULASI WAWANCARA AI ---
# Jumlah pertanyaan sebelum sesi ditutup dengan feedback. Sengaja dijaga singkat
# (target ~5 menit per sesi) agar terasa sebagai latihan cepat, bukan wawancara penuh.
INTERVIEW_QUESTION_COUNT = 5


def build_interview_prompt(target_role: str, profile_summary: str, aktivitas: str, history: List[InterviewTurn]) -> str:
    transcript = "\n".join(
        f"{'Pewawancara' if turn.role == 'ai' else 'Kandidat'}: {sanitize_ai_input(turn.content, max_length=1000)}"
        for turn in history
    )
    question_number = sum(1 for turn in history if turn.role == "user") + 1

    prompt = (
        "Anda adalah pewawancara HR & teknis berpengalaman yang sedang melatih kandidat untuk peran berikut "
        f"di HubTalent Indonesia: {target_role}.\n"
        f"Latar belakang kandidat: {aktivitas or 'tidak diketahui'}. Ringkasan profil: {profile_summary or 'Tidak ada data'}.\n\n"
    )

    if transcript:
        prompt += f"Transkrip wawancara sejauh ini:\n{transcript}\n\n"
        prompt += (
            f"Ini adalah pertanyaan ke-{question_number} dari {INTERVIEW_QUESTION_COUNT}. "
            "Berdasarkan jawaban kandidat sebelumnya, ajukan SATU pertanyaan lanjutan yang relevan — boleh menggali "
            "lebih dalam jawaban terakhir, atau berpindah ke aspek lain (teknis, situasional, atau soft skill) agar sesi tetap variatif. "
            "Jangan mengulang pertanyaan yang sudah ditanyakan.\n"
        )
    else:
        prompt += (
            "Ajukan SATU pertanyaan pembuka wawancara yang natural untuk peran ini — bisa perkenalan singkat, "
            "motivasi melamar, atau pengalaman relevan.\n"
        )

    prompt += (
        "Balas HANYA dengan teks pertanyaan itu sendiri, dalam Bahasa Indonesia, tanpa nomor, tanda kutip, "
        "label 'Pewawancara:', atau penjelasan tambahan apapun."
    )
    return prompt


def build_interview_feedback_prompt(target_role: str, profile_summary: str, history: List[InterviewTurn]) -> str:
    transcript = "\n".join(
        f"{'Pewawancara' if turn.role == 'ai' else 'Kandidat'}: {sanitize_ai_input(turn.content, max_length=1000)}"
        for turn in history
    )
    return (
        "Anda adalah pewawancara HR & teknis berpengalaman yang baru saja menyelesaikan sesi latihan wawancara "
        f"untuk peran {target_role} di HubTalent Indonesia.\n"
        f"Ringkasan profil kandidat: {profile_summary or 'Tidak ada data'}.\n\n"
        f"Transkrip lengkap sesi:\n{transcript}\n\n"
        "Berikan evaluasi singkat, jujur, dan membangun atas performa kandidat sepanjang sesi ini. "
        "Kembalikan HANYA JSON valid, tanpa teks lain, dengan skema berikut:\n"
        "{\n"
        "  \"overall_score\": <angka 1-10>,\n"
        "  \"summary\": \"...\",\n"
        "  \"strengths\": [\"...\"],\n"
        "  \"improvements\": [\"...\"],\n"
        "  \"closing_tip\": \"...\"\n"
        "}\n"
        "Aturan: Hasil dalam Bahasa Indonesia. Maksimal 4 item strengths, 4 item improvements. "
        "summary maksimal 3 kalimat. closing_tip berupa satu saran konkret untuk sesi wawancara sungguhan berikutnya."
    )


@app.post("/interview_simulation", dependencies=[Security(get_api_key)])
async def interview_simulation(input: InterviewSimulationInput):
    try:
        target_role = sanitize_ai_input(input.target_role)
        if not target_role:
            raise HTTPException(status_code=400, detail="target_role wajib diisi.")

        data = await ambil_profil_alumni(user_id=input.user_id)
        profile_summary = sanitize_ai_input(f"{data.get('skills', '')}. {data.get('gabungan_data', '')}", max_length=1200)
        aktivitas = data.get("aktivitas_primary", "")

        answered_count = sum(1 for turn in input.conversation_history if turn.role == "user")

        if answered_count >= INTERVIEW_QUESTION_COUNT:
            prompt = build_interview_feedback_prompt(target_role, profile_summary, input.conversation_history)
            raw_response = await call_llm_service(prompt, temperature=0.5)
            feedback = parse_interview_feedback_json(raw_response)
            return {"is_complete": True, "feedback": feedback}

        prompt = build_interview_prompt(target_role, profile_summary, aktivitas, input.conversation_history)
        raw_response = await call_llm_service(prompt, temperature=0.75)
        question = raw_response.strip().strip('"')
        return {
            "is_complete": False,
            "question": question,
            "question_number": answered_count + 1,
            "total_questions": INTERVIEW_QUESTION_COUNT,
        }

    except HTTPException as e:
        raise e
    except ValueError as e:
        raise HTTPException(status_code=500, detail=f"Format respons AI tidak valid: {str(e)}")
    except Exception as e:
        error_traceback = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}\n\nTraceback:\n{error_traceback}")