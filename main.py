from fastapi import FastAPI, HTTPException, Security, status
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from typing import Optional
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

# Endpoint utama untuk verifikasi status
@app.get("/")
def root():
    return {"message": "Alumni AI backend is running!"}

# --- MODEL DATA (PYDANTIC) ---

class RekomendasiInput(BaseModel):
    user_id: Optional[int] = None
    nama_lengkap: Optional[str] = None
    cohort_id: Optional[int] = None
    language: str = "id"

class WawasanInput(BaseModel):
    user_id: Optional[int] = None
    nama_lengkap: Optional[str] = None
    cohort_id: Optional[int] = None
    language: str = "id"

class ProyekInput(BaseModel):
    ide_proyek: str
    cohort_id: Optional[int] = None
    language: str = "id"

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
                all_relevant_alumni.append({
                    "nama_alumni_kolaborasi": alumni_gen["nama_lengkap"],
                    "aktivitas": alumni_gen["aktivitas"],
                    "relevance_skills": other_alumni_skills,
                    "relevance_detail_summary": other_alumni_full_profile_text,
                    "match_score": round(match_score, 2)
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
                f"{top_alumni_kolaborasi_content}\n\n"
                f"TUGAS ANDA:\n"
                f"Berdasarkan semua informasi di atas, berikan analisis komprehensif dalam format berikut:\n"
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
                f"{top_alumni_kolaborasi_content}\n\n"
                f"YOUR TASK:\n"
                f"Based on all the information above, provide a comprehensive analysis in the following format:\n"
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
                alumni_candidates.append({
                    "nama_lengkap": alumni_gen["nama_lengkap"],
                    "aktivitas": alumni_gen["aktivitas"],
                    "skills_gabungan": alumni_skills,
                    "full_profile_text": f"{alumni_skills}. {alumni_details}",
                    "match_score": round(match_score, 2)
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
            f"Berdasarkan deskripsi proyek dan daftar kandidat di atas, berikan analisis berikut dengan format yang jelas:\n\n"
            # --- PERUBAHAN DI SINI (Poin 1) ---
            f"**1. Gambaran Umum Proyek & Kebutuhan Talenta**\n"
            f"Berikan *overview* singkat (2-3 kalimat) yang merangkum tujuan utama proyek. Setelah itu, jabarkan jenis keahlian dan peran kunci yang dibutuhkan untuk menyukseskan proyek ini (contoh: Manajer Proyek, Ahli Pemasaran Digital, Desainer Grafis, dll.).\n\n"
            # --- PERUBAHAN DI SINI (Poin 2) ---
            f"**2. Rekomendasi Talenta (Maksimal 10 Alumni)**\n"
            f"Pilih **hingga 10 alumni** dari daftar kandidat yang paling ideal untuk proyek ini. Untuk setiap alumni yang direkomendasikan, sajikan dalam format daftar poin (bulleted list) yang rapi sebagai berikut:\n"
            f"   - **Nama Alumni:** [Nama Lengkap]\n"
            f"   - **Peran yang Direkomendasikan:** [Contoh: Project Manager, Spesialis Pemasaran, Pengembang Utama]\n"
            f"   - **Justifikasi:** [Jelaskan dalam 1-2 kalimat mengapa keahlian dan profil detail mereka sangat cocok untuk peran tersebut dalam konteks proyek ini].\n\n"
            f"**3. Pesan Penutup**\n"
            f"Berikan satu paragraf penutup yang profesional untuk menyimpulkan rekomendasi."
        ),
        "en": (
            f"You are an intelligent and strategic AI Talent Scout.\n\n"
            f"**SUBMITTED PROJECT:**\n{proyek_info}\n"
            f"**TOP ALUMNI CANDIDATES (based on keyword relevance):**\n{alumni_list_content}\n\n"
            f"**YOUR TASK:**\n"
            f"Based on the project description and the candidate list above, provide the following analysis in a clear format:\n\n"
            # --- CHANGED HERE (Point 1) ---
            f"**1. Project Overview & Talent Needs**\n"
            f"Provide a brief overview (2-3 sentences) summarizing the project's main goal. Afterward, list the key skills and roles needed to make this project successful (e.g., Project Manager, Digital Marketing Specialist, Graphic Designer, etc.).\n\n"
            # --- CHANGED HERE (Point 2) ---
            f"**2. Talent Recommendations (Up to 10 Alumni)**\n"
            f"Select **up to 10** of the most ideal alumni from the candidate list for this project. For each recommended alumnus, present them in a neat bulleted list format as follows:\n"
            f"   - **Alumnus Name:** [Full Name]\n"
            f"   - **Recommended Role:** [Example: Project Manager, Marketing Specialist, Lead Developer]\n"
            f"   - **Justification:** [Explain in 1-2 sentences why their skills and detailed profile are a perfect fit for that role in the context of this project].\n\n"
            f"**3. Closing Message**\n"
            f"Provide a professional closing paragraph to conclude the recommendation."
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
        prompt = build_prompt(data, input.language, source="profile")

        headers = {"Content-Type": "application/json"}
        gemini_api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"

        body = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.7, "maxOutputTokens": 2500}
        }
        
        async with httpx.AsyncClient(timeout=90.0) as client:
            res = await client.post(gemini_api_url, headers=headers, json=body)
            res.raise_for_status()
            content = res.json()["candidates"][0]["content"]["parts"][0]["text"]
            return {"rekomendasi": content.strip()}

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

        headers = {"Content-Type": "application/json"}
        gemini_api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"

        body = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.7, "maxOutputTokens": 2500}
        }
        
        async with httpx.AsyncClient(timeout=90.0) as client:
            res = await client.post(gemini_api_url, headers=headers, json=body)
            res.raise_for_status()
            content = res.json()["candidates"][0]["content"]["parts"][0]["text"]
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

        headers = {"Content-Type": "application/json"}
        gemini_api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"

        body = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.7, "maxOutputTokens": 2500}
        }
        
        async with httpx.AsyncClient(timeout=90.0) as client:
            res = await client.post(gemini_api_url, headers=headers, json=body)
            res.raise_for_status()
            content = res.json()["candidates"][0]["content"]["parts"][0]["text"]
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

        headers = {"Content-Type": "application/json"}
        gemini_api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"

        body = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.7, "maxOutputTokens": 2500}
        }
        
        async with httpx.AsyncClient(timeout=90.0) as client:
            res = await client.post(gemini_api_url, headers=headers, json=body)
            res.raise_for_status()
            content = res.json()["candidates"][0]["content"]["parts"][0]["text"]
            return {"rekomendasi_proyek": content.strip()}

    except HTTPException as e:
        raise e
    except Exception as e:
        error_traceback = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}\n\nTraceback:\n{error_traceback}")


class GapAnalysisInput(BaseModel):
    user_id: int
    target_role: str
    language: str = "id"


@app.post("/learning_path", dependencies=[Security(get_api_key)])
async def learning_path(input: GapAnalysisInput):
    try:
        conn = await asyncpg.connect(SUPABASE_DB_URL)
        try:
            # 1. Fetch user's current skills
            user_row = await conn.fetchrow(
                "SELECT nama_lengkap, skill_gabungan, aktivitas FROM alumni_db WHERE id = $1", 
                input.user_id
            )
            if not user_row:
                raise HTTPException(status_code=404, detail="Profil talenta tidak ditemukan.")
            
            user_name = user_row["nama_lengkap"]
            user_skills = user_row["skill_gabungan"] or ""
            user_activity = user_row["aktivitas"] or ""
            
            # 2. Fetch jobs from jobs table matching the target_role
            jobs = await conn.fetch(
                """
                SELECT job_title, company, requirements, description 
                FROM jobs 
                WHERE (job_title ILIKE $1 OR category ILIKE $1) AND is_active = true 
                ORDER BY last_checked DESC LIMIT 8
                """,
                f"%{input.target_role}%"
            )
            
            # Fallback 1
            if not jobs:
                first_word = input.target_role.split()[0] if input.target_role.strip() else ""
                if first_word:
                    jobs = await conn.fetch(
                        """
                        SELECT job_title, company, requirements, description 
                        FROM jobs 
                        WHERE (job_title ILIKE $1 OR category ILIKE $1) AND is_active = true 
                        ORDER BY last_checked DESC LIMIT 8
                        """,
                        f"%{first_word}%"
                    )
            
            # Fallback 2
            if not jobs:
                jobs = await conn.fetch(
                    """
                    SELECT job_title, company, requirements, description 
                    FROM jobs 
                    WHERE is_active = true 
                    ORDER BY last_checked DESC LIMIT 6
                    """
                )
                
            # 3. Format jobs context
            jobs_context_list = []
            for job in jobs:
                title = job["job_title"]
                company = job["company"]
                reqs = ", ".join(job["requirements"] or []) or job["description"][:300]
                jobs_context_list.append(f"- {title} di {company}. Persyaratan: {reqs}")
            
            jobs_context = "\n".join(jobs_context_list)
            
            # 4. Build prompt
            prompt = (
                f"Analisis Kebutuhan Karir untuk: {user_name}\n"
                f"Target Pekerjaan / Peran: {input.target_role}\n"
                f"Keahlian Pengguna Saat Ini: {user_skills}\n"
                f"Aktivitas Saat Ini: {user_activity}\n\n"
                f"Data Persyaratan Industri Aktual (LinkedIn & Kalibrr):\n"
                f"{jobs_context}\n\n"
                f"TUGAS ANDA:\n"
                f"Bandingkan keahlian pengguna dengan kebutuhan nyata industri di atas. Hasilkan respons JSON yang valid berisi:\n"
                f"1. `gap_analysis`: Daftar keahlian penting yang belum dimiliki pengguna tapi sangat dibutuhkan industri untuk peran '{input.target_role}' (maksimal 4 item). Sertakan penjelasan singkat ('description') mengapa ini penting.\n"
                f"2. `learning_path`: Rencana belajar bertahap (minimal 3 langkah berurutan) untuk menguasai keahlian yang kurang tersebut. Setiap langkah harus memiliki 'step' (integer), 'topic' (nama topik), 'courses_certs' (rekomendasi kursus/sertifikasi spesifik dalam array string, misal: 'Google IT Support Certification', 'React - The Complete Guide on Udemy'), dan 'action_plan' (langkah belajarnya).\n"
                f"3. `checklist`: Daftar aksi konkret (minimal 5 item string) persiapan kerja yang sangat spesifik (misal: 'Perbarui resume dengan skill React', 'Buat portofolio website statis', 'Selesaikan sertifikasi AWS Cloud Practitioner').\n\n"
                f"Gunakan bahasa Indonesia yang profesional, jelas, dan memotivasi."
            )
            
            # 5. Call Gemini API with structured JSON output schema
            headers_api = {"Content-Type": "application/json"}
            gemini_api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
            
            body = {
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.2,
                    "responseMimeType": "application/json",
                    "responseSchema": {
                        "type": "OBJECT",
                        "properties": {
                            "gap_analysis": {
                                "type": "ARRAY",
                                "items": {
                                    "type": "OBJECT",
                                    "properties": {
                                        "skill": {"type": "STRING"},
                                        "description": {"type": "STRING"}
                                    },
                                    "required": ["skill", "description"]
                                }
                            },
                            "learning_path": {
                                "type": "ARRAY",
                                "items": {
                                    "type": "OBJECT",
                                    "properties": {
                                        "step": {"type": "INTEGER"},
                                        "topic": {"type": "STRING"},
                                        "courses_certs": {
                                            "type": "ARRAY",
                                            "items": {"type": "STRING"}
                                        },
                                        "action_plan": {"type": "STRING"}
                                    },
                                    "required": ["step", "topic", "courses_certs", "action_plan"]
                                }
                            },
                            "checklist": {
                                "type": "ARRAY",
                                "items": {"type": "STRING"}
                            }
                        },
                        "required": ["gap_analysis", "learning_path", "checklist"]
                    }
                }
            }
            
            async with httpx.AsyncClient(timeout=120.0) as client:
                res = await client.post(gemini_api_url, headers=headers_api, json=body)
                res.raise_for_status()
                content = res.json()["candidates"][0]["content"]["parts"][0]["text"]
                
                import json
                parsed_json = json.loads(content)
                return parsed_json
                
        finally:
            await conn.close()
            
    except HTTPException as e:
        raise e
    except Exception as e:
        error_traceback = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}\n\nTraceback:\n{error_traceback}")