COMMON_SKILLS = [
    "python", "sql", "java", "javascript", "html", "css", "react", "node", "fastapi",
    "docker", "kubernetes", "aws", "azure", "gcp", "tensorflow", "keras", "pytorch",
    "machine learning", "deep learning", "nlp", "rag", "vector database", "mongodb",
    "postgresql", "spark", "etl", "tableau", "power bi", "excel", "linux", "git",
]

def extract_skills(text, skill_bank=COMMON_SKILLS):
    lowered = str(text).lower()
    return sorted({skill for skill in skill_bank if skill in lowered})

def compare_skills(resume_text, job_description):
    resume_skills = set(extract_skills(resume_text))
    jd_skills = set(extract_skills(job_description))
    return {
        "common_skills": sorted(resume_skills & jd_skills),
        "missing_skills": sorted(jd_skills - resume_skills),
    }