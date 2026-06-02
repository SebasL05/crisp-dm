import streamlit as st
import pandas as pd
import numpy as np
import re
from pathlib import Path
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="AI Resume Matcher", page_icon="📄", layout="wide")

# ==========================================
# 1. FUNCIONES DE CARGA Y PROCESAMIENTO (CACHED)
# ==========================================
@st.cache_data(show_spinner="Cargando datos y procesando documentos...")
def load_and_process_data():
    base_dir = Path("data")
    vacancies_df = pd.read_csv(base_dir / "5_vacancies.csv")
    
    cv_rows = []
    cv_dir = base_dir / "CV"
    for p in sorted(cv_dir.glob("*.docx"), key=lambda x: int(x.stem)):
        try:
            doc = Document(p)
            text = "\n".join(par.text for par in doc.paragraphs if par.text.strip())
            cv_rows.append({"cv_id": p.stem, "text": text})
        except Exception:
            pass
    cv_df = pd.DataFrame(cv_rows)
    
    # Preprocesar vacantes
    vacancies_df["job_text"] = vacancies_df["job_title"].fillna("") + " " + vacancies_df["job_description"].fillna("")
    
    return vacancies_df, cv_df

@st.cache_data(show_spinner="Calculando modelo de ranking...")
def calculate_ranking(vacancies_df, cv_df):
    # Componente 1: TF-IDF
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=6000)
    all_texts = vacancies_df["job_text"].tolist() + cv_df["text"].tolist()
    X = vectorizer.fit_transform(all_texts)
    
    V_matrix = X[:len(vacancies_df)]
    C_matrix = X[len(vacancies_df):]
    tfidf_sim = cosine_similarity(C_matrix, V_matrix)
    
    # Componente 2: Extracción simple de Skills
    tech_skills = ["python", "java", "c\+\+", "sql", "aws", "docker", "agile", "scrum", "linux", "unix", 
                   "react", "angular", "node", "javascript", "machine learning", "data science", "nlp"]
    
    results = []
    for v_idx, v_row in vacancies_df.iterrows():
        v_text_lower = v_row["job_text"].lower()
        v_skills = [s for s in tech_skills if re.search(r'\b' + s + r'\b', v_text_lower)]
        
        for c_idx, c_row in cv_df.iterrows():
            c_text_lower = c_row["text"].lower()
            c_skills = [s for s in v_skills if re.search(r'\b' + s + r'\b', c_text_lower)]
            
            skill_score = len(c_skills) / len(v_skills) if v_skills else 0.0
            t_score = tfidf_sim[c_idx, v_idx]
            
            # Modelo Híbrido
            final_score = (0.20 * t_score) + (0.80 * skill_score)
            
            results.append({
                "vacancy_id": v_row["id"],
                "job_title": v_row["job_title"],
                "cv_id": c_row["cv_id"],
                "final_score": final_score,
                "tfidf_score": t_score,
                "skill_score": skill_score,
                "matched_skills": ", ".join(c_skills).title() if c_skills else "Ninguna",
                "cv_preview": c_row["text"][:250] + "..."
            })
            
    return pd.DataFrame(results)

# ==========================================
# 2. INTERFAZ DE USUARIO (UI)
# ==========================================
st.title("📄 AI Resume Matcher (CRISP-DM Deployment)")
st.markdown("Esta aplicación automatiza el **rankeo de CVs** frente a vacantes disponibles, utilizando un **Modelo Híbrido** basado en similitud de texto (TF-IDF) y extracción de habilidades técnicas.")

try:
    vac_df, cvs_df = load_and_process_data()
    rankings_df = calculate_ranking(vac_df, cvs_df)
    
    st.sidebar.header("⚙️ Filtros")
    st.sidebar.write(f"**Total Vacantes:** {len(vac_df)}")
    st.sidebar.write(f"**Total CVs Evaluados:** {len(cvs_df)}")
    
    # Seleccionar vacante
    vac_options = {row["id"]: f"ID {row['id']} - {row['job_title']}" for _, row in vac_df.iterrows()}
    selected_vac_id = st.sidebar.selectbox("Seleccione la Vacante a analizar:", options=list(vac_options.keys()), format_func=lambda x: vac_options[x])
    
    # Top N
    top_n = st.sidebar.slider("Mostrar Top N Candidatos", min_value=3, max_value=20, value=5)
    
    # Mostrar descripción de la vacante
    st.subheader(f"💼 Vacante Seleccionada: {vac_options[selected_vac_id]}")
    vac_desc = vac_df[vac_df["id"] == selected_vac_id].iloc[0]["job_description"]
    with st.expander("Ver descripción completa de la vacante"):
        st.write(vac_desc)
        
    # Filtrar resultados
    vac_results = rankings_df[rankings_df["vacancy_id"] == selected_vac_id].sort_values(by="final_score", ascending=False).head(top_n)
    
    st.markdown("---")
    st.subheader(f"🏆 Top {top_n} Mejores Candidatos")
    
    for rank, (_, row) in enumerate(vac_results.iterrows(), 1):
        with st.container():
            col1, col2, col3 = st.columns([1, 3, 1])
            with col1:
                st.metric(label=f"Rank #{rank}", value=f"CV ID: {row['cv_id']}")
            with col2:
                st.write(f"**Skills Coincidentes:** {row['matched_skills']}")
                st.caption(f"_{row['cv_preview']}_")
            with col3:
                st.metric(label="Match %", value=f"{row['final_score'] * 100:.1f}%")
                st.caption(f"TF-IDF: {row['tfidf_score']:.2f} | Skills: {row['skill_score']:.2f}")
            st.divider()

except Exception as e:
    st.error(f"Error al cargar la aplicación: {str(e)}")
    st.info("Asegúrate de que la carpeta 'data' exista y contenga '5_vacancies.csv' y la carpeta 'CV'.")
