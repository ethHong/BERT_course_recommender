import streamlit as st
import pandas as pd
import json
from utils import (
    get_prob,
    device,
    get_prob_lists,
    compare_sentence,
    query_jds,
    query,
    tfidf_score,
    tfidf_matrix,
)
from ast import literal_eval
from stqdm import stqdm

with open("joblists.txt") as file:
    lines = file.readlines()
    jobs = [line.rstrip() for line in lines]

DB = pd.read_csv("JDs_final.csv").dropna()
data = pd.read_csv("processed_courses_data.csv")


def get_recommendation(DB, data, jobname, by="course_info"):
    JD_sentences = query_jds(DB, jobname).description.values
    DB["Query_Score"] = DB.description.progress_apply(
        lambda x: compare_sentence(x, jobs[0])
    )
    target = DB.sort_values(by="Query_Score", ascending=False)[:10]
    JD_sentences = target.description.values
    data["Recommendation_score"] = data[by].progress_apply(
        lambda x: compare_sentence(x, JD_sentences)
    )
    return data.sort_values(by="Recommendation_score", ascending=False)[:26][
        ["Course_Name", "course_info", "syllabus", "div", "Recommendation_score"]
    ]


st.title("Course Recommender🤔")

if device == "cpu":
    processor = "🖥️"
else:
    processor = "💽"

option = st.checkbox("💻From referece IT jobs?")
if option:
    job = st.selectbox("Choose your job", jobs)

else:
    job = st.text_input("Put job you are interested")

btn = st.button("Run recommendation!")
stqdm.pandas()

if btn:
    with st.spinner("⌛Generating Recommendation!"):
        recommendation = get_recommendation(DB, data, job)
    st.success("Recommended course for {} ".format(job))
    st.write(recommendation)
