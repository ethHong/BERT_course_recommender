from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
import torch
from sentence_transformers import SentenceTransformer, util
import gensim
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
import numpy as np
import pandas as pd

device = "cuda:0" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-mnli")
nli_model = (
    AutoModelForSequenceClassification.from_pretrained(
        "facebook/bart-large-mnli"
    ).cuda()
    if torch.cuda.is_available()
    else AutoModelForSequenceClassification.from_pretrained("facebook/bart-large-mnli")
)


def get_prob(sequence, label):
    premise = sequence
    hypothesis = f"This example is {label}."

    # run through model pre-trained on MNLI
    x = tokenizer.encode(
        premise, hypothesis, return_tensors="pt", truncation_strategy="only_first"
    )
    logits = nli_model(x.to(device))[0]

    # we throw away "neutral" (dim 1) and take the probability of
    # "entailment" (2) as the probability of the label being true
    entail_contradiction_logits = logits[:, [0, 2]]
    probs = entail_contradiction_logits.softmax(dim=1)
    prob_label_is_true = probs[:, 1]
    return prob_label_is_true[0].item()


def get_prob_lists(sequence, labels):
    out = []
    for l in labels:
        out.append(get_prob(sequence, l))
    return out


compare_model = SentenceTransformer("sentence-transformers/multi-qa-MiniLM-L6-cos-v1")


def compare_sentence(query, docs):
    query_emb = compare_model.encode(query)
    doc_emb = compare_model.encode(docs)
    scores = util.dot_score(query_emb, doc_emb)[0].to(device).tolist()
    return np.mean(scores)


def query_jds(DB, keyword):
    keywords = " ".join(gensim.utils.simple_preprocess(keyword, deacc=True))
    temp_tf_matrix = tfidf_matrix(DB, tokenized="tokenized", name="Title")
    target = query(DB, keywords, temp_tf_matrix)
    return target


def query(df, keywords, tf_matrix):

    keywords = " ".join(gensim.utils.simple_preprocess(keywords, deacc=True))
    df["Query_score"] = tfidf_score(tf_matrix, keywords)
    q = df.loc[df["Query_score"] > 0.3].sort_values(by="Query_score", ascending=False)

    result = q[:5].reset_index(drop=True)
    # print(result[["Title", "Query_score"]])
    return result.drop("Query_score", axis=1)


def tfidf_score(tf_matrix, keyword):
    vector = np.array([0] * tf_matrix.shape[1])
    for i in keyword.split():
        if i in tf_matrix.index:
            vector = vector + tf_matrix.loc[i].values
    return vector


def tfidf_matrix(data, tokenized="tokenized", name="Course_Name"):
    corpus = [" ".join(i) for i in data[tokenized]]
    tfidf_voctorize = TfidfVectorizer().fit(corpus)

    avg_score = tfidf_voctorize.transform(corpus).toarray().T
    vocab = tfidf_voctorize.get_feature_names()
    courses = data[name].values
    avg_score = preprocessing.minmax_scale(avg_score.T).T
    scores = pd.DataFrame(avg_score, index=vocab, columns=courses)
    return scores
