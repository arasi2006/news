import numpy as np, torch, streamlit as st
import rumor_gin as R
from transformers import AutoModel, AutoTokenizer


@st.cache_resource
def load():
    tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    bert = AutoModel.from_pretrained("bert-base-uncased").eval()
    model = R.RumorGIN(layers=4)
    model.load_state_dict(torch.load("model.pt", map_location="cpu"))
    model.eval()
    return tok, bert, model, np.load("user_mean.npy"), np.load("sim_mean.npy")


@torch.no_grad()
def predict(source, replies):
    tok, bert, model, user_mean, sim_mean = load()
    texts, edges = [source.strip()], []
    lines = [l.strip() for l in replies.split("\n") if l.strip()]
    for i, l in enumerate(lines, start=1):
        parent = 0
        if ">" in l and l.split(">", 1)[0].strip().isdigit():
            p, l = l.split(">", 1)
            p = int(p)
            parent = p if 0 <= p < i else 0
        texts.append(l.strip())
        edges.append((parent, i))
    if not edges:
        edges = [(0, 0)]
    n = len(texts)
    enc = tok(texts, padding=True, truncation=True, max_length=64, return_tensors="pt")
    emb = bert(**enc, output_hidden_states=True).hidden_states[-2][:, 0].numpy()
    ev = R.make_event(emb, np.tile(user_mean, (n, 1)), np.tile(sim_mean, (n, 1)), edges, 0)
    xt, xu, xs, ei, b, y, B = R.collate([ev])
    prob = torch.softmax(model(xt, xu, xs, ei, b, B), 1)[0].numpy()
    return float(prob[1])


st.title("Rumor Detection (GIN + Data Fusion)")
st.caption("Research demo trained on PHEME. Not a fact-checking tool.")
source = st.text_area("Source tweet", "Police confirm road closed near Anna bridge after a truck accident, avoid the area")
replies = st.text_area("Replies (one per line, '1>text' = reply to 1st reply)",
                       "Thanks for the update, taking another route\n1>Confirmed, I just passed there\nStay safe everyone")
if st.button("Predict") and source.strip():
    with st.spinner("Running model..."):
        p = predict(source, replies)
    st.subheader("Rumor" if p >= 0.5 else "Non-rumor")
    st.progress(p, text=f"Rumor probability: {p:.0%}")
