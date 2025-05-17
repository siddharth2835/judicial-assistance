# legalbot_with_auth.py
import os, yaml, bcrypt, numpy as np, streamlit as st
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
import streamlit_authenticator as stauth

MONGO_URI = st.secrets["MONGO_URI"]         # set in Cloud → Secrets
client     = MongoClient(MONGO_URI, serverSelectionTimeoutMS=8000)

try:
    client.admin.command("ping")            # fast connectivity check
except Exception as e:
    st.error(f"MongoDB connection failed: {e}")
    st.stop()      
DB_NAME     = "chatbot_db"                              
db      = client[DB_NAME]
qa_col  = db["sq_ans"]
user_col= db["users"]


# ----------------------- 2. AUTH HELPERS -------------------
def fetch_credentials():
    """Build the dict structure streamlit-authenticator expects, from MongoDB."""
    cred_dict = {"usernames": {}}
    for doc in user_col.find():
        cred_dict["usernames"][doc["username"]] = {
            "name"    : doc["name"],
            "email"   : doc["email"],
            "password": doc["password"],   # already hashed
        }
    # cookie settings – tweak if you like
    full_config = {
        "credentials": cred_dict,
        "cookie": {
            "name" : "legalbot_cookie",
            "key"  : "supersecret_cookie_key",   # change in prod
            "expiry_days": 7
        },
        "preauthorized": {"emails": []}          # optional allow-list
    }
    return full_config

def add_user(username, name, email, raw_pwd):
    hashed = bcrypt.hashpw(raw_pwd.encode(), bcrypt.gensalt()).decode()
    user_col.insert_one(
        {"username": username, "name": name, "email": email, "password": hashed}
    )

# ----------------------- 3. PAGE CONFIG --------------------
st.set_page_config(page_title="JAI - Judicial Assistance Interface", page_icon="⚖️", layout="wide")

# ----------------------- 4. AUTHENTICATION FLOW ------------
config = fetch_credentials()
authenticator = stauth.Authenticate(
    config["credentials"],
    config["cookie"]["name"],
    config["cookie"]["key"],
    config["cookie"]["expiry_days"],
)

with st.sidebar:
    name, auth_status, username = authenticator.login(
        "Login"   
    ) 
# ---------- unsuccessful states ----------
if auth_status is False:
    st.sidebar.error("Username / password incorrect.")
    st.stop()
elif auth_status is None:
    st.sidebar.info("Please log in.")
    # ------- registration form (visible only when not logged in) ------
    with st.sidebar.expander("Register new account"):
        with st.form("register"):
            r_name  = st.text_input("Full name")
            r_user  = st.text_input("Username (unique)")
            r_email = st.text_input("Email")
            r_pwd   = st.text_input("Password", type="password")
            r_sub   = st.form_submit_button("Create")
            if r_sub:
                if user_col.find_one({"username": r_user}):
                    st.warning("Username already exists.")
                else:
                    add_user(r_user, r_name, r_email, r_pwd)
                    st.success("User created. Please log in.")
                    
if not auth_status:
    # Either not entered credentials yet (None) or entered wrong ones (False)
    st.image("jai_logo.png", width=200)
    st.title("⚖️ JAI — Judicial Assistance Interface")
    st.markdown(
        """
        Welcome to **JAI** — your instant helper for judicial FAQs.<br>
        👉 Please log in via the sidebar to start chatting.
        """,
        unsafe_allow_html=True,
    )
    st.stop()

# ------------  logged-in: show log-out  -------------
with st.sidebar.expander("Logout"):
    authenticator.logout("Logout")
st.sidebar.success(f"Logged in as **{name}**")

# ----------------------- 5. LOAD MODEL & DATA -------------
@st.cache_resource(ttl=3000)
def load_model():
    return SentenceTransformer("paraphrase-MiniLM-L3-v2")

@st.cache_resource
def load_qas():
    docs = list(qa_col.find({}, {"_id": 0}))
    emb  = np.array([d["embedding"] for d in docs], dtype=np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    return docs, emb

model, (qa_docs, EMB) = load_model(), load_qas()

# ----------------------- 6. UI STATE -----------------------
if "chat" not in st.session_state: st.session_state.chat = []
if "input_text" not in st.session_state: st.session_state.input_text = ""

BUBBLE_CSS = """
<style>
.user-bubble, .bot-bubble{
  display:inline-block;                /* let width follow the text   */
  padding:10px;
  border-radius:10px;
  margin:10px 0;
  max-width:70%;                       /* wrap long questions nicely  */
  font-size:0.94rem; line-height:1.45;
  word-wrap:break-word;                /* prevent overflow on long URLs */
}
.user-bubble{ background: lightblue;color:black; margin-left:auto; float:right }
.bot-bubble { background:#D3D3D3; color:black; float:left }
</style>
"""
st.markdown(BUBBLE_CSS, unsafe_allow_html=True)

# ----------------------- 7. SIDEBAR HISTORY ---------------
st.sidebar.markdown("### 🕑 Question History")
for i, (u, _) in enumerate(st.session_state.chat, 1):
    st.sidebar.markdown(f"**{i}.** {u}")

if st.sidebar.button("Clear Chat"):
    st.session_state.chat = []
    

# ----------------------- 8. MAIN APP ----------------------

st.title("⚖️ JAI — Judicial Assistance Interface")

def answer_question():
    q = st.session_state.input_text.strip()
    if not q: return
    v = model.encode([q]).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    idx = int(np.argmax(EMB @ v.T))
    ans = qa_docs[idx]["answer"]
    st.session_state.chat.append((q, ans))
    st.session_state.input_text = ""

# input bar
c1, c2 = st.columns([5,1])
with c1:
    st.text_input("Ask your judicial question:",
                  key="input_text",
                  on_change=answer_question)


# render chat
for u_msg, b_msg in st.session_state.chat:
    st.markdown(f"<div class='user-bubble'>{u_msg}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='bot-bubble'>{b_msg}</div>", unsafe_allow_html=True)
