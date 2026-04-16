#dashboard.py
import json
from pathlib import Path
from datetime import datetime
import os

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

try:
    from streamlit_autorefresh import st_autorefresh
except Exception:
    st_autorefresh = None
import requests
from urllib.parse import urljoin

from sklearn.metrics import (
    roc_curve, auc, confusion_matrix,
    precision_score, recall_score,
    f1_score, accuracy_score
)
from sklearn.model_selection import StratifiedKFold

from server_utils import read_json

st.set_page_config(page_title="Teacher Identification System", layout="wide")

APP_DIR = Path(__file__).resolve().parent
DB_PATH = APP_DIR / "db" / "teachers.json"
LOGS_PATH = APP_DIR / "logs" / "attempts.jsonl"
LOGO_PATH = APP_DIR / "logo_mutah.png"

def load_jsonl(path):
    if not path.exists():
        return pd.DataFrame()
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except:
                pass
    return pd.DataFrame(rows)

def get_teacher_image(tid):
    img_dir = APP_DIR / "dataset" / tid / "images"
    if not img_dir.exists():
        return None
    for n in ["front.jpg", "1.jpg", "0.jpg"]:
        p = img_dir / n
        if p.exists():
            return p
    imgs = list(img_dir.glob("*"))
    return imgs[0] if imgs else None

db = read_json(str(DB_PATH), default={"teachers": []})
teachers = db.get("teachers", [])

df_teachers = pd.DataFrame([
    {
        "ID": t.get("id"),
        "Name": t.get("name"),
        "Face Samples": len(t.get("face_embeddings", [])),
        "Voice Samples": len(t.get("voice_embeddings", []))
    }
    for t in teachers
])

df_logs = load_jsonl(LOGS_PATH)

h1, h2 = st.columns([1, 6])
with h1:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=130)
with h2:
    st.markdown("## Mutah University")
    st.markdown("### Faculty of Information Technology")
    st.markdown("#### Face & Voice Teacher Authentication System")

st.divider()

tab1, tab2, tab3 = st.tabs(["Dashboard", "Educational Analytics", "Admin Panel"])

with tab1:
    selected_name = st.selectbox("Select Doctor", df_teachers["Name"].tolist())
    row = df_teachers[df_teachers["Name"] == selected_name].iloc[0]
    tid = row["ID"]

    c1, c2 = st.columns([1, 2])

    with c1:
        img = get_teacher_image(tid)
        if img:
            st.image(Image.open(img), width=260)

    with c2:
        today = datetime.now().date().isoformat()
        if not df_logs.empty:
            today_entries = df_logs[
                (df_logs["final_id"] == tid) &
                (df_logs["decision"] == "ACCEPT") &
                (df_logs["ts"].astype(str).str.startswith(today))
            ]
            count_today = len(today_entries)
        else:
            count_today = 0

        m1, m2, m3 = st.columns(3)
        m1.metric("Doctor ID", tid)
        m2.metric("Entries Today", count_today)
        m3.metric("Total Samples", int(row["Face Samples"] + row["Voice Samples"]))

    st.plotly_chart(
        px.pie(df_teachers, values="Face Samples", names="Name", title="Face Data Distribution"),
        use_container_width=True
    )

    st.plotly_chart(
        px.pie(df_teachers, values="Voice Samples", names="Name", title="Voice Data Distribution"),
        use_container_width=True
    )

with tab2:

    df_logs = load_jsonl(LOGS_PATH)

    if df_logs.empty:
        st.info("No recognition logs available yet.")
        st.stop()

    c1, c2, c3 = st.columns(3)
    with c1:
        mode = st.selectbox("Recognition Mode", ["fusion", "face", "voice"])
    with c2:
        decision_filter = st.multiselect(
            "Decision",
            ["ACCEPT", "REJECT"],
            default=["ACCEPT", "REJECT"]
        )
    with c3:
        auto_refresh = st.checkbox("Auto refresh", value=True)

    if auto_refresh and st_autorefresh is not None:
        st_autorefresh(interval=4000, key="analytics_refresh")

    df = df_logs.copy()

    df = df[df["type"] == mode]
    df = df[df["decision"].isin(decision_filter)]

    if df.empty:
        st.warning("No data for selected filters.")
        st.stop()

    scores = df["fused_score"].fillna(0)
    y_true = (df["decision"] == "ACCEPT").astype(int)

    fpr, tpr, thresholds = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)

    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, name=f"AUC = {roc_auc:.3f}"))
    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(dash="dash")))
    fig_roc.update_layout(title="ROC Curve", xaxis_title="FPR", yaxis_title="TPR")
    st.plotly_chart(fig_roc, use_container_width=True)

    best_idx = (tpr - fpr).argmax()
    thr = thresholds[best_idx]
    y_pred = (scores >= thr).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    st.plotly_chart(px.imshow(cm, text_auto=True, title="Confusion Matrix"), use_container_width=True)

    metrics_df = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1", "AUC"],
        "Value": [
            accuracy_score(y_true, y_pred),
            precision_score(y_true, y_pred),
            recall_score(y_true, y_pred),
            f1_score(y_true, y_pred),
            roc_auc
        ]
    })

    st.table(metrics_df)

    skf = StratifiedKFold(n_splits=5)
    rows = []

    for i, (_, test_idx) in enumerate(skf.split(scores, y_true)):
        yt = y_true.iloc[test_idx]
        yp = (scores.iloc[test_idx] >= thr).astype(int)
        rows.append([
            accuracy_score(yt, yp),
            precision_score(yt, yp),
            recall_score(yt, yp),
            f1_score(yt, yp)
        ])

    fold_df = pd.DataFrame(
        rows,
        columns=["Accuracy", "Precision", "Recall", "F1"],
        index=[f"Fold {i+1}" for i in range(len(rows))]
    )

    st.plotly_chart(
        px.imshow(
            fold_df,
            text_auto=".2f",
            color_continuous_scale="Turbo",
            title="K-Fold Performance Heatmap"
        ),
        use_container_width=True
    )

with tab3:

    admin_live, admin_pending = st.tabs(["Live Enrollment", "Pending Approvals"])

    with admin_live:
            cam = st.camera_input("Live Camera Capture")

            if "show_add_form" not in st.session_state:
                st.session_state.show_add_form = False

            if cam:
                st.image(cam, width=260)

                c1, c2 = st.columns(2)
                with c1:
                    if st.button("Add Person"):
                        st.session_state.show_add_form = True
                with c2:
                    if st.button("Ignore Request"):
                        st.warning("Request ignored")

                if st.session_state.show_add_form:
                    st.divider()
                    name = st.text_input("Name")
                    number = st.text_input("Number")

                    audio = st.audio_input("Record Voice Sample")

                    if audio:
                        st.audio(audio)

                    if st.button("Confirm Add"):
                        st.success(f"Person added: {name} | {number}")
                        st.session_state.show_add_form = False

            st.divider()
            st.table(df_teachers)

    with admin_pending:
            st.subheader('Pending approvals (Admin)')

            server_base = st.text_input('AI Server URL', value=os.getenv('SERVER_URL', 'http://127.0.0.1:8000'), key='srv_base_tab3')
            if server_base.endswith('/'):
                server_base = server_base[:-1]

            cA, cB, cC = st.columns([1.3, 1.3, 2.4])
            with cA:
                enable_notify = st.checkbox('Show toast on new pending', value=True, key='notify_pending')
            with cB:
                enable_autorefresh = st.checkbox('Auto-refresh', value=True, key='auto_pending')
            with cC:
                auto_refresh_seconds = st.slider('Auto-refresh interval (sec)', min_value=3, max_value=30, value=6, key='auto_pending_sec')

            if enable_autorefresh and st_autorefresh is not None:
                st_autorefresh(interval=auto_refresh_seconds * 1000, key='pending_autorefresh')
            elif enable_autorefresh and st_autorefresh is None:
                st.caption("Auto-refresh requires 'streamlit-autorefresh'. Install: pip install streamlit-autorefresh")

            st.caption('New teachers appear here after a REJECT + registration request. Approve to add them to the database.')

            pending = []
            try:
                r = requests.get(urljoin(server_base + '/', 'api/admin/pending'), timeout=5)
                r.raise_for_status()
                payload = r.json() or {}
                pending = payload.get('pending', []) or []

                if enable_notify:
                    seen = set(st.session_state.get('seen_pending_ids', []))
                    current = [p.get('pending_id') for p in pending if p.get('pending_id')]
                    new_ids = [pid for pid in current if pid not in seen]
                    if new_ids:
                        st.session_state['seen_pending_ids'] = list(seen.union(new_ids))
                        first = next((p for p in pending if p.get('pending_id') == new_ids[0]), {})
                        st.toast(f"New pending request: {first.get('name','Unknown')} ({len(new_ids)} new)", icon='⚠️')
            except Exception as e:
                st.error(f'Cannot reach AI server at {server_base}: {e}')
                pending = []

            if not pending:
                st.info('No pending requests.')
                st.stop()

            st.write(f"**Pending count:** {len(pending)}")

            for item in pending:
                pid = item.get('pending_id')
                if not pid:
                    continue

                name_default = item.get('name') or ''
                created_at = item.get('created_at', '-')
                source = "NAO" if item.get("robot_captured") else "PC"

                with st.expander(f"{name_default or 'Unknown'}  —  {pid}", expanded=False):
                    left, right = st.columns([2, 3])

                    with left:
                        st.markdown(f'**Pending ID:** `{pid}`')
                        st.markdown(f'**Captured at:** {created_at}')
                        st.markdown(f'**Source:** {source}')

                        try:
                            img = requests.get(urljoin(server_base + '/', f'api/admin/pending/{pid}/image'), timeout=10).content
                            if img:
                                st.image(img, caption='Captured image', use_container_width=True)
                        except Exception:
                            st.warning('No image available')

                        try:
                            aud = requests.get(urljoin(server_base + '/', f'api/admin/pending/{pid}/audio'), timeout=10).content
                            if aud:
                                st.audio(aud, format='audio/wav')
                        except Exception:
                            st.warning('No audio available')

                        try:
                            na = requests.get(urljoin(server_base + '/', f'api/admin/pending/{pid}/name_audio'), timeout=10).content
                            if na:
                                st.caption("Name audio (optional)")
                                st.audio(na, format='audio/wav')
                        except Exception:
                            pass

                    with right:
                        new_name = st.text_input("Name (edit before approve)", value=name_default, key=f"nm_{pid}")

                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button('Approve', key=f'app_{pid}'):
                                try:
                                    rr = requests.post(
                                        urljoin(server_base + '/', 'api/admin/pending/decision'),
                                        json={'pending_id': pid, 'action': 'approve', 'name': new_name},
                                        timeout=10
                                    )
                                    rr.raise_for_status()
                                    st.success(rr.json())
                                    st.rerun()
                                except Exception as e:
                                    st.error(f'Approve failed: {e}')

                        with col2:
                            if st.button('Reject', key=f'rej_{pid}'):
                                try:
                                    rr = requests.post(
                                        urljoin(server_base + '/', 'api/admin/pending/decision'),
                                        json={'pending_id': pid, 'action': 'reject'},
                                        timeout=10
                                    )
                                    rr.raise_for_status()
                                    st.warning(rr.json())
                                    st.rerun()
                                except Exception as e:
                                    st.error(f'Reject failed: {e}')

                        st.caption("Raw pending payload")
                        st.json(item, expanded=False)

                    st.divider()

