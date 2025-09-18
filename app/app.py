"""
app.py (thin UI layer)
Purpose: Streamlit-only glue. Renders widgets/tabs, collects user inputs, and delegates
all work to the controller.
Why: Keeps UI concerns (layout/state widgets) separate from business logic so you
can unit test logic without Streamlit.
What is inside:
- Session state defaults (model, persona, style, etc.).
- Sidebar controls, JD ingest UI, chat transcript rendering.
- One call site to the controller: controller.chat_once(...).
- Token/cost display using services.pricing.
Do nots: No prompt building, no PDF parsing, no LLM calls, no data mutating beyond
st_session.
Testing: Treat as smoke/integration (run app, click through). Put logic elsewhere.
"""

import streamlit as st
from audio_recorder_streamlit import audio_recorder
from datetime import datetime
from typing import Optional
import hashlib

from core.utils.voice import transcribe_wav_bytes, autoplay_html
from core.controller import InterviewSessionController
from core.models import (
    LLMSettings,
    InterviewerRole,
    PracticeMode,
    InterviewerPersona,
    InterviewStyle,
    JobDescription,
)
from core.services.pricing import PRICE_TABLE, estimate_cost
from core.controller_roadmap import RoadmapController


# ---------------------------
# Page config
# ---------------------------
st.set_page_config(
    page_title="Interview Prep App",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)
# ---------------------------
# UI constants
# ---------------------------

INTERVIEWER_ROLES = [
    InterviewerRole.HR.value,
    InterviewerRole.HIRING_MANAGER.value,
    InterviewerRole.PM.value,
]


PRACTICE_MODES = [
    PracticeMode.BEHAVIORAL.value,
    PracticeMode.TECHNICAL.value,
    PracticeMode.SITUATIONAL.value,
]
PERSONAS = [
    InterviewerPersona.FRIENDLY.value,
    InterviewerPersona.NEUTRAL.value,
    InterviewerPersona.STRICT.value,
]
STYLES = [
    InterviewStyle.CONCISE.value,
    InterviewStyle.DETAILED.value,
]
SENIORITY_CHOICES = ["Junior", "Mid", "Senior", "Team Lead"]
# map UI label -> normalized token you use elsewhere

# ---------------------------
# Session state init
# ---------------------------
st_session = st.session_state
if "controller" not in st_session:
    try:
        st_session.controller = InterviewSessionController()
        st.toast("OpenAI client ready!", icon="✅")
    except Exception as e:
        st_session.controller = None
        st.toast(f"OpenAI init failed: {e}")

st_session.setdefault("messages", [])
st_session.setdefault("tokens_in", 0)
st_session.setdefault("tokens_out", 0)
st_session.setdefault("model", "gpt-4o-mini")
st_session.setdefault("style", STYLES[0])
st_session.setdefault("temperature", 0.7)
st_session.setdefault("top_p", 0.9)

st_session.setdefault("interviewer_role", INTERVIEWER_ROLES[0])
st_session.setdefault("interviewer_role_draft", st_session.interviewer_role)
st_session.setdefault("persona", PERSONAS[1])
st_session.setdefault("persona_draft", st_session.persona)
st_session.setdefault("interviewer_set", False)

st_session.setdefault("last_handoff_summary", "")

st_session.setdefault(
    "last_feedback",
    {"scores": None, "summary": "", "improved_answer": "", "next_actions": ""},
)

st_session.setdefault("job_description", None)
st_session.setdefault("jd_locked", False)
st_session.setdefault("practice_mode", PRACTICE_MODES[0])
st_session.setdefault("proposed_role", "")
st_session.setdefault("proposed_seniority", "")
st_session.setdefault("proposed_seniority_draft", "")
st_session.setdefault("jd_parse_meta", {})

st_session.setdefault("session_start_ts", datetime.now().timestamp())

st_session.setdefault("plan_json", None)
st_session.setdefault("plan_generated", False)
st_session.setdefault("infographic", None)

st_session.setdefault("speak_replies", False)
st_session.setdefault("voice_mode", False)
st_session.setdefault("last_voice_sig", None)
st_session.setdefault("tts_queue", [])
st_session.setdefault("tts_text_queue", [])  # list[str]
st_session.setdefault("tts_audio_queue", [])  # list[bytes]


# ---------------------------
# Helpers
# ---------------------------
def get_controller():
    """Return the controller object (or None)."""
    return st_session.get("controller")


def get_ready_controller():
    """Return controller only if it's initialized and ready."""
    controller = get_controller()
    if not controller:
        return None
    try:
        return controller if controller.is_ready() else None
    except Exception:
        return None


def current_history():
    """Safe read of the active chat history regardless of controller shape."""
    controller = get_ready_controller()
    if controller and hasattr(controller, "get_history"):
        try:
            return controller.get_history()
        except Exception:
            pass
    return getattr(controller, "history", None) or st_session.get("messages", [])


def interviewer_role_enum():
    return InterviewerRole(st_session.interviewer_role)


def practice_mode_enum():
    return PracticeMode(st_session.practice_mode)


def interviewer_persona_enum():
    return InterviewerPersona(st_session.persona)


def style_enum():
    return InterviewStyle(st_session.style)


def make_llm_settings(max_tokens: int = 512) -> LLMSettings:
    return LLMSettings(
        model=st_session.model,
        temperature=float(st_session.temperature),
        top_p=float(st_session.top_p),
        max_tokens=max_tokens,
    )


def jd_to_dict(jd) -> dict:
    """Support either dict or dataclass instance."""
    if jd is None:
        return {}
    return getattr(jd, "__dict__", jd)


def reset_session():
    st_session.messages = []
    st_session.tokens_in = 0
    st_session.tokens_out = 0
    st_session.job_description = None
    st_session.jd_locked = False
    st_session.interviewer_set = False
    st_session.interviewer_role = INTERVIEWER_ROLES[0]
    st_session.persona = PERSONAS[1]
    st_session.interviewer_role_draft = st_session.interviewer_role
    st_session.persona_draft = st_session.persona
    st_session.proposed_role = ""
    st_session.proposed_seniority = ""
    st_session.proposed_seniority_draft = ""
    st_session.session_start_ts = datetime.now().timestamp()
    st_session.plan_json = None
    st_session.plan_generated = False
    st_session.infographic = None

    controller = get_controller()
    if controller:
        controller.reset()


def render_handoff_memo(memo) -> None:
    """Pretty UI for a handoff memo object."""
    if not memo:
        return

    role_label = getattr(memo.interviewer_role, "value", memo.interviewer_role)
    persona_label = getattr(memo.interviewer_persona, "value", memo.interviewer_persona)

    created = getattr(memo, "created_at", None)
    header = f"From **{role_label}** ({persona_label})"
    if created:
        header += f" · {created:%Y-%m-%d %H:%M}"

    st.markdown(header)
    st.write(memo.summary)

    cols = st.columns(3)
    sections = [
        ("**Strengths**", getattr(memo, "strengths", []) or []),
        ("**Concerns**", getattr(memo, "concerns", []) or []),
        ("**Recommendations**", getattr(memo, "recommendations", []) or []),
    ]
    for (title, items), col in zip(sections, cols):
        with col:
            st.markdown(title)
            if items:
                st.markdown("\n".join(f"- {it}" for it in items))
            else:
                st.markdown("—")


def on_characteristics_toggle():
    """
    - If unlocked → COMMIT & LOCK (no rerun call here; Streamlit will
    rerun automatically).
    - If locked   → GENERATE HANDOFF MEMO → UNLOCK → clear chat UI
    (no rerun call).
    """
    controller = get_ready_controller()

    if not st_session.interviewer_set:
        # --- COMMIT & LOCK ---
        st_session.interviewer_role = st_session.interviewer_role_draft
        st_session.persona = st_session.persona_draft

        # Persist enums to controller
        if controller and hasattr(controller, "set_interviewer"):
            try:
                controller.set_interviewer(
                    role=interviewer_role_enum(),
                    persona=interviewer_persona_enum(),
                )
            except Exception as e:
                st.toast(f"Could not persist interviewer: {e}")

        # Keep drafts aligned with committed
        st_session.interviewer_role_draft = st_session.interviewer_role
        st_session.persona_draft = st_session.persona

        st_session.interviewer_set = True
        st.toast("Interviewer locked.")

        return

    # --- currently LOCKED → GENERATE HANDOFF MEMO, then UNLOCK + CLEAR CHAT ---
    try:
        if controller and hasattr(controller, "generate_handoff"):
            history = current_history()
            has_user_turn = any(m.get("role") == "user" for m in history)
            st_session.last_handoff = (
                controller.generate_handoff() if has_user_turn else None
            )
        else:
            st_session.last_handoff = None
    except Exception as e:
        st_session.last_handoff = None
        st.toast(f"Handoff failed: {e}")

    reset_feedback_ui()

    st_session.interviewer_set = False
    st_session.interviewer_role_draft = st_session.interviewer_role
    st_session.persona_draft = st_session.persona

    # Clear the transcript so the next interviewer starts fresh
    clear_chat_ui()
    st.toast("Interviewer unlocked. You can adjust and press “Set Interviewer”.")
    # DO NOT call st.rerun() here either.


def clear_chat_ui():
    """Wipe the visible transcript and controller history."""
    st_session.messages = []
    controller = get_ready_controller()
    if controller:
        try:
            # preferred: a clear method if you add one later
            controller.state.history.clear()
        except Exception:
            pass


def start_interview():
    """
    When interviewer is set:
      - If no history → generate greeting+intro request (one assistant turn).
      - If last turn is user (they introduced themselves) → ask first real question.
      - If last turn is assistant → do nothing (waiting for user reply).
    """
    if not st_session.interviewer_set:
        return

    controller = get_ready_controller()
    if not controller:
        return

    history = current_history()
    if not history:
        reply, meta = controller.greet_and_open(
            settings=make_llm_settings(160),
            role=st_session.proposed_role,
            seniority=st_session.proposed_seniority,
            interviewer_role=interviewer_role_enum(),
            persona=interviewer_persona_enum(),
            style=style_enum(),
            mode=practice_mode_enum(),
        )
        return


# ── FEEDBACK HELPERS ───────────────────────────────────────────────────────────
def _extract_last_qa(history: list[dict[str, str]]) -> tuple[str, str]:
    """
    Return (last_question_from_assistant, last_user_answer) from the history.
    If not present, returns ("", "").
    """
    last_user = ""
    last_assistant = ""
    # Walk from end to start; pick the most recent user and assistant turns
    for m in reversed(history or []):
        role = m.get("role")
        if not last_user and role == "user":
            last_user = (m.get("content") or "").strip()
        elif not last_assistant and role == "assistant":
            last_assistant = (m.get("content") or "").strip()
        if last_user and last_assistant:
            break

    if last_assistant and not last_assistant.endswith("?"):
        for m in reversed(history or []):
            if m.get("role") == "assistant":
                txt = (m.get("content") or "").strip()
                if txt.endswith("?"):
                    last_assistant = txt
                    break
    return last_assistant, last_user


def _render_rubric(scores: dict[str, float] | None):
    """
    scores: dict with 0..1 floats. Keys:
      - answer_quality
      - star_structure
      - technical_depth
      - communication
    If None, renders placeholders.
    """
    # Default placeholders
    if not scores:
        scores = {
            "answer_quality": None,
            "star_structure": None,
            "technical_depth": None,
            "communication": None,
        }

    labels = [
        ("answer_quality", "Answer quality"),
        ("star_structure", "STAR structure"),
        ("technical_depth", "Technical depth"),
        ("communication", "Communication"),
    ]
    cols = st.columns(4)
    for (key, label), col in zip(labels, cols):
        with col:
            val = scores.get(key)
            if val is None:
                st.metric(label, "—")
            else:
                pct = int(round(val * 100))
                st.metric(label, f"{pct}%")
                st.progress(val)


def reset_feedback_ui():
    """Clear rubric + coaching outputs in the Feedback tab."""
    st_session.setdefault("last_feedback", {})
    st_session.last_feedback = {
        "scores": None,  # dict[str, float] in 0..1, or None
        "summary": "",  # short textual feedback
        "improved_answer": "",  # LLM rewrite
        "next_actions": "",  # bullets
    }


def _format_duration(seconds: float) -> str:
    seconds = int(max(0, seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m}m {s:02d}s"


def show_infographic(img_payload):
    kind = img_payload.get("kind")
    data = img_payload.get("data")

    if kind == "url":
        if not data:
            st.error("Image URL is empty.")
            return
        st.image(data, use_container_width=True)  # URL string
        return

    if kind == "bytes":
        if not data:
            st.error("Image bytes are empty.")
            return
        fmt = img_payload.get("format", "PNG")
        st.image(
            data, use_container_width=True, output_format=fmt
        )  # <- avoids format sniffing
        return

    st.error("No image received from generator.")


# ---------------------------
# SIDEBAR: settings & JD ingest
# ---------------------------
with st.sidebar:
    st.markdown("# Settings")

    # Model
    st_session.model = st.selectbox(
        "Model",
        list(PRICE_TABLE.keys()),
        index=list(PRICE_TABLE.keys()).index(st_session.model),
    )
    st.divider()

    # ----------------- JD ingest: paste or PDF -----------------
    st.markdown("## Job Description")

    disabled_jd = st_session.jd_locked

    jd_mode = st.radio(
        "Input mode",
        ["Upload PDF", "Paste text"],
        horizontal=True,
        disabled=disabled_jd,
    )

    jd_text: Optional[str] = None
    uploaded_pdf = None
    if jd_mode == "Paste text":
        jd_text = st.text_area(
            "Paste JD text",
            placeholder=(
                "Paste the full job description here or "
                + "at least seniority level and role title..."
            ),
            height=220,
            disabled=disabled_jd,
        )
    else:
        uploaded_pdf = st.file_uploader(
            "Upload PDF", type=["pdf"], disabled=disabled_jd
        )

    if not st_session.jd_locked and st.button(
        "Analyze", type="primary", disabled=disabled_jd
    ):
        try:
            controller = get_ready_controller()
            if jd_mode == "Paste text" and jd_text and jd_text.strip():
                job, meta = controller.process_text_input(jd_text)
            elif jd_mode == "Upload PDF" and uploaded_pdf:
                job, meta = controller.process_pdf(uploaded_pdf)
                if job is None:
                    st.toast("No selectable text found in the PDF.")
                    job = None
            else:
                job = None
                raise ValueError("No input provided.")

            if job:
                st_session.job_description = job

                # Proposed role/seniority from LLM
                st_session.proposed_role = (job.get("title") or "").strip()
                st_session.proposed_seniority = (
                    job.get("meta", {}).get("seniority") or ""
                ).strip()
                st_session.jd_parse_meta = job.get("meta", {})

                st.toast(
                    "Job details analyzed. Please confirm role & seniority.", icon="✅"
                )
        except ValueError as e:
            st.error(str(e))

    # After analysis (or manual), ask for confirmation (unlocked)
    if not st_session.jd_locked:
        st.markdown("**Confirm role & seniority**")

        # ROLE is read-only
        st.text_input(
            "Role",
            value=st_session.proposed_role or "—",
            key="proposed_role_view",
            disabled=True,
        )

        # SENIORITY: if parsed -> show read-only; else let the user pick
        if st_session.proposed_seniority:
            st.text_input(
                "Seniority",
                value=st_session.proposed_seniority.replace("_", " ").capitalize(),
                key="proposed_seniority_view",
                disabled=True,
            )
        else:
            # let user pick a draft value; do NOT commit until Confirm
            st.selectbox(
                "Seniority",
                options=SENIORITY_CHOICES,
                index=0,
                key="proposed_seniority_draft",
                help="Pick seniority if it wasn't detected from the job description.",
            )

        c1, c2 = st.columns([1, 1])
        initial_seniority = bool(st_session.proposed_seniority)
        has_seniority = bool(
            st_session.proposed_seniority or st_session.proposed_seniority_draft
        )
        confirm_clicked = c1.button(
            "Confirm",
            type="primary",
            disabled=not (st_session.proposed_role and has_seniority),
        )
        clear_clicked = c2.button("Clear")

        if clear_clicked:
            st_session.proposed_role = ""
            st_session.proposed_seniority = ""
            st_session.proposed_seniority_draft = ""
            st_session.job_description = None
            st_session.jd_parse_meta = {}
            st.toast("Cleared. Please provide a job description again.", icon="🧹")

        if confirm_clicked:
            if not initial_seniority:
                seniority_input = (
                    st_session.proposed_seniority
                    or st_session.proposed_seniority_draft
                    or ""
                )

                jd_prev_map = jd_to_dict(st_session.get("job_description"))
                meta_src = st_session.get("jd_parse_meta", {}) or {}

                description = (
                    meta_src.get("raw_text") or jd_prev_map.get("description") or ""
                ).strip()

                st_session.job_description = JobDescription(
                    title=st_session.proposed_role or jd_prev_map.get("title"),
                    company=meta_src.get("company") or jd_prev_map.get("company"),
                    location=meta_src.get("location") or jd_prev_map.get("location"),
                    description=f"{seniority_input.replace('_', ' ').capitalize()} "
                    f"{description}",
                    meta={
                        **(jd_prev_map.get("meta") or {}),
                        "source": "user_confirmed",
                        "seniority": seniority_input,
                    },
                )
                st_session.proposed_seniority = st_session.proposed_seniority_draft

            st_session.jd_locked = True
            st_session.jd_parse_meta = {}

            if controller := get_ready_controller():
                controller.set_job_description(st_session.job_description)

            st.toast("Role & seniority confirmed and locked.", icon="🔒")
            st.rerun()

    # If locked, show a compact preview
    if st_session.jd_locked and st_session.job_description:
        with st.expander("Preview parsed JD"):
            jd_map = jd_to_dict(st_session.job_description)
            st.markdown(f"**Title:** {jd_map.get('title') or '(no title)'}")
            src = (jd_map.get("meta", {}) or {}).get("requirements_source", "specific")
            tag = "specific" if src == "specific" else "general"
            st.caption(f"Requirements source: **{tag}**")
            desc = jd_map.get("description", "")
            st.text(desc[:1000] + ("..." if len(desc) > 1000 else ""))
    st.divider()

    # --------------------- Interviewer role ---------------------
    st.markdown("## Interviewer Role")

    disabled_chars = st_session.interviewer_set
    st.selectbox(
        "Interviewer role",
        INTERVIEWER_ROLES,
        key="interviewer_role_draft",
        disabled=disabled_chars,
    )
    st.selectbox(
        "Interviewer persona", PERSONAS, key="persona_draft", disabled=disabled_chars
    )

    label = (
        "Set Interviewer" if not st_session.interviewer_set else "Change Interviewer"
    )
    st.button(label, type="secondary", on_click=on_characteristics_toggle)
    st.divider()

    st.markdown("## Interview Controls")
    # Practice mode
    st_session.practice_mode = st.selectbox(
        "Practice mode",
        PRACTICE_MODES,
        index=PRACTICE_MODES.index(st_session.practice_mode),
    )
    st_session.style = st.radio(
        "Response style",
        STYLES,
        index=STYLES.index(st_session.style),
        horizontal=True,
    )
    st_session.temperature = st.slider(
        "Temperature", 0.0, 1.0, st_session.temperature, 0.05
    )
    st_session.top_p = st.slider("Top-p", 0.0, 1.0, st_session.top_p, 0.05)
    # Reset everything
    st.markdown("## Session Controls")
    st.write("Reset all session data and start fresh.")
    st.button("Reset session", type="primary", on_click=reset_session)

# ---------------------------
# Header
# ---------------------------
st.title("Interview Prep App")
if st_session.jd_locked:
    st.caption(
        f" · Role focus: **{st_session.proposed_role}** **("
        f"{st_session.proposed_seniority.replace('_', ' ').capitalize()})**"
    )

# ---------------------------
# Main tabs
# ---------------------------
st.html(
    """
<style>
.stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
    font-size: 18px;
}
</style>
"""
)
(
    about_tab,
    practice_tab,
    feedback_tab,
    plan_tab,
    pricing_tab,
) = st.tabs(["About", "Practice", "Feedback", "Plan", "Pricing"])

with about_tab:
    st.subheader("About this app")
    st.markdown(
        """
        This app is designed to help you prepare for interviews by simulating
         real interview scenarios.
        Some rules:
        - if you do not wish to add job description, then add into a text field: 
        "Junior Data Scientist" or any other role and seniority.
        - seniority levels supported: Junior, Mid-level, Senior, Team Lead.
        - If you have a need for to support other seniority levels, please
        contact me at ...
        - longer than 15000 characters job descriptions will be truncated.
        - Seniority supported: "Junior", "Mid", "Senior", "Team Lead"
        - 7 day prep plan
        """
    )

# --- Practice tab (scrollable transcript + input below) ---
with practice_tab:
    if not st_session.jd_locked:
        st.info("Please add and confirm a Job Description in the sidebar.")
    elif not st_session.interviewer_set:
        st.warning("Please set the Interviewer role and persona in the sidebar.")
    else:
        st.caption(
            f"Interviewer: **{interviewer_role_enum().value}** "
            f"**({interviewer_persona_enum().value})**"
        )
        st.caption(f"Interview mode: **{practice_mode_enum().value}**")

        start_interview()  # will only greet once, see earlier guard we added

        # Voice toggles
        vcol1, vcol2 = st.columns([1, 1])
        with vcol1:
            st_session.voice_mode = st.toggle(
                "🎙️ Voice mode", value=st_session.voice_mode
            )
        with vcol2:
            st_session.speak_replies = st.toggle(
                "🔊 Speak assistant replies", value=st_session.speak_replies
            )

        if st_session.tts_audio_queue:
            mp3 = st_session.tts_audio_queue.pop(0)
            st.html(autoplay_html(mp3))  # optional
            # st.audio(mp3, format="audio/mp3")

        # 3b) If we have text queued and speaking is enabled, synthesize exactly one
        controller = get_ready_controller()
        if controller and st_session.speak_replies and st_session.tts_text_queue:
            next_text = st_session.tts_text_queue.pop(0)
            with st.spinner("Preparing audio…"):
                try:
                    audio_bytes, _tts_meta = controller.speak(
                        next_text, voice="alloy", tts_model="gpt-4o-mini-tts"
                    )
                    if audio_bytes:
                        st_session.tts_audio_queue.append(audio_bytes)
                        # trigger next run to actually play it
                        st.rerun()
                except Exception as e:
                    st.toast(f"TTS failed: {e}", icon="⚠️")

        # Transcript (only render messages here)
        transcript = st.container(height=500, border=True)
        with transcript:
            for msg in current_history():
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

        # ------------- Unified input handler -------------
        user_text = None
        submitted = False

        if st_session.voice_mode:
            st.markdown("**Press to record, then release**")
            wav_bytes = audio_recorder(
                pause_threshold=2,
                sample_rate=16_000,
                text="Hold to talk",
                icon_size="2x",
            )
            if wav_bytes:

                sig = hashlib.sha1(wav_bytes).hexdigest()
                if sig != st_session.get("last_voice_sig"):
                    if controller:
                        with st.spinner("Transcribing…"):
                            user_text = transcribe_wav_bytes(wav_bytes)  # your helper
                        if user_text:
                            st_session.last_voice_sig = sig
                            submitted = True
        else:
            raw = st.chat_input("Type your answer…")
            if raw is not None and raw.strip():
                user_text = raw.strip()
                submitted = True

        if submitted and not user_text:
            st.toast("Please enter a non-empty message.", icon="⚠️")
        # Only process a turn if we actually have user_text
        if controller and user_text:
            try:
                controller.append_user(user_text)
                reply, meta = controller.chat_once(
                    settings=make_llm_settings(512),
                    role=st_session.proposed_role,
                    persona=interviewer_persona_enum(),
                    style=style_enum(),
                    seniority=st_session.proposed_seniority,
                    mode=practice_mode_enum(),
                    interviewer_role=interviewer_role_enum(),
                    user_text=user_text,
                )

                # ---- Speak assistant reply (optional) ----
                if st_session.speak_replies and reply.strip():
                    st_session.tts_text_queue.append(reply)

            except Exception as e:
                st.toast(f"Chat flow failed: {e}", icon="⚠️")

            st.rerun()


# --- Feedback tab (placeholder) ---
with feedback_tab:
    st.subheader("Feedback")
    st.caption(
        "Score your latest answer, preview an improved version, and get next actions."
    )

    # --- Pull last exchange (question + your answer) ---
    history = current_history()
    last_q, last_a = _extract_last_qa(history)

    qcol, acol = st.columns([1, 1])
    with qcol:
        st.markdown("**Last question**")
        st.text_area(
            "last_question",
            value=last_q or "—",
            height=110,
            disabled=True,
            label_visibility="collapsed",
        )
    with acol:
        st.markdown("**Your answer**")
        st.text_area(
            "your_answer",
            value=last_a or "—",
            height=110,
            disabled=True,
            label_visibility="collapsed",
        )

    st.divider()

    # --- Actions row ---
    act1, act2, act3 = st.columns([1, 1, 4])
    score_clicked = act1.button(
        "Score last answer", type="primary", disabled=not last_a
    )
    improve_clicked = act2.button("Suggest improved answer", disabled=not last_a)
    with act3:
        st.caption("Tip: Answer a question in Practice first, then score it here.")

    # Ensure container exists
    st_session.setdefault(
        "last_feedback",
        {"scores": None, "summary": "", "improved_answer": "", "next_actions": ""},
    )

    # --- Handle clicks FIRST (so results show in the same run) ---
    controller = get_ready_controller()

    # --- Handle clicks FIRST (so results show immediately) ---
    if controller and score_clicked:
        try:
            res, meta = controller.score_last_answer(
                settings=make_llm_settings(240),
                role=st_session.proposed_role,
                seniority=st_session.proposed_seniority,
                mode=practice_mode_enum(),
                interviewer_role=interviewer_role_enum(),
                persona=interviewer_persona_enum(),
                style=style_enum(),
            )
            # write into the single source-of-truth container
            st_session.last_feedback["scores"] = res.get("scores")
            st_session.last_feedback["summary"] = res.get("summary", "")
        except Exception as e:
            st.toast(str(e))

    if controller and improve_clicked:
        try:
            res, meta = controller.improve_last_answer(  # <-- correct function
                settings=make_llm_settings(380),
                role=st_session.proposed_role,
                seniority=st_session.proposed_seniority,
                mode=practice_mode_enum(),
                interviewer_role=interviewer_role_enum(),
                persona=interviewer_persona_enum(),
                style=style_enum(),
                rescore=False,  # set True if you also want scores for improved answer
            )
            st_session.last_feedback["improved_answer"] = res.get("improved_answer", "")
            st_session.last_feedback["next_actions"] = res.get("next_actions", "")
            # Optionally include scores if you called with rescore=True:
            if "scores" in res:
                st_session.last_feedback["scores"] = res["scores"]
        except Exception as e:
            st.toast(str(e))

    # --- Render with UPDATED state (no second click needed) ---
    st.markdown("#### Rubric")
    _render_rubric(
        st_session.last_feedback.get("scores")
    )  # keep your existing renderer

    if (st_session.last_feedback.get("summary") or "").strip():
        st.caption(st_session.last_feedback["summary"])

    st.markdown("#### Improved answer (preview)")
    st.text_area(
        "improved_answer_preview",
        value=st_session.last_feedback.get("improved_answer", ""),
        height=160,
        placeholder="(Not generated yet)",
        key="improved_answer_preview",
    )

    st.markdown("#### Next actions")
    st.text_area(
        "next_actions_preview",
        value=st_session.last_feedback.get("next_actions", ""),
        height=100,
        placeholder="(Not generated yet)",
        key="next_actions_preview",
    )

    # --- Handoff memos below ---
    controller = get_ready_controller()
    st.html(
        """
    <style>
    ul { margin-top: 0.25rem; }
    </style>
    """
    )
    if controller and controller.state.handoffs:
        st.divider()
        st.subheader("Handoff history")
        for i, m in enumerate(reversed(controller.state.handoffs), 1):
            with st.expander(
                f"Memo #{i} — {getattr(m.interviewer_role, 'value', m.interviewer_role)}"
            ):
                render_handoff_memo(m)


with plan_tab:
    st.subheader("Interview Prep Plan")
    st.caption(
        "Generate a tailored 7-day prep plan based on the job description if provided. "
        "Scroll down to see detailed plan."
    )
    if "plan_generated" not in st.session_state:
        st_session.plan_generated = False

    if st.button("Generate 7-day plan", disabled=st.session_state.plan_generated):
        controller = get_ready_controller()
        if controller:
            jd_map = jd_to_dict(st_session.job_description)
            requirements = (jd_map.get("meta", {}) or {}).get("requirements", [])

            plan, meta = controller.generate_prep_plan(
                role=st_session.proposed_role,
                seniority=st_session.proposed_seniority,
                requirements=requirements,  # can be list[str] or a long string
                model=st_session.model,
            )
            st_session.plan_json = plan
            st_session.plan_generated = True

        roadmap_ctrl = RoadmapController()
        if roadmap_ctrl:
            img_payload, meta = roadmap_ctrl.generate_infographic(st_session.plan_json)
            st_session.infographic = img_payload
            st.rerun()

    if st_session.get("infographic"):
        st.divider()
        show_infographic(st_session.infographic)

    if st_session.get("plan_json"):
        st.divider()
        plan = st_session["plan_json"]
        st.markdown(f'## Detailed {plan.get("title", "7-Day Interview Prep Plan")}')
        st.metric(
            "Daily time (min)", plan.get("overview", {}).get("daily_time_minutes", 90)
        )
        for d in plan.get("days", []):
            with st.expander(f'{d.get("day", "Day ?")} — {d.get("theme", "")}'):
                st.markdown("**Goals**")
                for g in d.get("goals", []):
                    st.markdown(f"- {g}")
                for t in d.get("tasks", []):
                    st.markdown(
                        f'**{t.get("label", "Task")}** ({t.get("minutes", 0)} min)'
                    )
                    for it in t.get("items", []):
                        st.markdown(f"- {it}")
                if d.get("deliverables"):
                    st.markdown("**Deliverables**")
                    for x in d["deliverables"]:
                        st.markdown(f"- {x}")


with pricing_tab:
    st.subheader("Insights & Cost")
    st.caption("Live usage and cost estimate.")

    controller = get_ready_controller()
    history = current_history()

    # Session time
    now_ts = datetime.now().timestamp()
    session_secs = now_ts - float(st_session.get("session_start_ts", now_ts))

    # Turn counts
    user_turns = sum(1 for m in (history or []) if m.get("role") == "user")
    interviewer_turns = sum(1 for m in (history or []) if m.get("role") == "assistant")

    # Tokens & cost (prefer controller’s totals)
    tokens_in = getattr(controller, "tokens_in", 0) if controller else 0
    tokens_out = getattr(controller, "tokens_out", 0) if controller else 0
    model_used = getattr(controller, "model_used", None) or st_session.model
    est_cost = estimate_cost(model_used, tokens_in, tokens_out)

    # Top row: time & turns
    c1, c2, _ = st.columns([1, 1, 1])
    with c1:
        st.metric("Tokens (in)", f"{tokens_in:,}")

    with c2:
        st.metric("Tokens (out)", f"{tokens_out:,}")

    c3, c4, _ = st.columns([1, 1, 1])
    with c3:
        st.metric("Estimated cost based on last used model", f"${est_cost:,.4f}")
    with c4:
        st.metric("Last used model", model_used)

    c4, _, _ = st.columns([1, 1, 1])
    with c4:
        st.metric("Session time", _format_duration(session_secs))

st.divider()
st.caption(
    "Privacy tip: Do not  paste sensitive personal data. This is a prototype scaffold."
)
