"""
UI layer
Purpose: Streamlit-only glue. Renders widgets/tabs, collects user inputs, and delegates
all work to the controller. Keeps UI concerns (layout/state widgets) separate from
business logic so logic can be unit tested without Streamlit.
"""

import streamlit as st
from audio_recorder_streamlit import audio_recorder
from datetime import datetime
from typing import Optional
import hashlib

from core.services.llm_openai import OpenAILLMClient
from core.services.voice import autoplay_html
from core.controller import InterviewSessionController
from core.models import (
    LLMSettings,
    InterviewerRole,
    PracticeMode,
    InterviewerPersona,
    InterviewStyle,
    JobDescription,
    HandoffMemo,
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

# ---------------------------
# Session state init
# ---------------------------
st_session = st.session_state
st_session.setdefault("controller", None)
st_session.setdefault("roadmap_controller", None)
st_session.setdefault("api_key_set", False)
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
st_session.setdefault(
    "last_feedback",
    {"scores": None, "summary": "", "improved_answer": "", "next_actions": ""},
)
st_session.setdefault("job_description", None)
st_session.setdefault("jd_locked", False)
st_session.setdefault("jd_parse_meta", {})
st_session.setdefault("jd_text_input", "")
st_session.setdefault("jd_upload_key", 0)
st_session.setdefault("jd_clear_request", False)
st_session.setdefault("practice_mode", PRACTICE_MODES[0])
st_session.setdefault("proposed_role", "")
st_session.setdefault("proposed_seniority", "")
st_session.setdefault("proposed_seniority_draft", "")
st_session.setdefault("session_start_ts", datetime.now().timestamp())
st_session.setdefault("plan_json", None)
st_session.setdefault("plan_generated", False)
st_session.setdefault("infographic", None)
st_session.setdefault("speak_replies", False)
st_session.setdefault("voice_mode", False)
st_session.setdefault("last_voice_sig", None)
st_session.setdefault("tts_queue", [])
st_session.setdefault("tts_text_queue", [])
st_session.setdefault("tts_audio_queue", [])


# ---------------------------
# Helpers
# ---------------------------
def get_controller():
    """Return the controller object."""
    return st_session.get("controller")


def get_roadmap_controller():
    """Return the roadmap controller object."""
    return st_session.get("roadmap_controller")


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
    """Return the committed interviewer role as enum."""
    return InterviewerRole(st_session.interviewer_role)


def practice_mode_enum():
    """Return the committed practice mode as enum."""
    return PracticeMode(st_session.practice_mode)


def interviewer_persona_enum():
    """Return the committed interviewer persona as enum."""
    return InterviewerPersona(st_session.persona)


def style_enum():
    """Return the committed interview style as enum."""
    return InterviewStyle(st_session.style)


def make_llm_settings(max_tokens: int = 512) -> LLMSettings:
    """Build LLMSettings from session state."""
    return LLMSettings(
        model=st_session.model,
        temperature=float(st_session.temperature),
        top_p=float(st_session.top_p),
        max_tokens=max_tokens,
    )


def jd_to_dict(jd: object) -> dict:
    """Support either dict or dataclass instance."""
    if jd is None:
        return {}
    return getattr(jd, "__dict__", jd)


def reset_session():
    """Wipe all session state except API key and model choice."""

    st_session.messages = []
    st_session.tokens_in = 0
    st_session.tokens_out = 0
    st_session.style = STYLES[0]
    st_session.temperature = 0.7
    st_session.top_p = 0.9
    st_session.job_description = None
    st_session.jd_locked = False
    st_session.jd_parse_meta = {}
    st_session.jd_text_input = ""
    st_session.jd_upload_key += 1
    st_session.interviewer_set = False
    st_session.interviewer_role = INTERVIEWER_ROLES[0]
    st_session.persona = PERSONAS[1]
    st_session.interviewer_role_draft = st_session.interviewer_role
    st_session.persona_draft = st_session.persona
    st_session.practice_mode = PRACTICE_MODES[0]
    st_session.proposed_role = ""
    st_session.proposed_seniority = ""
    st_session.proposed_seniority_draft = ""
    st_session.session_start_ts = datetime.now().timestamp()
    st_session.last_feedback = {
        "scores": None,
        "summary": "",
        "improved_answer": "",
        "next_actions": "",
    }
    st_session.plan_json = None
    st_session.plan_generated = False
    st_session.infographic = None

    st_session.setdefault("speak_replies", False)
    st_session.setdefault("voice_mode", False)
    st_session.setdefault("last_voice_sig", None)
    st_session.setdefault("tts_queue", [])
    st_session.setdefault("tts_text_queue", [])
    st_session.setdefault("tts_audio_queue", [])

    controller = get_controller()
    if controller:
        controller.reset()

    roadmap_controller = get_roadmap_controller()
    if roadmap_controller:
        roadmap_controller.reset()


def render_handoff_memo(memo: HandoffMemo) -> None:
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
    """Set or change interviewer role/persona."""
    controller = get_ready_controller()

    if not st_session.interviewer_set:
        st_session.interviewer_role = st_session.interviewer_role_draft
        st_session.persona = st_session.persona_draft

        if controller and hasattr(controller, "set_interviewer"):
            try:
                controller.set_interviewer(
                    role=interviewer_role_enum(),
                    persona=interviewer_persona_enum(),
                )
            except Exception as e:
                st.toast(f"Could not persist interviewer: {e}")

        st_session.interviewer_role_draft = st_session.interviewer_role
        st_session.persona_draft = st_session.persona
        st_session.interviewer_set = True
        st.toast("Interviewer locked.")
        return

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

    clear_chat_ui()
    st.toast("Interviewer unlocked. You can adjust and press “Set Interviewer”.")


def clear_chat_ui():
    """Wipe the visible transcript and controller history."""
    st_session.messages = []
    controller = get_ready_controller()
    if controller:
        try:
            controller.state.history.clear()
        except Exception:
            pass


def start_interview():
    """If not started yet, greet the user and ask the first question."""
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


def extract_last_qa_old(history: list[dict[str, str]]) -> tuple[str, str]:
    """
    Return (last_question_from_assistant, last_user_answer) from the history.
    If not present, returns ("", "").
    """
    last_user = ""
    last_assistant = ""

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


def extract_last_qa(history: list[dict[str, str]]) -> tuple[str, str]:
    """
    Return (previous_assistant_question, last_user_answer) from the history.
    If not present, returns ("", "").
    """
    last_user = ""
    prev_assistant = ""

    for idx in range(len(history) - 1, -1, -1):
        m = history[idx]
        if m.get("role") == "user":
            last_user = (m.get("content") or "").strip()
            for j in range(idx - 1, -1, -1):
                if history[j].get("role") == "assistant":
                    prev_assistant = (history[j].get("content") or "").strip()
                    break
            break

    return prev_assistant, last_user


def render_rubric(scores: dict[str, float] | None):
    """Render the 4-part rubric with scores and progress bars."""
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
    """Clear rubric and coaching outputs in the Feedback tab."""
    st_session.setdefault("last_feedback", {})
    st_session.last_feedback = {
        "scores": None,
        "summary": "",
        "improved_answer": "",
        "next_actions": "",
    }


def format_duration(seconds: float) -> str:
    """Format seconds as Hh Mm Ss, skipping hours if zero."""
    seconds = int(max(0, seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m}m {s:02d}s"


def show_infographic(img_payload):
    """Render an infographic image from URL or bytes payload."""
    kind = img_payload.get("kind")
    data = img_payload.get("data")

    if kind == "url":
        if not data:
            st.error("Image URL is empty.")
            return
        st.image(data, width=600)
        return

    if kind == "bytes":
        if not data:
            st.error("Image bytes are empty.")
            return
        fmt = img_payload.get("format", "PNG")
        st.image(data, output_format=fmt, width=600)
        return

    st.error("No image received from generator.")


# ---------------------------
# SIDEBAR: settings & JD ingest
# ---------------------------
with st.sidebar:
    st.markdown("# Settings")

    st.markdown("## OPEN AI API Key Required")
    user_api_key = st.sidebar.text_input(
        "Enter your API key",
        type="password",
        help="We do not store your key. It stays in your session only.",
    )
    if not user_api_key:
        st.warning("Please enter your API key in the sidebar to continue.")
        st.stop()
    else:
        if not st_session.api_key_set:
            try:
                llm = OpenAILLMClient(api_key=user_api_key)
                llm.client.models.list()
            except Exception as e:
                st.error(f"OpenAI client init failed: {e}")
                st.stop()
            try:
                st_session.controller = InterviewSessionController(llm)
                st_session.roadmap_controller = RoadmapController(llm)
                st_session.api_key_set = True
            except Exception as e:
                st_session.controller = None
                st.toast(f"OpenAI init failed: {e}")
                st.stop()

    st_session.model = st.selectbox(
        "Model",
        list(PRICE_TABLE.keys()),
        index=list(PRICE_TABLE.keys()).index(st_session.model),
    )
    st.divider()

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
            key="jd_text_input",
            placeholder=(
                "Paste the full job description here or "
                + "at least seniority level and role title..."
            ),
            height=220,
            disabled=disabled_jd,
        )
    else:
        uploaded_pdf = st.file_uploader(
            "Upload PDF",
            key=f"jd_uploader_{st_session.jd_upload_key}",
            type=["pdf"],
            disabled=disabled_jd,
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

    if not st_session.jd_locked:
        st.markdown("**Confirm role & seniority**")
        st.text_input(
            "Role",
            value=st_session.proposed_role or "—",
            key="proposed_role_view",
            disabled=True,
        )
        if st_session.proposed_seniority:
            st.text_input(
                "Seniority",
                value=st_session.proposed_seniority.replace("_", " ").capitalize(),
                key="proposed_seniority_view",
                disabled=True,
            )
        else:
            st.selectbox(
                "Seniority",
                options=SENIORITY_CHOICES,
                index=0,
                key="proposed_seniority_draft",
                help="Pick seniority if it wasn't detected from the job description.",
            )

        c1, c2 = st.columns([1, 1])

        initial_seniority = bool(st_session.proposed_seniority)
        has_role = bool(st_session.proposed_role)
        has_seniority = bool(
            st_session.proposed_seniority or st_session.proposed_seniority_draft
        )
        confirm_disabled = not (has_role and has_seniority)
        clear_disabled = not (has_role and has_seniority)

        confirm_clicked = c1.button(
            "Confirm", type="primary", disabled=confirm_disabled
        )
        clear_clicked = c2.button("Clear", disabled=clear_disabled)

        if clear_clicked:
            st_session.proposed_role = ""
            st_session.proposed_seniority = ""
            st_session.proposed_seniority_draft = ""
            st_session.job_description = None
            st_session.jd_parse_meta = {}
            st.toast("Cleared. Please provide a job description again.", icon="🧹")
            st.rerun()

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
        Welcome to the Interview Prep App!
        This tool is designed to help you prepare for job interviews by simulating
        realistic interview scenarios and providing personalized feedback, 
        as well as the preparation plan.

        What to expect:

        - Practice answering behavioral, technical, and situational interview questions with an AI interviewer. 
        - Customize your session by selecting the interviewer’s role, persona, and interview style.
        - Upload a job description (PDF or text) or simply specify your target role and seniority level.
        - Receive a tailored 7-day preparation plan based on your chosen role and requirements.
        - Get instant feedback on your answers, including scoring, improvement suggestions, and next steps.
        - Optionally use voice mode for spoken answers and listen to AI responses.
        """
    )
    st.markdown(
        """
        All sessions are private—your data and API key remain secure in your browser session.
        Supported seniority levels: Junior, Mid, Senior, Team Lead
        Note: Job descriptions longer than 15,000 characters will be truncated.

        If you need support for other seniority levels or have feedback, please contact the developer.
        """
    )

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

        start_interview()

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
            st.html(autoplay_html(mp3))

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
                        st.rerun()
                except Exception as e:
                    st.toast(f"TTS failed: {e}", icon="⚠️")

        transcript = st.container(height=500, border=True)
        with transcript:
            for msg in current_history():
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

        user_text = None
        submitted = False

        if st_session.voice_mode:
            st.markdown("**Note on voice input:** it does not work on short inputs.")
            wav_bytes = audio_recorder(
                pause_threshold=2,
                sample_rate=16_000,
                text="Press to record",
                icon_size="2x",
            )
            if wav_bytes:

                sig = hashlib.sha1(wav_bytes).hexdigest()
                if sig != st_session.get("last_voice_sig"):
                    if controller:
                        try:
                            with st.spinner("Transcribing…"):
                                user_text, _stt = controller.voice_to_text(wav_bytes)
                        except Exception as e:
                            st.toast(f"Transcription failed: {e}", icon="⚠️")
                            user_text = None
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
                if st_session.speak_replies and reply.strip():
                    st_session.tts_text_queue.append(reply)

            except Exception as e:
                st.toast(f"Chat flow failed: {e}", icon="⚠️")

            st.rerun()

with feedback_tab:
    st.subheader("Feedback")
    st.caption(
        "Score your latest answer, preview an improved version, and get next actions."
    )
    history = current_history()
    last_q, last_a = extract_last_qa(history)

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

    act1, act2, act3 = st.columns([1, 1, 4])
    score_clicked = act1.button(
        "Score last answer", type="primary", disabled=not last_a
    )
    improve_clicked = act2.button("Suggest improved answer", disabled=not last_a)
    with act3:
        st.caption("Tip: Answer a question in Practice first, then score it here.")

    st_session.setdefault(
        "last_feedback",
        {"scores": None, "summary": "", "improved_answer": "", "next_actions": ""},
    )

    controller = get_ready_controller()
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
            st_session.last_feedback["scores"] = res.get("scores")
            st_session.last_feedback["summary"] = res.get("summary", "")
        except Exception as e:
            st.toast(str(e))

    if controller and improve_clicked:
        try:
            res, meta = controller.improve_last_answer(
                settings=make_llm_settings(380),
                role=st_session.proposed_role,
                seniority=st_session.proposed_seniority,
                mode=practice_mode_enum(),
                interviewer_role=interviewer_role_enum(),
                persona=interviewer_persona_enum(),
                style=style_enum(),
                rescore=False,
            )
            st_session.last_feedback["improved_answer"] = res.get("improved_answer", "")
            st_session.last_feedback["next_actions"] = res.get("next_actions", "")
            if "scores" in res:
                st_session.last_feedback["scores"] = res["scores"]
        except Exception as e:
            st.toast(str(e))

    st.markdown("#### Rubric")
    render_rubric(st_session.last_feedback.get("scores"))

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
        st.caption(
            "Previous interviewers' memos used by the current interviewer. "
            "These are generated when you change the interviewer."
        )
        for i, m in enumerate(reversed(controller.state.handoffs), 1):
            with st.expander(
                f"Memo #{i} — {getattr(m.interviewer_role, 'value', m.interviewer_role)}"
            ):
                render_handoff_memo(m)


with plan_tab:
    st.subheader("Interview Prep Plan")
    st.caption(
        "Generate a tailored 7-day prep plan based on the job description if provided. "
        "Scroll down to see a roadmap."
    )
    if "plan_generated" not in st_session:
        st_session.plan_generated = False

    if st.button("Generate 7-day plan", disabled=st_session.plan_generated):
        controller = get_ready_controller()
        if controller:
            jd_map = jd_to_dict(st_session.job_description)
            requirements = (jd_map.get("meta", {}) or {}).get("requirements", [])

            plan, meta = controller.generate_prep_plan(
                role=st_session.proposed_role,
                seniority=st_session.proposed_seniority,
                requirements=requirements,
                model=st_session.model,
            )
            st_session.plan_json = plan
            st_session.plan_generated = True

        roadmap_controller = get_roadmap_controller()
        if roadmap_controller:
            img_payload, meta = roadmap_controller.generate_infographic(
                st_session.plan_json
            )
            st_session.infographic = img_payload
            st.rerun()

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

    if st_session.get("infographic"):
        st.divider()
        show_infographic(st_session.infographic)

with pricing_tab:
    st.subheader("Insights & Cost")
    st.caption("Live usage and cost estimate.")

    controller = get_ready_controller()
    history = current_history()

    now_ts = datetime.now().timestamp()
    session_secs = now_ts - float(st_session.get("session_start_ts", now_ts))

    user_turns = sum(1 for m in (history or []) if m.get("role") == "user")
    interviewer_turns = sum(1 for m in (history or []) if m.get("role") == "assistant")

    tokens_in = getattr(controller, "tokens_in", 0) if controller else 0
    tokens_out = getattr(controller, "tokens_out", 0) if controller else 0
    model_used = getattr(controller, "model_used", None) or st_session.model
    est_cost = estimate_cost(model_used, tokens_in, tokens_out)

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
        st.metric("Session time", format_duration(session_secs))

    if st_session.get("infographic"):
        st.caption("Note: Roadmap generation costs are not included in this estimate.")

st.divider()
st.caption(
    "Privacy tip: Do not  paste sensitive personal data. This is a prototype scaffold."
)
