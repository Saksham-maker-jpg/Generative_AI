"""
AI Website Development Team — Streamlit UI
Multi-agent system: Supervisor + Requirements + Design + Build + Test
"""

import streamlit as st
import time
import traceback

# ─────────────────────────────────────────
# Page config
# ─────────────────────────────────────────
st.set_page_config(
    page_title="AI Website Development Team",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────
# Header
# ─────────────────────────────────────────
st.title("🏗️ AI Website Development Team")
st.markdown(
    "A **multi-agent system** powered by [Ollama llama3.2](https://ollama.com) + LangGraph. "
    "Describe your website and five AI agents collaborate to build it for you."
)

st.divider()

# ─────────────────────────────────────────
# Sidebar — agent info
# ─────────────────────────────────────────
with st.sidebar:
    st.header("🤖 The Team")

    agents_info = [
        ("👁️", "Supervisor Agent", "Orchestrates the workflow, evaluates each agent's output, and decides the next step."),
        ("📋", "Requirements Agent", "Analyzes your description and produces a structured requirements document."),
        ("🎨", "Design Agent", "Creates a detailed UI/UX design specification with colors, fonts, layout, and interactions."),
        ("⚙️", "Build Agent", "Writes the complete single-file HTML/CSS/JS website from the spec."),
        ("🧪", "Testing Agent", "Reviews the code for bugs, missing features, and quality issues — then fixes them."),
    ]

    for icon, name, desc in agents_info:
        with st.expander(f"{icon} {name}"):
            st.caption(desc)

    st.divider()
    st.markdown("**Model:** `llama3.2` via Ollama")
    st.markdown("**Orchestration:** LangGraph")
    st.caption("⚠️ Requires Ollama running locally with the `llama3.2` model pulled.")

# ─────────────────────────────────────────
# Input form
# ─────────────────────────────────────────
col1, col2 = st.columns([2, 1])

with col1:
    with st.form("build_form", clear_on_submit=False):
        user_request = st.text_area(
            "**Describe the website you want to build:**",
            placeholder=(
                "e.g. A portfolio website for a freelance photographer specialising in landscape "
                "and wildlife photography. Should have a dark, dramatic aesthetic with full-screen "
                "image galleries, a bio section, a contact form, and social media links."
            ),
            height=160,
            help="Be as descriptive as you like — more detail leads to a better result.",
        )
        submitted = st.form_submit_button(
            "🚀 Build My Website",
            use_container_width=True,
            type="primary",
        )

with col2:
    st.markdown("#### Example requests")
    examples = [
        "A landing page for a SaaS project management tool with pricing tiers, feature highlights, and a CTA.",
        "A personal blog about cooking with recipe cards, categories, and a newsletter signup.",
        "A fitness studio website with class schedule, trainer profiles, and membership plans.",
        "An agency portfolio showcasing branding and web design projects with a case study layout.",
    ]
    for ex in examples:
        st.caption(f"• {ex}")

# ─────────────────────────────────────────
# Pipeline execution
# ─────────────────────────────────────────
if submitted:
    if not user_request.strip():
        st.warning("Please enter a website description before clicking Build.")
        st.stop()

    st.divider()
    st.subheader("⚡ Building your website…")

    # ── Agent display config ──────────────────────
    AGENT_META = {
        "supervisor": {
            "icon": "👁️",
            "label": "Supervisor Agent",
            "color": "#6366f1",
        },
        "requirements": {
            "icon": "📋",
            "label": "Requirements Agent",
            "color": "#0ea5e9",
        },
        "design": {
            "icon": "🎨",
            "label": "Design Agent",
            "color": "#a855f7",
        },
        "build": {
            "icon": "⚙️",
            "label": "Build Agent",
            "color": "#22c55e",
        },
        "test": {
            "icon": "🧪",
            "label": "Testing Agent",
            "color": "#f59e0b",
        },
        "finalize": {
            "icon": "✅",
            "label": "Final Review",
            "color": "#10b981",
        },
    }

    PIPELINE_STEPS = ["requirements", "design", "build", "test", "finalize"]
    TOTAL_STEPS = len(PIPELINE_STEPS)

    # ── Progress bar ──────────────────────────────
    progress_bar = st.progress(0, text="Initialising agents…")

    # ── Per-agent output containers ───────────────
    agent_containers = {}
    for step in PIPELINE_STEPS:
        meta = AGENT_META[step]
        agent_containers[step] = st.expander(
            f"{meta['icon']} {meta['label']} — waiting…",
            expanded=False,
        )

    # Supervisor notes container
    supervisor_container = st.expander("👁️ Supervisor Notes", expanded=False)

    # ── Run the graph ─────────────────────────────
    final_state = None
    error_msg = None
    completed_steps = 0

    try:
        from graph import run_pipeline  # imported here so Streamlit doesn't fail on import

        for event in run_pipeline(user_request):
            # Each event is {node_name: state_dict}
            for node_name, state_update in event.items():
                meta = AGENT_META.get(node_name, {"icon": "🔄", "label": node_name, "color": "#888"})
                agent_outputs = state_update.get("agent_outputs", {})
                supervisor_logs = state_update.get("supervisor_log", [])

                if node_name == "supervisor":
                    # Update supervisor notes expander
                    if supervisor_logs:
                        latest = supervisor_logs[-1]
                        with supervisor_container:
                            st.markdown(f"**Stage:** `{latest.get('stage', '')}`")
                            st.info(latest.get("note", ""))

                elif node_name in agent_containers:
                    # Update progress
                    step_idx = PIPELINE_STEPS.index(node_name) if node_name in PIPELINE_STEPS else 0
                    progress = min((step_idx + 1) / TOTAL_STEPS, 1.0)
                    progress_bar.progress(
                        progress,
                        text=f"{meta['icon']} {meta['label']} completed…",
                    )
                    completed_steps = step_idx + 1

                    # Fill the agent's expander with its output
                    output_text = agent_outputs.get(node_name, "")
                    if output_text:
                        # Rebuild the expander label to show ✅
                        with agent_containers[node_name]:
                            if node_name == "build":
                                char_count = len(state_update.get("html_code", ""))
                                st.success(f"Generated **{char_count:,}** characters of HTML/CSS/JS")
                                with st.expander("View raw code"):
                                    st.code(
                                        state_update.get("html_code", "")[:3000] + "\n…[truncated]",
                                        language="html",
                                    )
                            elif node_name == "test":
                                st.success("Code reviewed and fixed.")
                                st.markdown(output_text)
                            elif node_name == "finalize":
                                st.success("✅ All agents complete. Website approved by Supervisor.")
                            else:
                                st.markdown(output_text)

                # Keep the last full state
                final_state = state_update

    except Exception as e:
        error_msg = str(e)
        tb = traceback.format_exc()
        # Check for common Ollama connection error
        if "connection" in error_msg.lower() or "refused" in error_msg.lower() or "ollama" in error_msg.lower():
            st.error(
                "❌ **Could not connect to Ollama.**\n\n"
                "Make sure Ollama is running and the `llama3.2` model is available:\n"
                "```bash\n"
                "ollama serve\n"
                "ollama pull llama3.2 \n"
                "```"
            )
        else:
            st.error(f"❌ Pipeline error: {error_msg}")
            with st.expander("Full traceback"):
                st.code(tb)

    # ── Final output ──────────────────────────────
    if final_state and final_state.get("final_html"):
        progress_bar.progress(1.0, text="✅ Website built successfully!")
        st.balloons()

        st.divider()
        st.subheader("🎉 Your Website is Ready!")

        final_html = final_state["final_html"]

        col_a, col_b = st.columns([1, 1])

        with col_a:
            st.download_button(
                label="⬇️  Download website.html",
                data=final_html,
                file_name="website.html",
                mime="text/html",
                use_container_width=True,
                type="primary",
            )
            st.caption("Open the downloaded file in any browser to preview your website.")

        with col_b:
            st.metric("File size", f"{len(final_html):,} bytes")
            st.metric("Agents used", "5")

        # HTML source preview
        with st.expander("📄 View generated HTML source", expanded=False):
            st.code(final_html, language="html")

    elif not error_msg:
        st.warning(
            "The pipeline completed but no HTML was produced. "
            "This can happen if the model returned unexpected output. Try again."
        )