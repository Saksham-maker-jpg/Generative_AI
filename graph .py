"""Multi-agent website development pipeline using LangGraph and Ollama."""

from typing import Literal, TypedDict

from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph


class WebDevState(TypedDict):
    user_request: str
    requirements: str
    design_spec: str
    html_code: str
    test_report: str
    final_html: str
    next_agent: str
    supervisor_log: list
    agent_outputs: dict
    iteration: int


def get_llm(temperature: float = 0.7) -> ChatOllama:
    return ChatOllama(model="llama3.2", temperature=temperature)


def extract_html(text: str) -> str:
    """Extract HTML from a fenced or unfenced model response."""
    if "```html" in text:
        return text.split("```html", 1)[1].split("```", 1)[0].strip()
    if "```" in text:
        candidate = text.split("```", 2)[1].strip()
        if candidate.lower().startswith(("<!doctype", "<html")):
            return candidate
    return text.strip()


def _save(state: WebDevState, name: str, value: str, **updates) -> WebDevState:
    outputs = dict(state.get("agent_outputs", {}))
    outputs[name] = value
    return {**state, **updates, "agent_outputs": outputs}


def requirements_agent(state: WebDevState) -> WebDevState:
    prompt = f"""Act as a senior requirements analyst. For this website request:
{state['user_request']}
Create a structured requirements document covering overview, features, content,
visual direction, sitemap, UX, technical/accessibility constraints, and out of scope items."""
    content = get_llm(.5).invoke(prompt).content
    return _save(state, "requirements", content, requirements=content,
                 next_agent="supervisor_after_requirements")


def design_agent(state: WebDevState) -> WebDevState:
    prompt = f"""Act as a lead UI/UX designer. Design an implementable specification for:
Request: {state['user_request']}
Requirements: {state['requirements']}
Include exact colors, typography, spacing, responsive layouts, components,
effects, animations, imagery, accessibility, and breakpoints."""
    content = get_llm(.8).invoke(prompt).content
    return _save(state, "design", content, design_spec=content,
                 next_agent="supervisor_after_design")


def build_agent(state: WebDevState) -> WebDevState:
    prompt = f"""Build a complete production-quality single-file website.
Request: {state['user_request']}
Requirements: {state['requirements']}
Design: {state['design_spec']}
Return ONLY raw HTML beginning with <!DOCTYPE html> and ending with </html>.
Put CSS in one style tag and JavaScript in one script tag. Use semantic,
accessible, responsive HTML with a nav, hero, at least three content sections,
footer, smooth scrolling, animations, mobile navigation, and real interactions.
Use no CSS framework or external JavaScript CDN."""
    html = extract_html(get_llm(.3).invoke(prompt).content)
    return _save(state, "build", f"Generated {len(html)} characters.",
                  html_code=html, next_agent="supervisor_after_build")


def tester_agent(state: WebDevState) -> WebDevState:
    prompt = f"""Review and fix this complete website for HTML/CSS/JavaScript errors,
responsiveness, accessibility, navigation, and feature completeness.
Request: {state['user_request']}
Code:
{state['html_code']}
Return a brief line beginning TEST REPORT: followed by the full corrected HTML,
starting with <!DOCTYPE html> and ending with </html>."""
    raw = get_llm(.2).invoke(prompt).content
    start = raw.upper().find("<!DOCTYPE")
    html = extract_html(raw[start:] if start >= 0 else raw)
    report = raw[:start].strip() if start > 0 else "TEST REPORT: Code reviewed and optimized."
    if len(html) < 200:
        html = state["html_code"]
        report += " Extraction failed; original retained."
    return _save(state, "tester", report, html_code=html, test_report=report,
                  next_agent="supervisor_after_test")


def supervisor_node(state: WebDevState) -> WebDevState:
    key = state.get("next_agent", "requirements")
    routes = {
        "requirements": "requirements",
        "supervisor_after_requirements": "design",
        "supervisor_after_design": "build",
        "supervisor_after_build": "test",
        "supervisor_after_test": "finalize",
    }
    route = routes.get(key, "finalize")
    note = get_llm(.4).invoke(
        f"Briefly explain this pipeline transition: {key} -> {route}. Request: {state['user_request']}"
    ).content
    logs = list(state.get("supervisor_log", []))
    logs.append({"stage": key, "note": note})
    return {**state, "next_agent": route, "supervisor_log": logs,
            "iteration": state.get("iteration", 0) + 1}


def finalize_node(state: WebDevState) -> WebDevState:
    outputs = dict(state.get("agent_outputs", {}))
    outputs["supervisor"] = "Website approved."
    return {**state, "final_html": state.get("html_code", ""),
            "agent_outputs": outputs}


def route_supervisor(state: WebDevState) -> Literal["requirements", "design", "build", "test", "finalize"]:
    return state.get("next_agent", "finalize")


def build_graph():
    workflow = StateGraph(WebDevState)
    for name, node in (("supervisor", supervisor_node), ("requirements", requirements_agent),
                       ("design", design_agent), ("build", build_agent),
                       ("test", tester_agent), ("finalize", finalize_node)):
        workflow.add_node(name, node)
    workflow.add_edge(START, "supervisor")
    workflow.add_conditional_edges("supervisor", route_supervisor,
                                   {x: x for x in ("requirements", "design", "build", "test", "finalize")})
    for name in ("requirements", "design", "build", "test"):
        workflow.add_edge(name, "supervisor")
    workflow.add_edge("finalize", END)
    return workflow.compile()


def run_pipeline(user_request: str):
    initial: WebDevState = {
        "user_request": user_request, "requirements": "", "design_spec": "",
        "html_code": "", "test_report": "", "final_html": "",
        "next_agent": "requirements", "supervisor_log": [],
        "agent_outputs": {}, "iteration": 0,
    }
    return build_graph().stream(initial)
"""
Multi-Agent Website Development System using LangGraph + Ollama (llama3.2)

Agents:
  - supervisor: Routes tasks between agents, evaluates work quality
  - requirements: Gathers and structures project requirements
  - design: Creates detailed design specification
  - build: Generates complete single-file HTML/CSS/JS website
  - tester: Reviews, tests, and fixes the generated code
"""

from typing import TypedDict, Literal
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END


# ─────────────────────────────────────────
# State
# ─────────────────────────────────────────

class WebDevState(TypedDict):
    user_request: str
    requirements: str
    design_spec: str
    html_code: str
    test_report: str
    final_html: str
    next_agent: str          # routing key for supervisor conditional edges
    supervisor_log: list     # list of supervisor commentary strings
    agent_outputs: dict      # {agent_name: output_text} for UI display
    iteration: int           # loop counter for safety


# ─────────────────────────────────────────
# LLM factory
# ─────────────────────────────────────────

def get_llm(temperature: float = 0.7) -> ChatOllama:
    return ChatOllama(model="llama3.2", temperature=temperature)


# ─────────────────────────────────────────
# Helper: extract code block from LLM output
# ─────────────────────────────────────────

def extract_html(text: str) -> str:
    """Pull HTML out of markdown code fences if present."""
    if "```html" in text:
        parts = text.split("```html")
        if len(parts) > 1:
            return parts[1].split("```")[0].strip()
    if "```" in text:
        parts = text.split("```")
        if len(parts) > 1:
            candidate = parts[1].strip()
            if candidate.lower().startswith("<!doctype") or candidate.lower().startswith("<html"):
                return candidate
    return text.strip()


# ─────────────────────────────────────────
# Agent nodes
# ─────────────────────────────────────────

def requirements_agent(state: WebDevState) -> WebDevState:
    llm = get_llm(temperature=0.5)

    prompt = f"""You are a Senior Requirements Analyst on a web development team.

A client has requested a website with the following description:
"{state['user_request']}"

Your task: produce a thorough, structured Requirements Document.

Include ALL of the following sections:
1. **Project Overview** — one-paragraph summary of purpose and target audience
2. **Core Features & Functionality** — numbered list of must-have features
3. **Content Requirements** — pages needed, content blocks, copy tone
4. **Visual Design Direction** — mood, style, color palette hints, imagery style
5. **Navigation & Structure** — sitemap / page hierarchy
6. **User Experience Requirements** — key user journeys, interactions
7. **Technical Constraints** — browser support, performance, accessibility
8. **Out-of-Scope Items** — what this version will NOT include

Be specific. Avoid vague language. The design and development team will act directly on this document."""

    response = llm.invoke(prompt)
    content = response.content

    outputs = dict(state.get("agent_outputs", {}))
    outputs["requirements"] = content

    return {
        **state,
        "requirements": content,
        "next_agent": "supervisor_after_requirements",
        "agent_outputs": outputs,
    }


def design_agent(state: WebDevState) -> WebDevState:
    llm = get_llm(temperature=0.8)

    prompt = f"""You are a Lead UI/UX Designer on a web development team.

Client Request: "{state['user_request']}"

Requirements Document:
{state['requirements']}

Your task: write a detailed Design Specification that a developer can implement directly.

Include ALL of the following:
1. **Color Palette** — primary, secondary, accent, background, text colors (provide exact hex codes)
2. **Typography** — heading font, body font, font sizes for h1/h2/h3/p/small, line-height, letter-spacing
3. **Spacing System** — base unit, padding/margin scale
4. **Layout Architecture** — describe each major section (header, hero, sections, footer) with layout approach (flex/grid)
5. **Component Styles** — buttons (states: default/hover/active), cards, form elements, navigation links
6. **Visual Effects** — shadows, borders, border-radius, gradients if any
7. **Animation & Interaction** — scroll animations, hover transitions, loading effects
8. **Responsive Breakpoints** — mobile (<768px), tablet (768–1024px), desktop (>1024px) behavior
9. **Imagery & Icons** — describe visual tone; if icons are used, describe style (use inline SVG or CSS only)
10. **Overall Aesthetic** — 2-3 sentences capturing the look and feel

Be precise with values. No vague terms like "clean" or "modern" — only concrete, implementable details."""

    response = llm.invoke(prompt)
    content = response.content

    outputs = dict(state.get("agent_outputs", {}))
    outputs["design"] = content

    return {
        **state,
        "design_spec": content,
        "next_agent": "supervisor_after_design",
        "agent_outputs": outputs,
    }


def build_agent(state: WebDevState) -> WebDevState:
    llm = get_llm(temperature=0.3)

    prompt = f"""You are a Senior Full-Stack Web Developer. Build a complete, production-quality website.

Client Request: "{state['user_request']}"

Requirements:
{state['requirements']}

Design Specification:
{state['design_spec']}

STRICT RULES — follow every one:
- Output ONLY raw HTML. No explanations. No markdown. No code fences.
- The file MUST start with exactly: <!DOCTYPE html>
- The file MUST end with exactly: </html>
- ALL CSS goes inside a single <style> tag in <head>
- ALL JavaScript goes inside a single <script> tag just before </body>
- NO external CDN links for CSS frameworks (Bootstrap, Tailwind, etc.) — write all CSS from scratch
- Google Fonts @import is allowed
- Write real, meaningful placeholder content appropriate for the website topic
- Must be fully mobile-responsive
- Must include smooth scroll, hover transitions, and CSS animations
- Must have at minimum: header with nav, hero section, 3+ content sections, footer
- JavaScript must add real interactivity (mobile nav toggle, scroll effects, form handling, etc.)
- Code must be clean, well-structured, and commented

Write the complete HTML file now:"""

    response = llm.invoke(prompt)
    html = extract_html(response.content)

    outputs = dict(state.get("agent_outputs", {}))
    outputs["build"] = f"Generated {len(html)} characters of HTML/CSS/JS code."

    return {
        **state,
        "html_code": html,
        "next_agent": "supervisor_after_build",
        "agent_outputs": outputs,
    }


def tester_agent(state: WebDevState) -> WebDevState:
    llm = get_llm(temperature=0.2)

    prompt = f"""You are a Senior QA Engineer and Frontend Developer. Review and fix the generated website code.

Client Request: "{state['user_request']}"

Code to review:
{state['html_code']}

YOUR TASKS:
1. Check HTML structure validity (doctype, head, body, meta tags, viewport)
2. Verify CSS completeness — look for missing styles, broken layout rules, undefined variables
3. Review JavaScript — fix syntax errors, undefined functions, broken event listeners
4. Confirm mobile responsiveness — media queries present and correct
5. Check all navigation links work (use # anchors if no real pages)
6. Verify all required features from the client request are implemented
7. Add any missing content sections
8. Improve visual polish — better spacing, transitions, or missing hover states
9. Ensure accessibility basics (alt text, semantic HTML, focus states)
10. Fix any issues found

WRITE YOUR TEST REPORT first (prefix it with "TEST REPORT:"), then output the complete fixed HTML file.

STRICT RULES for the HTML output:
- Start the HTML section with exactly: <!DOCTYPE html>
- End with exactly: </html>
- Output the full corrected HTML — not just the diff

TEST REPORT:
[write issues found here]

FIXED HTML:"""

    response = llm.invoke(prompt)
    raw = response.content

    # Split test report from HTML
    test_report = ""
    html = ""

    if "TEST REPORT:" in raw and "<!DOCTYPE" in raw.upper():
        parts = raw.split("TEST REPORT:", 1)
        after_report = parts[1] if len(parts) > 1 else raw
        # Find where HTML starts
        idx = after_report.upper().find("<!DOCTYPE")
        if idx >= 0:
            test_report = "TEST REPORT:" + after_report[:idx].strip()
            html = after_report[idx:].strip()
        else:
            test_report = "TEST REPORT:" + after_report
            html = state["html_code"]
    elif "<!DOCTYPE" in raw.upper():
        idx = raw.upper().find("<!DOCTYPE")
        test_report = raw[:idx].strip() if idx > 0 else "Code reviewed."
        html = raw[idx:].strip()
    else:
        html = extract_html(raw)
        test_report = "Code reviewed and optimized."

    # Fallback to original if extraction fails badly
    if len(html) < 200:
        html = state["html_code"]
        test_report += " (kept original — extraction issue)"

    outputs = dict(state.get("agent_outputs", {}))
    outputs["tester"] = test_report or "Testing complete. Code reviewed and validated."

    return {
        **state,
        "html_code": html,
        "test_report": test_report,
        "next_agent": "supervisor_after_test",
        "agent_outputs": outputs,
    }


# ─────────────────────────────────────────
# Supervisor node
# ─────────────────────────────────────────

def supervisor_node(state: WebDevState) -> WebDevState:
    """
    The Supervisor evaluates completed work and decides the next agent to call.
    Uses a lightweight LLM call to produce a commentary on each transition.
    """
    llm = get_llm(temperature=0.4)
    next_agent = state.get("next_agent", "requirements")
    logs = list(state.get("supervisor_log", []))
    iteration = state.get("iteration", 0)

    # Map routing key → human label and next action
    transition_map = {
        "requirements": {
            "context": "The project is starting. No work has been done yet.",
            "decision": "Route to Requirements Agent",
            "route": "requirements",
        },
        "supervisor_after_requirements": {
            "context": f"Requirements Agent has delivered:\n{state.get('requirements', '')[:400]}...",
            "decision": "Evaluate requirements quality and route to Design Agent",
            "route": "design",
        },
        "supervisor_after_design": {
            "context": f"Design Agent has delivered a design spec ({len(state.get('design_spec',''))} chars).",
            "decision": "Evaluate design spec and route to Build Agent",
            "route": "build",
        },
        "supervisor_after_build": {
            "context": f"Build Agent has generated {len(state.get('html_code',''))} characters of HTML/CSS/JS.",
            "decision": "Evaluate build output and route to Testing Agent",
            "route": "test",
        },
        "supervisor_after_test": {
            "context": f"Testing Agent has reviewed the code. Test report: {state.get('test_report','')[:300]}",
            "decision": "Final review complete — approve and finalize",
            "route": "finalize",
        },
    }

    info = transition_map.get(next_agent, {"context": "Unknown state.", "decision": "Finalize.", "route": "finalize"})

    prompt = f"""You are a Project Supervisor overseeing a team of AI web development agents.

Client Request: "{state['user_request']}"

Current Situation: {info['context']}

Your Decision: {info['decision']}

Write a brief supervisor note (2–3 sentences) explaining:
- What you observed about the work just completed (if any)
- What the next step is and why
- Any specific guidance for the next agent

Keep it professional and concise."""

    response = llm.invoke(prompt)
    logs.append({
        "stage": next_agent,
        "note": response.content,
    })

    return {
        **state,
        "supervisor_log": logs,
        "next_agent": info["route"],
        "iteration": iteration + 1,
    }


def finalize_node(state: WebDevState) -> WebDevState:
    """Supervisor approves and sets final_html."""
    html = state.get("html_code", "")
    outputs = dict(state.get("agent_outputs", {}))
    outputs["supervisor"] = "✅ Website approved. All agents have completed their work successfully."

    return {
        **state,
        "final_html": html,
        "agent_outputs": outputs,
    }


# ─────────────────────────────────────────
# Routing function for supervisor edges
# ─────────────────────────────────────────

def route_supervisor(state: WebDevState) -> Literal[
    "requirements", "design", "build", "test", "finalize"
]:
    return state.get("next_agent", "finalize")


# ─────────────────────────────────────────
# Build & compile the graph
# ─────────────────────────────────────────

def build_graph():
    workflow = StateGraph(WebDevState)

    # Register nodes
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("requirements", requirements_agent)
    workflow.add_node("design", design_agent)
    workflow.add_node("build", build_agent)
    workflow.add_node("test", tester_agent)
    workflow.add_node("finalize", finalize_node)

    # Entry: always start at supervisor
    workflow.add_edge(START, "supervisor")

    # Supervisor routes to workers
    workflow.add_conditional_edges(
        "supervisor",
        route_supervisor,
        {
            "requirements": "requirements",
            "design": "design",
            "build": "build",
            "test": "test",
            "finalize": "finalize",
        },
    )

    # All workers return to supervisor
    workflow.add_edge("requirements", "supervisor")
    workflow.add_edge("design", "supervisor")
    workflow.add_edge("build", "supervisor")
    workflow.add_edge("test", "supervisor")

    # Finalize is terminal
    workflow.add_edge("finalize", END)

    return workflow.compile()


# ─────────────────────────────────────────
# Public runner (used by app.py)
# ─────────────────────────────────────────

def run_pipeline(user_request: str):
    """
    Run the full multi-agent pipeline.
    Returns a generator of (event_dict) as LangGraph streams each step.
    """
    graph = build_graph()
    initial_state: WebDevState = {
        "user_request": user_request,
        "requirements": "",
        "design_spec": "",
        "html_code": "",
        "test_report": "",
        "final_html": "",
        "next_agent": "requirements",
        "supervisor_log": [],
        "agent_outputs": {},
        "iteration": 0,
    }
    return graph.stream(initial_state)
