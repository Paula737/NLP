"""
Report Generator Agent
=======================
Job: take a finished, cited answer (plus the original question and sources)
and produce a polished, downloadable PDF report.

This is implemented as a real LangChain tool-calling agent - it has genuine
access to design tools (font family, color theme) as well as content tools
(headings, paragraphs, bullet lists, tables), and decides for itself how to
style and structure the report rather than following a fixed template.

Tools available to the agent:
  1. set_style          - choose a font family and color theme (the "design
                           tools" - font + colors)
  2. add_heading        - add a styled section heading
  3. add_paragraph      - add a body paragraph
  4. add_bullet_list    - add a bulleted list
  5. add_table          - add a data table (e.g. for sources/citations)
  6. finalize_report    - render everything into an actual PDF file

Entry point: generate_report(question, answer_text, sources) -> PDF file path
"""

import os
import re
import sys
import unicodedata
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain.agents import create_agent

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.units import inch
from reportlab.lib import colors as rl_colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    ListFlowable, ListItem, HRFlowable,
)

GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
REPORTS_DIR = Path(__file__).resolve().parent.parent / "data" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def _llm():
    return ChatGroq(model=GROQ_MODEL, temperature=0.3)


# Punctuation above codepoint 0xFF that WinAnsiEncoding (used by reportlab's
# base-14 fonts) actually DOES support and renders fine - must not be
# mistaken for "foreign script" and stripped.
_SAFE_HIGH_CODEPOINTS = {0x2013, 0x2014, 0x2018, 0x2019, 0x201C, 0x201D, 0x2026, 0x2022}


def _is_renderable(ch: str) -> bool:
    return ord(ch) <= 0xFF or ord(ch) in _SAFE_HIGH_CODEPOINTS


def sanitize_text(text: str) -> str:
    """reportlab's base-14 fonts (Helvetica/Times/Courier) only support the
    WinAnsi encoding (Latin-1 plus a handful of "smart punctuation" glyphs
    like curly quotes and em/en dashes). Two distinct problems this guards
    against:
      1. Invisible Unicode 'format' control characters (category Cf) - most
         notably LRM/RLM bidi marks that leak in from OCR'd Arabic/Hebrew
         content - render as solid black boxes instead of being invisible.
      2. Genuine non-Latin script (actual Arabic/Hebrew/CJK letters) cannot
         be rendered at all by these fonts. Such runs are collapsed into a
         single clear placeholder rather than one black box per character -
         while carefully NOT touching legitimate smart punctuation (curly
         quotes, em/en dashes, ellipsis, bullets) that these fonts do
         support despite being above the raw Latin-1 codepoint range.
    """
    if not text:
        return text
    text = "".join(" " if unicodedata.category(ch) == "Cf" else ch for ch in text)

    n = len(text)
    result = []
    i = 0
    while i < n:
        ch = text[i]
        if _is_renderable(ch):
            result.append(ch)
            i += 1
            continue
        # Start of a non-renderable run: extend through inline spaces as
        # long as more non-renderable content follows, so a whole foreign
        # phrase collapses into ONE placeholder instead of one per word.
        j = i
        while j < n:
            if _is_renderable(text[j]) and text[j] != " ":
                break
            if text[j] == " ":
                k = j
                while k < n and text[k] == " ":
                    k += 1
                if k < n and not _is_renderable(text[k]):
                    j = k
                    continue
                break
            j += 1
        result.append(" [non-Latin script omitted] ")
        i = j
    cleaned = "".join(result)
    return re.sub(r" {2,}", " ", cleaned).strip()


# ---------------------------------------------------------------------------
# DESIGN TOOLS: font families + color themes
# ---------------------------------------------------------------------------
# reportlab's 12 built-in "base 14" fonts require no external font files at
# all, which sidesteps the kind of system-dependency pain we hit with
# Tesseract - this works identically on every machine, zero setup.
FONT_MAP = {
    "sans": {"body": "Helvetica", "bold": "Helvetica-Bold", "italic": "Helvetica-Oblique"},
    "serif": {"body": "Times-Roman", "bold": "Times-Bold", "italic": "Times-Italic"},
    "mono": {"body": "Courier", "bold": "Courier-Bold", "italic": "Courier-Oblique"},
}

THEME_MAP = {
    "navy": {
        "heading": rl_colors.HexColor("#14263F"),
        "accent": rl_colors.HexColor("#2F8F7E"),
        "text": rl_colors.HexColor("#1A1A1A"),
        "table_header_bg": rl_colors.HexColor("#14263F"),
        "table_header_text": rl_colors.white,
    },
    "warm": {
        "heading": rl_colors.HexColor("#7A4A1E"),
        "accent": rl_colors.HexColor("#C9862E"),
        "text": rl_colors.HexColor("#2B2117"),
        "table_header_bg": rl_colors.HexColor("#C9862E"),
        "table_header_text": rl_colors.white,
    },
    "mono": {
        "heading": rl_colors.HexColor("#111111"),
        "accent": rl_colors.HexColor("#555555"),
        "text": rl_colors.HexColor("#111111"),
        "table_header_bg": rl_colors.HexColor("#333333"),
        "table_header_text": rl_colors.white,
    },
}


class ReportBuilder:
    """Mutable, per-report state shared across tool calls within one
    generation run: accumulates flowables (the PDF's content, in order) and
    tracks the currently chosen font/theme so later add_* calls render
    consistently with whatever set_style chose."""

    def __init__(self):
        self.flowables: list = []
        self.font = "sans"
        self.theme = "navy"

    def fonts(self):
        return FONT_MAP[self.font]

    def palette(self):
        return THEME_MAP[self.theme]

    def heading_style(self, level: int) -> ParagraphStyle:
        f, c = self.fonts(), self.palette()
        size = {1: 22, 2: 15, 3: 12}.get(level, 12)
        return ParagraphStyle(
            f"Heading{level}", fontName=f["bold"], fontSize=size,
            textColor=c["heading"], leading=size * 1.25,
            spaceBefore=14 if level > 1 else 0, spaceAfter=8,
        )

    def body_style(self) -> ParagraphStyle:
        f, c = self.fonts(), self.palette()
        return ParagraphStyle(
            "Body", fontName=f["body"], fontSize=10.5, leading=15.5,
            textColor=c["text"], spaceAfter=8, alignment=TA_LEFT,
        )


# ---------------------------------------------------------------------------
# TOOL FACTORIES (bound to one ReportBuilder instance per report)
# ---------------------------------------------------------------------------
def make_report_tools(builder: ReportBuilder):

    @tool
    def set_style(font: str, theme: str) -> str:
        """Choose the report's visual design: font family and color theme.
        Call this FIRST, before adding any content, based on the tone of
        the material (e.g. 'serif'+'navy' for a formal analytical report,
        'sans'+'warm' for something more approachable, 'mono'+'mono' for a
        technical/data-heavy report).

        Args:
            font: One of 'sans', 'serif', 'mono'.
            theme: One of 'navy', 'warm', 'mono'.
        """
        if font not in FONT_MAP:
            return f"Invalid font '{font}'. Choose one of: {list(FONT_MAP)}"
        if theme not in THEME_MAP:
            return f"Invalid theme '{theme}'. Choose one of: {list(THEME_MAP)}"
        builder.font = font
        builder.theme = theme
        return f"Style set: font={font}, theme={theme}"

    @tool
    def add_heading(text: str, level: int = 1) -> str:
        """Add a section heading. Level 1 is the report title (use once, at
        the top), level 2 for major sections, level 3 for sub-sections.

        Args:
            text: The heading text.
            level: 1, 2, or 3.
        """
        builder.flowables.append(Paragraph(sanitize_text(text), builder.heading_style(level)))
        if level == 1:
            c = builder.palette()
            builder.flowables.append(HRFlowable(width="100%", thickness=1.5, color=c["accent"], spaceAfter=14))
        return "Heading added."

    @tool
    def add_paragraph(text: str) -> str:
        """Add a body paragraph of text.

        Args:
            text: The paragraph content (plain text; basic <b>/<i> tags allowed).
        """
        builder.flowables.append(Paragraph(sanitize_text(text), builder.body_style()))
        return "Paragraph added."

    @tool
    def add_bullet_list(items: List[str]) -> str:
        """Add a bulleted list.

        Args:
            items: List of bullet point strings.
        """
        style = builder.body_style()
        bullets = [ListItem(Paragraph(sanitize_text(item), style), leftIndent=14) for item in items]
        builder.flowables.append(ListFlowable(bullets, bulletType="bullet", start="circle"))
        builder.flowables.append(Spacer(1, 8))
        return f"Added {len(items)} bullet points."

    @tool
    def add_table(headers: List[str], rows: List[List[str]]) -> str:
        """Add a data table - ideal for a Sources/Citations section
        (document name + page number) or any tabular comparison.

        Args:
            headers: Column header labels.
            rows: List of rows, each a list of cell strings matching headers.
        """
        c = builder.palette()
        data = [[sanitize_text(h) for h in headers]] + [[sanitize_text(cell) for cell in row] for row in rows]
        table = Table(data, hAlign="LEFT", repeatRows=1)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), c["table_header_bg"]),
            ("TEXTCOLOR", (0, 0), (-1, 0), c["table_header_text"]),
            ("FONTNAME", (0, 0), (-1, 0), builder.fonts()["bold"]),
            ("FONTNAME", (0, 1), (-1, -1), builder.fonts()["body"]),
            ("FONTSIZE", (0, 0), (-1, -1), 9.5),
            ("GRID", (0, 0), (-1, -1), 0.5, rl_colors.HexColor("#CCCCCC")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [rl_colors.white, rl_colors.HexColor("#F5F5F5")]),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ]))
        builder.flowables.append(table)
        builder.flowables.append(Spacer(1, 10))
        return "Table added."

    @tool
    def finalize_report(filename: str) -> str:
        """Render everything added so far into a final PDF file. Call this
        exactly ONCE, as the last step, after set_style and all content has
        been added.

        Args:
            filename: Desired filename, e.g. 'skills_summary_report'
                (the .pdf extension is added automatically if omitted).
        """
        safe_name = filename if filename.lower().endswith(".pdf") else filename + ".pdf"
        safe_name = re.sub(r"[^\w\-. ]", "_", safe_name).strip() or "report.pdf"
        out_path = REPORTS_DIR / safe_name

        doc = SimpleDocTemplate(
            str(out_path), pagesize=LETTER,
            leftMargin=0.9 * inch, rightMargin=0.9 * inch,
            topMargin=0.9 * inch, bottomMargin=0.9 * inch,
            title=safe_name.replace(".pdf", ""),
        )
        doc.build(list(builder.flowables))
        return str(out_path)

    return [set_style, add_heading, add_paragraph, add_bullet_list, add_table, finalize_report]


# ---------------------------------------------------------------------------
# FULL REPORT AGENT
# ---------------------------------------------------------------------------
REPORT_SYSTEM_PROMPT = """You are the Report Generator Agent in a multi-agent
RAG system. You take a question, its cited answer, and the sources it came
from, and turn them into a polished, professional PDF report.

Process (follow in order):
1. Call `set_style` first, choosing a font ('sans', 'serif', or 'mono') and
   color theme ('navy', 'warm', or 'mono') that suits the content's tone.
2. Call `add_heading` with the report title (derived from the question), level=1.
3. Call `add_paragraph` for a short executive summary (1-3 sentences).
4. Call `add_heading` (level=2) for a "Detailed Answer" section, then
   `add_paragraph` and/or `add_bullet_list` to present the full answer
   content clearly. Preserve all facts and figures exactly as given - do
   not invent anything not present in the provided answer.
5. Call `add_heading` (level=2) for a "Sources" section, then `add_table`
   with columns like ["Document", "Page"] listing every source provided.
6. Call `finalize_report` exactly once, with a short descriptive filename
   based on the question.

IMPORTANT - script limitation: this report's fonts can only render Latin
script (English and most European languages). If the source answer quotes
non-Latin text (Arabic, Hebrew, Chinese, etc.), do NOT include that script
verbatim anywhere in the report - it will render as unreadable black boxes.
Instead, only include the English translation/transliteration/description
of that content, exactly as it appears in the provided answer.

Do not skip finalize_report. Do not call it more than once.
"""


def build_report_agent(builder: ReportBuilder):
    return create_agent(
        model=_llm(),
        tools=make_report_tools(builder),
        system_prompt=REPORT_SYSTEM_PROMPT,
    )


def generate_report(question: str, answer_text: str, sources: List[dict]) -> str:
    """
    Runs the question + final answer + sources through the Report Generator
    Agent, which decides the styling and structure, then produces a real
    PDF file. Returns the absolute path to the generated PDF.

    sources: list of {"document": str, "pages": List[int]} dicts (matching
    the Answer Agent's source_formatter output).
    """
    builder = ReportBuilder()
    agent = build_report_agent(builder)

    sources_block = "\n".join(
        f"- {s['document']}: pages {', '.join(map(str, s.get('pages', [])))}"
        for s in sources
    ) or "(no sources)"

    input_text = f"""Question: {question}

Final cited answer:
{answer_text}

Sources:
{sources_block}"""

    result = agent.invoke(
        {"messages": [{"role": "user", "content": input_text}]},
        config={"recursion_limit": 20},
    )

    # The finalize_report tool's return value (the file path) is the
    # authoritative result - pull it from the tool call history rather than
    # trusting the agent's final text summary.
    for msg in result["messages"]:
        if getattr(msg, "name", None) == "finalize_report":
            content = msg.content
            return content.strip('"') if isinstance(content, str) else str(content)

    raise RuntimeError("Report agent finished without calling finalize_report.")


if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "What are the main skills listed?"
    a = sys.argv[2] if len(sys.argv) > 2 else "The candidate has strong Python and ML skills (resume.pdf, p.1)."
    path = generate_report(q, a, [{"document": "resume.pdf", "pages": [1]}])
    print(f"Report saved to: {path}")