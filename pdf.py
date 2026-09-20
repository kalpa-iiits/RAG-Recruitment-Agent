"""Render a generated resume into a one-page PDF.

The rewrite endpoints return plain text that the model was asked to format
as light markdown: a name and contact block, `##` section headings,
`**bold**`, `*italic*` and `-` bullets. This lays that out as a conventional
single-column resume — centred name over a role line and a contact line,
then sections whose headings sit above a full-width rule.

Two things the layout is built around:

*One page.* A resume that runs three lines onto a second page is the common
case, and the fix is not to cut the words but to set them slightly smaller.
Everything here is sized off a single scale factor, so the document can be
built, measured, and rebuilt a notch tighter until it fits. Below
MIN_SCALE the type stops being comfortable to read, so that is where it
stops trying and lets the resume run long rather than shrink it to nothing.

*Plain.* One column, one typeface family, no rules but the section
underlines. Keyword scanners read this before a person does.
"""

import html
import re
from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    HRFlowable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
)

_BULLET = re.compile(r"^\s*[-*•]\s+(.*)$")
_HEADING = re.compile(r"^\s*(#{1,6})\s+(.*)$")
_RULE = re.compile(r"^\s*([-*_])\1{2,}\s*$")
_BOLD = re.compile(r"\*\*(.+?)\*\*")
_ITALIC = re.compile(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)")
# "Languages: Python, C++" — the skill-category shape, whose label carries
# the bold in the reference layout even when the model didn't mark it up.
_LABEL = re.compile(r"^([A-Z][\w &/+.,()\-]{0,44}):\s+(?=\S)")

# Headings a resume might open with when it has no name block of its own.
_SECTION_LABELS = frozenset(
    {
        "SUMMARY",
        "PROFESSIONAL SUMMARY",
        "OBJECTIVE",
        "PROFILE",
        "EXPERIENCE",
        "PROFESSIONAL EXPERIENCE",
        "WORK EXPERIENCE",
        "EMPLOYMENT HISTORY",
        "EDUCATION",
        "SKILLS",
        "TECHNICAL SKILLS",
        "CERTIFICATIONS",
        "PROJECTS",
        "CONTACT",
    }
)

_NAVY = colors.HexColor("#1f3864")
_HEADING_BLUE = colors.HexColor("#2f5496")
_RULE_GREY = colors.HexColor("#333f50")
_INK = colors.HexColor("#1a1a1a")
_META = colors.HexColor("#404040")

# How far the document may be scaled down in pursuit of a single page.
MIN_SCALE = 0.74
_STEPS = 8


def _styles(scale: float) -> dict:
    """Every size in the document, derived from one factor."""
    return {
        "name": ParagraphStyle(
            "ResumeName",
            fontName="Helvetica-Bold",
            fontSize=17 * scale,
            leading=20 * scale,
            alignment=TA_CENTER,
            textColor=_NAVY,
            spaceAfter=3 * scale,
        ),
        "meta": ParagraphStyle(
            "ResumeMeta",
            fontName="Helvetica",
            fontSize=8.4 * scale,
            leading=11 * scale,
            alignment=TA_CENTER,
            textColor=_META,
            spaceAfter=1.5 * scale,
        ),
        "section": ParagraphStyle(
            "ResumeSection",
            fontName="Helvetica-Bold",
            fontSize=9.4 * scale,
            leading=11.5 * scale,
            textColor=_HEADING_BLUE,
            spaceBefore=7.5 * scale,
            spaceAfter=1.5 * scale,
        ),
        "body": ParagraphStyle(
            "ResumeBody",
            fontName="Helvetica",
            fontSize=9.2 * scale,
            leading=12.2 * scale,
            textColor=_INK,
            alignment=TA_JUSTIFY,
            spaceAfter=3 * scale,
        ),
        "bullet": ParagraphStyle(
            "ResumeBullet",
            fontName="Helvetica",
            fontSize=9.2 * scale,
            leading=12.2 * scale,
            textColor=_INK,
            alignment=TA_LEFT,
            spaceAfter=1.5 * scale,
            # A round dot on the text baseline, rather than the raised
            # ZapfDingbats glyph a ListFlowable would reach for.
            bulletFontName="Helvetica",
            bulletFontSize=9.2 * scale,
            bulletIndent=2 * scale,
            leftIndent=11 * scale,
        ),
    }


def _inline(text: str) -> str:
    """Markdown emphasis to reportlab's mini-HTML, escaping the rest."""
    escaped = html.escape(text.strip())
    escaped = _BOLD.sub(r"<b>\1</b>", escaped)
    escaped = _ITALIC.sub(r"<i>\1</i>", escaped)
    return escaped


def _with_label(line: str) -> str:
    """Bold a leading `Label:` the model left plain, as the layout expects."""
    if line.lstrip().startswith(("*", "#")):
        return line
    return _LABEL.sub(lambda m: f"**{m.group(1)}:** ", line, count=1)


def _looks_like_heading(line: str) -> bool:
    """An unmarked section heading: short, upper case, no sentence punctuation.

    Models often emit `EXPERIENCE` rather than `## Experience`, and without
    this every section title would render as body text.
    """
    stripped = line.strip()
    if not 2 < len(stripped) <= 48 or stripped.endswith((".", ",", ";", ":")):
        return False
    letters = [c for c in stripped if c.isalpha()]
    return bool(letters) and all(c.isupper() for c in letters)


def _is_section(line: str) -> bool:
    return bool(_HEADING.match(line)) or _looks_like_heading(line)


def _split_header(lines: list[str]) -> tuple[list[str], list[str]]:
    """Separate the name/contact block from the sections that follow.

    The first non-blank line is the name, and the header runs from there to
    the first section heading; the role and contact lines in between sit
    centred beneath it.

    Position decides this, not shape. A name set in capitals —
    "KALPAJYOTI HANDIQUE" — is indistinguishable from a heading like
    "TECHNICAL SKILLS" by any test of the text itself, so judging the first
    line by appearance sent the whole header through as justified body text.
    What a resume opens with is a name.
    """
    first = next((i for i, line in enumerate(lines) if line.strip()), None)
    if first is None:
        return [], lines

    # Unless it plainly isn't one: an explicit markdown heading, or a section
    # label the writer led with because this resume carries no header at all.
    opening = lines[first].strip()
    if _HEADING.match(opening) or opening.upper() in _SECTION_LABELS:
        return [], lines

    for index in range(first + 1, len(lines)):
        if lines[index].strip() and _is_section(lines[index]):
            return lines[:index], lines[index:]
    return lines, []


def _story(text: str, scale: float) -> list:
    """Build the flowables afresh — reportlab consumes them on build."""
    styles = _styles(scale)
    lines = (text or "").splitlines()
    header, rest = _split_header(lines)

    story: list = []

    header_lines = [line.strip() for line in header if line.strip()]
    if header_lines:
        name = _inline(header_lines[0]).upper()
        story.append(Paragraph(name, styles["name"]))
        for line in header_lines[1:]:
            story.append(Paragraph(_inline(line), styles["meta"]))
        story.append(Spacer(1, 3 * scale))

    pending: list[str] = []

    def flush():
        if not pending:
            return
        for item in pending:
            story.append(Paragraph(item, styles["bullet"], bulletText="\u2022"))
        story.append(Spacer(1, 2.5 * scale))
        pending.clear()

    for raw in rest:
        line = raw.rstrip()

        if not line.strip():
            flush()
            continue

        # A horizontal rule of its own is redundant here: section headings
        # already carry one, and a second would just add noise.
        if _RULE.match(line):
            flush()
            continue

        heading = _HEADING.match(line)
        if heading or _looks_like_heading(line):
            flush()
            label = heading.group(2) if heading else line
            story.append(Paragraph(_inline(label).upper(), styles["section"]))
            story.append(
                HRFlowable(
                    width="100%",
                    thickness=0.7,
                    color=_RULE_GREY,
                    spaceBefore=1 * scale,
                    spaceAfter=3.5 * scale,
                )
            )
            continue

        bullet = _BULLET.match(line)
        if bullet:
            pending.append(_inline(_with_label(bullet.group(1))))
            continue

        flush()
        story.append(Paragraph(_inline(_with_label(line)), styles["body"]))

    flush()

    if not story:
        story.append(Paragraph("This resume is empty.", styles["body"]))

    return story


def _render(text: str, title: str, scale: float) -> tuple[bytes, int]:
    """One attempt at the whole document. Returns the bytes and page count."""
    buffer = BytesIO()
    margin = 15 * mm * max(scale, 0.85)
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=margin,
        rightMargin=margin,
        topMargin=13 * mm * max(scale, 0.85),
        bottomMargin=12 * mm * max(scale, 0.85),
        title=title,
        author="cvexpert",
    )
    doc.build(_story(text, scale))
    return buffer.getvalue(), doc.page


def render_resume_pdf(text: str, title: str = "Resume") -> bytes:
    """Lay the text out as a PDF and return the bytes.

    Rendered at full size first, then progressively tighter until it fits on
    one page. Long resumes bottom out at MIN_SCALE and run to a second page,
    which is the right trade: unreadable beats nothing, but not by much.
    """
    for step in range(_STEPS):
        scale = 1.0 - (1.0 - MIN_SCALE) * (step / (_STEPS - 1))
        document, pages = _render(text, title, scale)
        if pages <= 1:
            return document
    return document
