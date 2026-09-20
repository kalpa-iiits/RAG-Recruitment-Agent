"""Render a generated resume into a PDF.

The rewrite endpoints return plain text that the model was asked to format
"in a modern, clean style with clear section headings", which in practice
means light markdown: `#` headings, `**bold**`, and `-`/`*` bullets. This
turns that into something a person can actually send to an employer.

Deliberately plain: one column, one typeface family, generous margins. The
point is a legible document, not a design.
"""

import html
import re
from io import BytesIO

from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    HRFlowable,
    ListFlowable,
    ListItem,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
)

_BULLET = re.compile(r"^\s*[-*•]\s+(.*)$")
_HEADING = re.compile(r"^\s*(#{1,6})\s+(.*)$")
_RULE = re.compile(r"^\s*([-*_])\1{2,}\s*$")
_BOLD = re.compile(r"\*\*(.+?)\*\*")
_ITALIC = re.compile(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)")


def _styles() -> dict:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "ResumeTitle", parent=base["Title"], fontSize=18, leading=22, spaceAfter=2
        ),
        "h1": ParagraphStyle(
            "ResumeH1",
            parent=base["Heading1"],
            fontSize=12.5,
            leading=15,
            spaceBefore=12,
            spaceAfter=3,
            textColor="#1a1a1a",
        ),
        "h2": ParagraphStyle(
            "ResumeH2",
            parent=base["Heading2"],
            fontSize=11,
            leading=14,
            spaceBefore=9,
            spaceAfter=2,
            textColor="#333333",
        ),
        "body": ParagraphStyle(
            "ResumeBody",
            parent=base["BodyText"],
            fontSize=9.6,
            leading=13.5,
            spaceAfter=4,
            alignment=TA_JUSTIFY,
        ),
        "bullet": ParagraphStyle(
            "ResumeBullet",
            parent=base["BodyText"],
            fontSize=9.6,
            leading=13.5,
            spaceAfter=2,
        ),
    }


def _inline(text: str) -> str:
    """Markdown emphasis to reportlab's mini-HTML, escaping the rest."""
    escaped = html.escape(text.strip())
    escaped = _BOLD.sub(r"<b>\1</b>", escaped)
    escaped = _ITALIC.sub(r"<i>\1</i>", escaped)
    return escaped


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


def render_resume_pdf(text: str, title: str = "Resume") -> bytes:
    """Lay the text out as a PDF and return the bytes."""
    styles = _styles()
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title=title,
        author="cvexpert",
    )

    story: list = []
    pending_bullets: list[str] = []

    def flush_bullets():
        if not pending_bullets:
            return
        story.append(
            ListFlowable(
                [
                    ListItem(Paragraph(item, styles["bullet"]), leftIndent=10)
                    for item in pending_bullets
                ],
                bulletType="bullet",
                bulletFontSize=6,
                leftIndent=12,
                start="circle",
            )
        )
        story.append(Spacer(1, 4))
        pending_bullets.clear()

    for raw_line in (text or "").splitlines():
        line = raw_line.rstrip()

        if not line.strip():
            flush_bullets()
            continue

        if _RULE.match(line):
            flush_bullets()
            story.append(Spacer(1, 3))
            story.append(HRFlowable(width="100%", thickness=0.6, color="#cccccc"))
            story.append(Spacer(1, 5))
            continue

        heading = _HEADING.match(line)
        if heading:
            flush_bullets()
            level = len(heading.group(1))
            style = styles["h1"] if level <= 2 else styles["h2"]
            story.append(Paragraph(_inline(heading.group(2)), style))
            continue

        bullet = _BULLET.match(line)
        if bullet:
            pending_bullets.append(_inline(bullet.group(1)))
            continue

        flush_bullets()
        if _looks_like_heading(line):
            story.append(Paragraph(_inline(line), styles["h1"]))
        else:
            story.append(Paragraph(_inline(line), styles["body"]))

    flush_bullets()

    if not story:
        story.append(Paragraph("This resume is empty.", styles["body"]))

    doc.build(story)
    return buffer.getvalue()
