import os
import re

import matplotlib
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, Paragraph, Spacer, Table, TableStyle

_font_dir = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
pdfmetrics.registerFont(TTFont("DejaVu", os.path.join(_font_dir, "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DejaVu-Bold", os.path.join(_font_dir, "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFont(TTFont("DejaVu-Italic", os.path.join(_font_dir, "DejaVuSans-Oblique.ttf")))
pdfmetrics.registerFontFamily("DejaVu", normal="DejaVu", bold="DejaVu-Bold", italic="DejaVu-Italic",
                              boldItalic="DejaVu-Bold")

PAGE_W = A4[0] - 3.6 * cm

body = ParagraphStyle("body", fontName="DejaVu", fontSize=9.5, leading=13.5, alignment=TA_JUSTIFY, spaceAfter=5)
small = ParagraphStyle("small", parent=body, fontSize=7.5, leading=9.5, alignment=0, spaceAfter=0)
caption = ParagraphStyle("caption", parent=body, fontSize=8, leading=10, alignment=1, textColor=colors.grey)
h1 = ParagraphStyle("h1", fontName="DejaVu-Bold", fontSize=14, leading=18, spaceBefore=10, spaceAfter=6)
h2 = ParagraphStyle("h2", fontName="DejaVu-Bold", fontSize=11.5, leading=15, spaceBefore=8, spaceAfter=4)
title = ParagraphStyle("title", fontName="DejaVu-Bold", fontSize=17, leading=22, alignment=1, spaceAfter=8)
center = ParagraphStyle("center", parent=body, alignment=1)


def nbsp_numbers(text):
    return re.sub(r"(?<=\d) (?=\d{3}\b)", "\u00a0", text)


def P(text, style=body):
    return Paragraph(nbsp_numbers(text), style)


def bullets(items):
    st = ParagraphStyle("li", parent=body, alignment=0, leftIndent=12, firstLineIndent=-9, spaceAfter=2)
    return [Paragraph("•&nbsp;&nbsp;" + nbsp_numbers(t), st) for t in items]


def fig(path, width=1.0, cap=None):
    w, h = PILImage.open(path).size
    out = [Image(path, width=PAGE_W * width, height=PAGE_W * width * h / w)]
    if cap:
        out.append(Paragraph(cap, caption))
    out.append(Spacer(1, 6))
    return out


def auto_widths(rows, font_size, pad=8):
    need, full = [], []
    for j in range(len(rows[0])):
        words, lines = [0], [0]
        for i, r in enumerate(rows):
            font = "DejaVu-Bold" if i == 0 else "DejaVu"
            text = nbsp_numbers(str(r[j]))
            lines += [stringWidth(x, font, font_size) for x in text.split("<br/>")]
            words += [stringWidth(w, font, font_size) for w in text.replace("<br/>", " ").split()]
        need.append(max(words) + pad)
        full.append(max(lines) + pad)
    if sum(full) <= PAGE_W:
        extra = (PAGE_W - sum(full)) / len(full)
        return [f + extra for f in full]
    extra = PAGE_W - sum(need)
    want = [f - n for f, n in zip(full, need)]
    return [n + extra * w / sum(want) for n, w in zip(need, want)]


def table(rows, col_widths=None, header=True, font_size=7.5):
    st = ParagraphStyle("cell", parent=small, fontSize=font_size, leading=font_size + 2)
    sth = ParagraphStyle("cellh", parent=st, fontName="DejaVu-Bold")
    data = []
    for i, r in enumerate(rows):
        data.append([Paragraph(nbsp_numbers(str(x)), sth if header and i == 0 else st) for x in r])
    if col_widths:
        col_widths = [PAGE_W * w for w in col_widths]
    else:
        col_widths = auto_widths(rows, font_size)
    t = Table(data, colWidths=col_widths, repeatRows=1 if header else 0)
    style = [
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
    ]
    if header:
        style.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8e8e8")))
    t.setStyle(TableStyle(style))
    return [t, Spacer(1, 8)]
