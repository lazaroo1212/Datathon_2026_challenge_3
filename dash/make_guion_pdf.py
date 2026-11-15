"""make_guion_pdf.py — genera dash/guion_pitch.pdf desde dash/guion_pitch.md (v3).

Fix importante: fpdf2's multi_cell deja el cursor en el borde derecho; antes de cada
bloque de render hay que resetear X al margen izquierdo (reset_x()), o el siguiente
multi_cell(0) falla con "Not enough horizontal space".

Uso:  .venv/bin/python dash/make_guion_pdf.py
"""
import re
from pathlib import Path

from fpdf import FPDF

HERE = Path(__file__).resolve().parent
MD = HERE / "guion_pitch.md"
OUT = HERE / "guion_pitch.pdf"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

AZUL = (31, 78, 121)
AZUL_CLARO = (222, 235, 247)
VERDE_CLARO = (226, 240, 229)
NEGRO = (40, 40, 40)
LM = 18


class Guion(FPDF):
    def header(self):
        if self.page_no() > 1:
            self.set_font("dv", "", 8)
            self.set_text_color(130, 130, 130)
            self.cell(0, 6, "Yukiyuk · Battery Ops Cockpit · Guion del pitch (v3)", align="R")
            self.ln(10)

    def footer(self):
        self.set_y(-12)
        self.set_font("dv", "", 8)
        self.set_text_color(130, 130, 130)
        self.cell(0, 8, f"Página {self.page_no()}/{{nb}}", align="C")


pdf = Guion(orientation="P", unit="mm", format="A4")
pdf.add_font("dv", "", FONT)
pdf.add_font("dv", "B", FONT_B)
pdf.set_margins(LM, 16, LM)
pdf.set_auto_page_break(True, margin=18)


def reset_x():
    pdf.set_x(LM)


pdf.add_page()
pdf.set_fill_color(*AZUL)
pdf.set_text_color(255, 255, 255)
pdf.rect(0, 0, 210, 34, "F")
pdf.set_xy(LM, 8)
pdf.set_font("dv", "B", 19)
pdf.cell(0, 10, "Battery Ops Cockpit — Guion definitivo del pitch")
pdf.set_xy(LM, 19)
pdf.set_font("dv", "", 11)
pdf.cell(0, 8, "Equipo Yukiyuk · 4 estudiantes de Bioinformática (UPC) · Datathon 2026 · 3 min 30 s")
pdf.set_text_color(*NEGRO)
pdf.set_y(43)
reset_x()

md = MD.read_text(encoding="utf-8")
lines = md.splitlines()
i = 0
while i < len(lines):
    ln = lines[i]

    # ── cabeceras ## ──
    if ln.startswith("## "):
        pdf.ln(2)
        reset_x()
        pdf.set_font("dv", "B", 13)
        pdf.set_text_color(*AZUL)
        pdf.multi_cell(0, 7, ln[3:].strip())
        reset_x()
        pdf.set_draw_color(*AZUL)
        pdf.set_line_width(0.4)
        pdf.line(LM, pdf.get_y(), 192, pdf.get_y())
        pdf.ln(2)
        pdf.set_text_color(*NEGRO)

    # ── subtítulo en cursiva (Equipo Yukiyuk · ...) ──
    elif ln.startswith("**Equipo Yukiyuk"):
        reset_x()
        pdf.set_font("dv", "", 10)
        pdf.set_text_color(90, 90, 90)
        pdf.multi_cell(0, 5, ln.strip("*"))
        pdf.set_text_color(*NEGRO)

    # ── tablas markdown ──
    elif ln.startswith("|"):
        tabla = []
        while i < len(lines) and lines[i].startswith("|"):
            if not re.match(r"^\|[\s\-|]+\|$", lines[i]):
                celdas = [c.strip().strip("*") for c in lines[i].strip("|").split("|")]
                tabla.append(celdas)
            i += 1
        i -= 1
        reset_x()
        if tabla:
            ncol = max(len(r) for r in tabla)
            widths = [174 / ncol] * ncol
            pdf.set_font("dv", "B", 8.5)
            pdf.set_fill_color(*AZUL_CLARO)
            pdf.set_text_color(*AZUL)
            for r in tabla[:1]:
                for c, w in zip(r, widths):
                    pdf.cell(w, 6, c[:60], border=0.2, fill=True)
                pdf.ln()
                reset_x()
            pdf.set_font("dv", "", 8.5)
            pdf.set_text_color(*NEGRO)
            for r in tabla[1:]:
                for c, w in zip(r, widths):
                    pdf.cell(w, 5.5, c[:60], border=0.2)
                pdf.ln()
                reset_x()
            pdf.ln(2)

    # ── citas literales del pitch (caja verde) ──
    elif ln.startswith("> "):
        texto = [ln[2:]]
        while i + 1 < len(lines) and lines[i + 1].startswith("> "):
            i += 1
            texto.append(lines[i][2:])
        reset_x()
        pdf.set_fill_color(*VERDE_CLARO)
        pdf.set_draw_color(120, 180, 120)
        pdf.set_left_margin(22)
        pdf.set_font("dv", "", 9.5)
        pdf.multi_cell(0, 5.2, " ".join(texto).replace("**", "").replace("*", ""),
                       border=1, fill=True, new_x="LMARGIN", new_y="NEXT")
        pdf.set_left_margin(LM)
        pdf.ln(1.5)

    # ── listas ──
    elif ln.startswith("- "):
        reset_x()
        pdf.set_font("dv", "", 9.5)
        pdf.set_text_color(*NEGRO)
        pdf.cell(4, 5.2, "•")
        pdf.multi_cell(0, 5.2, ln[2:].replace("**", ""))
    elif re.match(r"^\d+\. ", ln):
        num, resto = ln.split(". ", 1)
        reset_x()
        pdf.set_font("dv", "", 9.5)
        pdf.set_text_color(*NEGRO)
        pdf.cell(6, 5.2, f"{num}.")
        pdf.multi_cell(0, 5.2, resto.replace("**", ""))

    # ── texto normal ──
    elif ln.strip() and not ln.startswith("---"):
        reset_x()
        pdf.set_font("dv", "", 9.5)
        pdf.set_text_color(*NEGRO)
        pdf.multi_cell(0, 5.2, ln.replace("**", ""))
    elif ln.startswith("---"):
        pdf.ln(1)

    i += 1

pdf.output(OUT)
print(f"PDF generado: {OUT} ({OUT.stat().st_size/1024:.0f} KB, {pdf.page_no()} páginas)")
