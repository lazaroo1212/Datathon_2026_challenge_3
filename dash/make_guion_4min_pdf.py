"""make_guion_4min_pdf.py — genera dash/Guion_Datathon_UPC_Yukiyuk_4min.pdf
desde dash/guion_4min.md (versión corregida: heatmaps simultáneos + contexto
Bioinformática + cierre del equipo).

Diseño: 1 página por hablante con banda de color (Jan azul, Pei verde, Elayne naranja,
David morado). Reset de X antes de cada render (fpdf2 deja el cursor a la derecha tras
multi_cell).

Uso:  .venv/bin/python dash/make_guion_4min_pdf.py
"""
from pathlib import Path

from fpdf import FPDF

HERE = Path(__file__).resolve().parent
MD = HERE / "guion_4min.md"
OUT = HERE / "Guion_Datathon_UPC_Yukiyuk_4min.pdf"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
LM = 18

PAGINAS = [
    ("JAN", "0:00–1:00", "Pestaña 1 · ¿Comprar la batería?", (31, 78, 121), (222, 235, 247)),
    ("PEI", "1:00–2:00", "Pestaña 2 · ¿Cómo opera?", (14, 100, 66), (222, 240, 229)),
    ("ELAYNE", "2:00–3:00", "Pestañas 3 y 4 · Trade-off y Hard Mode", (166, 74, 0), (253, 235, 208)),
    ("DAVID", "3:00–4:00", "Pestañas 5 y 6 · Evidencia y método", (74, 44, 114), (237, 226, 247)),
]

md = MD.read_text(encoding="utf-8")
secciones = []
bloque = None
for ln in md.splitlines():
    if ln.startswith("## "):
        if bloque:
            secciones.append(bloque)
        bloque = {"titulo": ln[3:].strip(), "contenido": []}
    elif bloque is not None and ln.strip():
        bloque["contenido"].append(ln)
if bloque:
    secciones.append(bloque)


class Guion(FPDF):
    def footer(self):
        self.set_y(-12)
        self.set_font("dv", "", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 8, f"Yukiyuk · página {self.page_no()}/4 — consejo: ensayar con cronómetro "
                        "y cambiar de pestaña mientras habla la persona asignada", align="C")


pdf = Guion(orientation="P", unit="mm", format="A4")
pdf.add_font("dv", "", FONT)
pdf.add_font("dv", "B", FONT_B)
pdf.set_margins(LM, 14, LM)
pdf.set_auto_page_break(True, margin=18)

for (hablante, tiempo, pestaña, color_banda, color_fondo), sec in zip(PAGINAS, secciones):
    pdf.add_page()
    # banda de color con hablante y tiempo
    pdf.set_fill_color(*color_banda)
    pdf.rect(0, 0, 210, 26, "F")
    pdf.set_xy(LM, 5)
    pdf.set_font("dv", "B", 16)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 9, hablante)
    pdf.set_xy(LM, 15)
    pdf.set_font("dv", "", 11)
    pdf.cell(0, 7, f"{tiempo}  ·  {pestaña}")
    pdf.set_text_color(*NEGRO)
    pdf.set_y(34)

    for ln in sec["contenido"]:
        if ln.startswith("EN PANTALLA:"):
            reset = lambda: pdf.set_x(LM)
            reset()
            pdf.set_font("dv", "B", 9.5)
            pdf.set_text_color(*color_banda)
            pdf.multi_cell(0, 5, ln)
            pdf.ln(1)
        elif ln.startswith("«"):
            reset = lambda: pdf.set_x(LM)
            reset()
            pdf.set_fill_color(*color_fondo)
            pdf.set_draw_color(*color_banda)
            pdf.set_left_margin(LM + 4)
            pdf.set_font("dv", "", 10)
            pdf.multi_cell(0, 5.6, ln.strip("«»"), border=1, fill=True,
                           new_x="LMARGIN", new_y="NEXT")
            pdf.set_left_margin(LM)
            pdf.ln(2)
        else:
            reset = lambda: pdf.set_x(LM)
            reset()
            pdf.set_font("dv", "", 9.5)
            pdf.set_text_color(90, 90, 90)
            pdf.multi_cell(0, 5, ln)
            pdf.set_text_color(*NEGRO)

pdf.output(OUT)
print(f"PDF generado: {OUT} ({OUT.stat().st_size/1024:.0f} KB, {pdf.page_no()} páginas)")
