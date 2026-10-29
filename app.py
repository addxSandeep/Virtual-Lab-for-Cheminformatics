import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from html import escape

from rdkit import Chem
from rdkit.Chem import Draw, AllChem, Descriptors, Crippen, Lipinski, rdMolDescriptors
import py3Dmol
import streamlit.components.v1 as components
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.units import inch

st.set_page_config(
    page_title="Computational Chemistry Virtual Lab",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
/* ==========================================================================
   PREMIUM LIQUID GLASS & GLASSMORPHISM UI - CHEMINFORMATICS SUITE
   ========================================================================== */

@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    color: #f1f5f9;
}

/* Ambient Liquid Chemical Canvas Background */
.stApp {
    background-color: #060911;
    background-image: 
        radial-gradient(circle at 10% 12%, rgba(6, 182, 212, 0.18) 0%, transparent 40%),
        radial-gradient(circle at 88% 18%, rgba(139, 92, 246, 0.22) 0%, transparent 45%),
        radial-gradient(circle at 50% 50%, rgba(14, 165, 233, 0.12) 0%, transparent 55%),
        radial-gradient(circle at 82% 85%, rgba(236, 72, 153, 0.15) 0%, transparent 42%),
        radial-gradient(circle at 18% 88%, rgba(16, 185, 129, 0.14) 0%, transparent 40%);
    background-attachment: fixed;
    background-size: cover;
}

/* Translucent Frosted Glass Sidebar */
section[data-testid="stSidebar"] {
    background: rgba(9, 13, 24, 0.75) !important;
    backdrop-filter: blur(28px) saturate(200%) !important;
    -webkit-backdrop-filter: blur(28px) saturate(200%) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
    box-shadow: 8px 0 32px rgba(0, 0, 0, 0.5) !important;
}

/* Sidebar Branding Badge */
.sidebar-brand {
    background: linear-gradient(135deg, rgba(6, 182, 212, 0.15) 0%, rgba(139, 92, 246, 0.15) 100%);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-top: 1px solid rgba(255, 255, 255, 0.3);
    padding: 16px 18px;
    border-radius: 16px;
    margin-bottom: 20px;
    box-shadow: 0 8px 24px -4px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.2);
}

/* Liquid Glass Hero Header */
.liquid-hero {
    position: relative;
    background: linear-gradient(135deg, rgba(15, 23, 42, 0.75) 0%, rgba(30, 41, 59, 0.45) 100%);
    backdrop-filter: blur(28px) saturate(200%);
    -webkit-backdrop-filter: blur(28px) saturate(200%);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-top: 1px solid rgba(255, 255, 255, 0.38);
    border-radius: 22px;
    padding: 2.6rem 3rem;
    margin-bottom: 2rem;
    box-shadow: 0 20px 48px -12px rgba(0, 0, 0, 0.65), inset 0 1px 1px rgba(255, 255, 255, 0.25);
    overflow: hidden;
    text-align: center;
}
.liquid-hero::before {
    content: "";
    position: absolute;
    top: -40%;
    left: 25%;
    width: 50%;
    height: 100%;
    background: radial-gradient(circle, rgba(6, 182, 212, 0.25) 0%, transparent 70%);
    pointer-events: none;
    filter: blur(50px);
}
.glass-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: linear-gradient(135deg, rgba(6, 182, 212, 0.2) 0%, rgba(139, 92, 246, 0.2) 100%);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(34, 211, 238, 0.45);
    border-top: 1px solid rgba(255, 255, 255, 0.5);
    color: #67e8f9;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 6px 18px;
    border-radius: 9999px;
    margin-bottom: 1.1rem;
    box-shadow: 0 0 20px rgba(6, 182, 212, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.3);
}
.hero-title-liquid {
    font-size: 2.55rem;
    font-weight: 800;
    letter-spacing: -0.025em;
    margin: 0;
    background: linear-gradient(135deg, #ffffff 15%, #e2e8f0 60%, #38bdf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-shadow: 0 2px 16px rgba(0, 0, 0, 0.4);
}
.hero-subtitle-liquid {
    font-size: 1.08rem;
    color: #94a3b8;
    margin-top: 0.8rem;
    font-weight: 400;
    letter-spacing: 0.01em;
}

/* Glassmorphic Section Headers */
.section-header-liquid {
    font-size: 1.6rem;
    font-weight: 700;
    color: #f8fafc;
    letter-spacing: -0.02em;
    display: flex;
    align-items: center;
    gap: 10px;
    margin-top: 1.5rem;
    margin-bottom: 1rem;
}
.section-header-liquid span.icon-glow {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 38px;
    height: 38px;
    border-radius: 10px;
    background: linear-gradient(135deg, rgba(6, 182, 212, 0.2) 0%, rgba(139, 92, 246, 0.2) 100%);
    border: 1px solid rgba(255, 255, 255, 0.15);
    box-shadow: 0 0 14px rgba(6, 182, 212, 0.25);
    font-size: 1.1rem;
}

/* Layered Frosted Glass Panels */
.glass-panel {
    background: linear-gradient(135deg, rgba(15, 23, 42, 0.65) 0%, rgba(15, 23, 42, 0.45) 100%);
    backdrop-filter: blur(24px) saturate(190%);
    -webkit-backdrop-filter: blur(24px) saturate(190%);
    border: 1px solid rgba(255, 255, 255, 0.09);
    border-top: 1px solid rgba(255, 255, 255, 0.25);
    border-radius: 18px;
    padding: 24px 28px;
    margin-bottom: 1.6rem;
    box-shadow: 0 16px 40px -8px rgba(0, 0, 0, 0.55), inset 0 1px 0 rgba(255, 255, 255, 0.15);
}

/* Metric Cards - Refractive Liquid Glass */
div[data-testid="stMetric"] {
    background: linear-gradient(135deg, rgba(15, 23, 42, 0.72) 0%, rgba(30, 41, 59, 0.42) 100%) !important;
    backdrop-filter: blur(24px) saturate(200%) !important;
    -webkit-backdrop-filter: blur(24px) saturate(200%) !important;
    border: 1px solid rgba(255, 255, 255, 0.10) !important;
    border-top: 2px solid rgba(6, 182, 212, 0.7) !important;
    padding: 18px 22px !important;
    border-radius: 16px !important;
    box-shadow: 0 14px 34px -6px rgba(0, 0, 0, 0.6), inset 0 1px 0 rgba(255, 255, 255, 0.2) !important;
    transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
}
div[data-testid="stMetric"]:hover {
    transform: translateY(-3px) scale(1.01) !important;
    border-top: 2px solid rgba(34, 211, 238, 1) !important;
    box-shadow: 0 20px 42px -6px rgba(6, 182, 212, 0.35), inset 0 1px 1px rgba(255, 255, 255, 0.35) !important;
}
div[data-testid="stMetricLabel"] {
    color: #94a3b8 !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
}
div[data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-weight: 800 !important;
    font-size: 1.75rem !important;
    text-shadow: 0 0 18px rgba(6, 182, 212, 0.45) !important;
}

/* Liquid Neon Pills & Tags */
.neon-pill {
    display: inline-flex;
    align-items: center;
    background: linear-gradient(135deg, rgba(6, 182, 212, 0.15) 0%, rgba(59, 130, 246, 0.15) 100%);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(34, 211, 238, 0.35);
    border-top: 1px solid rgba(255, 255, 255, 0.4);
    color: #67e8f9;
    padding: 5px 14px;
    border-radius: 9999px;
    font-size: 0.82rem;
    font-weight: 600;
    margin-right: 8px;
    margin-bottom: 8px;
    box-shadow: 0 0 14px rgba(6, 182, 212, 0.18), inset 0 1px 0 rgba(255, 255, 255, 0.2);
}

/* Liquid Glass Buttons */
.stButton > button, div[data-testid="stFormSubmitButton"] > button {
    background: linear-gradient(135deg, rgba(6, 182, 212, 0.9) 0%, rgba(59, 130, 246, 0.85) 50%, rgba(139, 92, 246, 0.85) 100%) !important;
    color: #ffffff !important;
    border: 1px solid rgba(255, 255, 255, 0.35) !important;
    border-top: 1px solid rgba(255, 255, 255, 0.75) !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em !important;
    padding: 0.6rem 1.6rem !important;
    backdrop-filter: blur(16px) !important;
    -webkit-backdrop-filter: blur(16px) !important;
    box-shadow: 0 8px 24px rgba(6, 182, 212, 0.4), inset 0 1px 1px rgba(255, 255, 255, 0.5) !important;
    transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
}
.stButton > button:hover, div[data-testid="stFormSubmitButton"] > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 12px 32px rgba(6, 182, 212, 0.6), inset 0 1px 2px rgba(255, 255, 255, 0.7) !important;
    border-color: rgba(255, 255, 255, 0.8) !important;
}

/* Download Buttons */
.stDownloadButton > button {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.85) 0%, rgba(6, 182, 212, 0.85) 100%) !important;
    color: #ffffff !important;
    border: 1px solid rgba(255, 255, 255, 0.35) !important;
    border-top: 1px solid rgba(255, 255, 255, 0.7) !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    box-shadow: 0 8px 24px rgba(16, 185, 129, 0.35), inset 0 1px 1px rgba(255, 255, 255, 0.4) !important;
}
.stDownloadButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 12px 32px rgba(16, 185, 129, 0.55), inset 0 1px 2px rgba(255, 255, 255, 0.6) !important;
}

/* Glass Viewport for 2D Structural Microscope */
.mol-microscope-card {
    background: rgba(255, 255, 255, 0.96);
    border-radius: 16px;
    padding: 14px;
    box-shadow: 0 16px 36px rgba(0, 0, 0, 0.6), inset 0 0 0 1px rgba(255, 255, 255, 0.9);
}

/* Translucent Inputs & Controls */
input, textarea, div[data-baseweb="select"] > div {
    background: rgba(15, 23, 42, 0.7) !important;
    border: 1px solid rgba(255, 255, 255, 0.14) !important;
    border-radius: 10px !important;
    color: #f1f5f9 !important;
}
input:focus, textarea:focus, div[data-baseweb="select"] > div:focus-within {
    border-color: rgba(34, 211, 238, 0.8) !important;
    box-shadow: 0 0 16px rgba(6, 182, 212, 0.35) !important;
}

/* Translucent Frosted Alerts */
div[data-testid="stAlert"] {
    background: rgba(15, 23, 42, 0.65) !important;
    backdrop-filter: blur(20px) !important;
    -webkit-backdrop-filter: blur(20px) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 14px !important;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4) !important;
}

/* Dataframe Clean Glass Frame */
div[data-testid="stDataFrame"] {
    background: rgba(15, 23, 42, 0.6) !important;
    backdrop-filter: blur(20px) !important;
    -webkit-backdrop-filter: blur(20px) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 14px !important;
    overflow: hidden !important;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4) !important;
}
</style>
""", unsafe_allow_html=True)

EXAMPLES = {
    "Methane": "C",
    "Toluene": "Cc1ccccc1",
    "Ibuprofen": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "Paracetamol": "CC(=O)Nc1ccc(O)cc1",
    "Nicotine": "CN1CCCC1c1cccnc1",
    "Glucose": "OCC1OC(O)C(O)C(O)C1O"
}

if "active_smiles" not in st.session_state:
    st.session_state.active_smiles = "Cc1ccccc1"
if "student_name" not in st.session_state:
    st.session_state.student_name = ""
if "registration_number" not in st.session_state:
    st.session_state.registration_number = ""
if "assessment_result" not in st.session_state:
    st.session_state.assessment_result = None

def parse_smiles(smiles):
    """Return an RDKit molecule or None for empty/invalid SMILES."""
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    try:
        mol = Chem.MolFromSmiles(smiles.strip())
        return mol
    except Exception:
        return None

def calculate_bond_information(mol):
    """Count sigma bonds including H; count pi components from a Kekule form."""
    heavy_atom_bonds = mol.GetNumBonds()
    mol_h = Chem.AddHs(Chem.Mol(mol))
    sigma_bonds = mol_h.GetNumBonds()
    pi_bonds = 0
    kekule = Chem.Mol(mol)
    try:
        Chem.Kekulize(kekule, clearAromaticFlags=True)
        for bond in kekule.GetBonds():
            bt = bond.GetBondType()
            if bt == Chem.BondType.DOUBLE:
                pi_bonds += 1
            elif bt == Chem.BondType.TRIPLE:
                pi_bonds += 2
    except Exception:
        pi_bonds = sum(0.5 for b in mol.GetBonds() if b.GetIsAromatic())
    return {
        "Heavy-atom bonds": heavy_atom_bonds,
        "Sigma bonds (including H)": sigma_bonds,
        "Pi-bond components": pi_bonds,
        "Sigma + pi components": sigma_bonds + pi_bonds,
        "Atoms including H": mol_h.GetNumAtoms()
    }

def calculate_properties(mol):
    bonds = calculate_bond_information(mol)
    props = {
        "Molecular formula": rdMolDescriptors.CalcMolFormula(mol),
        "Molecular weight (g/mol)": round(Descriptors.MolWt(mol), 4),
        "Exact molecular weight (Da)": round(Descriptors.ExactMolWt(mol), 5),
        "Crippen LogP": round(Crippen.MolLogP(mol), 4),
        "TPSA (Å²)": round(rdMolDescriptors.CalcTPSA(mol), 3),
        "H-bond donors": int(Lipinski.NumHDonors(mol)),
        "H-bond acceptors": int(Lipinski.NumHAcceptors(mol)),
        "Rotatable bonds": int(Lipinski.NumRotatableBonds(mol)),
        "Ring count (RDKit)": int(Lipinski.RingCount(mol)),
        "Aromatic rings": int(rdMolDescriptors.CalcNumAromaticRings(mol)),
        "Aliphatic rings": int(rdMolDescriptors.CalcNumAliphaticRings(mol)),
        "Heavy atoms": int(mol.GetNumHeavyAtoms()),
        "Atoms in input graph": int(mol.GetNumAtoms()),
        "Fraction Csp3": round(rdMolDescriptors.CalcFractionCSP3(mol), 4),
        "Molar refractivity (Crippen)": round(Crippen.MolMR(mol), 4),
    }
    props.update(bonds)
    return props

def make_3d(mol):
    mol3d = Chem.AddHs(Chem.Mol(mol))
    params = AllChem.ETKDGv3()
    params.randomSeed = 2026
    status = AllChem.EmbedMolecule(mol3d, params)
    if status != 0:
        return None, "RDKit could not generate a 3D conformer for this structure."
    try:
        result = AllChem.MMFFOptimizeMolecule(mol3d, maxIters=500)
        method = "MMFF"
        if result < 0:
            raise ValueError("MMFF optimization unavailable")
    except Exception:
        try:
            AllChem.UFFOptimizeMolecule(mol3d, maxIters=500)
            method = "UFF"
        except Exception:
            method = "unoptimized embedded conformer"
    return mol3d, method

def show_3d(mol3d, style, show_surface):
    view = py3Dmol.view(width=850, height=520)
    view.addModel(Chem.MolToMolBlock(mol3d), "mol")
    if style == "Ball and stick":
        view.setStyle({"stick": {"radius": 0.16}, "sphere": {"scale": 0.28}})
    elif style == "Stick":
        view.setStyle({"stick": {"radius": 0.20}})
    elif style == "Space filling":
        view.setStyle({"sphere": {"scale": 0.92}})
    elif style == "Wireframe":
        view.setStyle({"line": {"linewidth": 2.2}})
    if show_surface:
        view.addSurface(py3Dmol.VDW, {"opacity": 0.40}, {"color": "#38bdf8"})
    # Sleek dark liquid canvas background
    view.setBackgroundColor("#080c16")
    view.zoomTo()
    components.html(view._make_html(), height=540, scrolling=False)

def make_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")

def create_pdf(student_name, reg_no, smiles, mol, props, notes=""):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4,
                            leftMargin=36, rightMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("ReportTitle", parent=styles["Heading1"],
                                 fontSize=18, leading=22, textColor=colors.HexColor("#17365D"),
                                 alignment=TA_CENTER)
    subtitle_style = ParagraphStyle("ReportSubTitle", parent=styles["Normal"],
                                    fontSize=10, leading=13, textColor=colors.HexColor("#4A607A"),
                                    alignment=TA_CENTER)
    heading = ParagraphStyle("SecHeading", parent=styles["Heading2"],
                             fontSize=12, leading=15, textColor=colors.HexColor("#17365D"))
    body = styles["Normal"]
    story = [
        Paragraph("Computational Chemistry Virtual Laboratory", title_style),
        Paragraph("Student Practical Report", subtitle_style),
        Spacer(1, 10),
    ]
    meta_data = [
        [Paragraph("<b>Student Name</b>", body), Paragraph(escape(student_name or "Not provided"), body),
         Paragraph("<b>Date</b>", body), Paragraph(pd.Timestamp.now().strftime("%Y-%m-%d"), body)],
        [Paragraph("<b>Registration No.</b>", body), Paragraph(escape(reg_no or "Not provided"), body),
         Paragraph("<b>Formula</b>", body), Paragraph(escape(str(props.get("Molecular formula", ""))), body)],
        [Paragraph("<b>Input SMILES</b>", body), Paragraph(escape(smiles), body),
         Paragraph("<b>Canonical SMILES</b>", body), Paragraph(escape(Chem.MolToSmiles(mol)), body)],
    ]
    meta_table = Table(meta_data, colWidths=[1.3 * inch, 2.3 * inch, 1.3 * inch, 2.3 * inch])
    meta_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F4F7FB")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#DCE5EF")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E5ECF4")),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story += [meta_table, Spacer(1, 10)]

    try:
        img_buffer = BytesIO()
        img = Draw.MolToImage(mol, size=(320, 220))
        img.save(img_buffer, format="PNG")
        img_buffer.seek(0)
        rl_img = Image(img_buffer, width=3.2 * inch, height=2.2 * inch)
    except Exception:
        rl_img = Paragraph("2D structure unavailable", body)

    prop_rows = [
        [Paragraph("<b>Descriptor</b>", body), Paragraph("<b>Value</b>", body)],
        ["Molecular weight", f"{props.get('Molecular weight (g/mol)', 0):.3f} g/mol"],
        ["Exact mass", f"{props.get('Exact molecular weight (Da)', 0):.5f} Da"],
        ["Crippen LogP", str(props.get("Crippen LogP", ""))],
        ["TPSA", f"{props.get('TPSA (Å²)', 0):.2f} Å²"],
        ["H-bond donors / acceptors", f"{props.get('H-bond donors', 0)} / {props.get('H-bond acceptors', 0)}"],
        ["Rotatable bonds", str(props.get("Rotatable bonds", 0))],
        ["Heavy atoms / Rings", f"{props.get('Heavy atoms', 0)} / {props.get('Ring count (RDKit)', 0)}"],
        ["Sigma bonds (incl. H)", str(props.get("Sigma bonds (including H)", 0))],
        ["Pi components", str(props.get("Pi-bond components", 0))],
    ]
    prop_table = Table(prop_rows, colWidths=[2.2 * inch, 1.6 * inch])
    prop_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17365D")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#DCE5EF")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F9FBFE")]),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))

    layout_table = Table([[rl_img, prop_table]], colWidths=[3.4 * inch, 3.8 * inch])
    layout_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
    ]))
    story += [Paragraph("Structure and Key Properties", heading), Spacer(1, 4), layout_table, Spacer(1, 10)]

    all_props = [[Paragraph("<b>Property</b>", body), Paragraph("<b>Computed Value</b>", body)]]
    for k, v in props.items():
        all_props.append([Paragraph(str(k), body), Paragraph(str(v), body)])
    full_table = Table(all_props, colWidths=[3.6 * inch, 3.6 * inch])
    full_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4A607A")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#DCE5EF")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F9FBFE")]),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    story += [Paragraph("Complete Computed Descriptor Set", heading), Spacer(1, 4), full_table]

    if notes.strip():
        story += [Spacer(1, 6), Paragraph("Student interpretation", heading),
                  Paragraph(escape(notes).replace("\n", "<br/>"), body)]
    doc.build(story)
    return buffer.getvalue()

# Sidebar: Liquid Glass Architecture
st.sidebar.markdown("""
<div class="sidebar-brand">
    <div style="display: flex; align-items: center; gap: 10px;">
        <span style="font-size: 1.8rem; filter: drop-shadow(0 0 10px rgba(6, 182, 212, 0.6));">🧪</span>
        <div>
            <div style="font-size: 0.72rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; color: #22d3ee;">VIRTUAL LAB SUITE</div>
            <div style="font-size: 1.15rem; font-weight: 800; color: #ffffff; letter-spacing: -0.01em;">Cheminformatics</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio("Navigation", [
    "🏠 Home", "📘 Theory", "🧬 Molecular Visualization",
    "📊 Molecular Properties", "📈 Structure–Property Comparison",
    "📝 Assessment", "📄 Final Report"
])

st.sidebar.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
st.sidebar.markdown("""
<div class="glass-panel" style="padding: 16px 18px; margin-bottom: 16px;">
    <div style="font-size: 0.78rem; font-weight: 700; color: #67e8f9; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 10px;">👤 Student Profile</div>
""", unsafe_allow_html=True)
st.sidebar.text_input("Student name", key="student_name")
st.sidebar.text_input("Registration number", key="registration_number")
st.sidebar.markdown("</div>", unsafe_allow_html=True)

st.sidebar.markdown("""
<div class="glass-panel" style="padding: 16px 18px;">
    <div style="font-size: 0.78rem; font-weight: 700; color: #a78bfa; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 8px;">🎯 Learning Outcomes</div>
    <div style="color: #94a3b8; font-size: 0.82rem; line-height: 1.6;">
        • SMILES connectivity & validation<br/>
        • 2D/3D conformer modeling<br/>
        • Molecular descriptors & LogP<br/>
        • Structure–property patterns<br/>
        • Rigorous academic reporting
    </div>
</div>
""", unsafe_allow_html=True)

# Main Dashboard Content Pages
if page == "🏠 Home":
    st.markdown("""
    <div class="liquid-hero">
        <div class="glass-badge">
            <span>⚡</span> LIQUID GLASS COMPUTATIONAL PLATFORM
        </div>
        <h1 class="hero-title-liquid">🧪 Computational Chemistry Virtual Laboratory</h1>
        <p class="hero-subtitle-liquid">Interactive simulation: Molecular graphs, 3D conformers, and predictive descriptors</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="section-header-liquid">
        <span class="icon-glow">✨</span> Overview
    </div>
    """, unsafe_allow_html=True)
    st.write("Explore how molecular structures are computationally encoded, visualized in 2D and 3D space, and evaluated using cheminformatics descriptors.")
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Learning modules", "7")
    c2.metric("Example molecules", len(EXAMPLES))
    c3.metric("Core approach", "Structure → properties")
    
    st.markdown("""
    <div class="glass-panel" style="margin-top: 1.6rem;">
        <h4 style="margin-top:0; color: #38bdf8; font-weight: 700; letter-spacing: -0.01em;">📌 Practical Workflow</h4>
        <ul style="color: #cbd5e1; line-height: 1.8; margin-bottom: 0;">
            <li>Select or enter SMILES text strings to parse molecular graphs.</li>
            <li>Render clean 2D topological structures and interactive 3D energy-minimized conformers.</li>
            <li>Calculate physicochemical properties, Lipinski parameters, and topological descriptors.</li>
            <li>Compare molecular cohorts to identify structural and quantitative trends.</li>
            <li>Evaluate comprehension via the knowledge assessment and export an official PDF report.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="section-header-liquid">
        <span class="icon-glow">🧪</span> Active Example Molecules
    </div>
    """, unsafe_allow_html=True)
    st.dataframe(pd.DataFrame([{"Molecule": k, "SMILES": v} for k, v in EXAMPLES.items()]),
                 use_container_width=True, hide_index=True)
    st.info("💡 **Recommended Pathway:** Begin with **Theory**, then proceed through **Molecular Visualization**, **Molecular Properties**, **Comparison**, **Assessment**, and **Final Report**.")
    st.warning("⚠️ **Educational Notice:** Computed descriptors are graph-derived and empirical predictions, serving as pedagogical models rather than direct experimental replacements.")

    st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div class="glass-panel" style="margin-top: 1.5rem; border-left: 3px solid #06b6d4;">
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 14px;">
            <span style="font-size: 1.6rem;">👨‍💻</span>
            <h3 style="margin: 0; color: #ffffff; font-weight: 700; letter-spacing: -0.01em;">Developed by</h3>
        </div>
        <div style="font-size: 1.2rem; font-weight: 700; color: #38bdf8; margin-bottom: 6px;">
            Sandeep Kumar Sunrait
        </div>
        <div style="color: #cbd5e1; font-size: 0.96rem; line-height: 1.8;">
            <span style="color: #94a3b8; font-weight: 600;">Registration Number:</span> <span class="neon-pill" style="margin: 0; padding: 2px 10px; font-size: 0.8rem;">RA2611026050327</span><br/>
            <span style="color: #94a3b8; font-weight: 600;">Program:</span> B.Tech in Computer Science and Engineering (AIML)<br/>
            <span style="color: #94a3b8; font-weight: 600;">Institution:</span> SRM Institute of Science and Technology, Tiruchirappalli
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.caption("🧪 Cheminformatics Virtual Laboratory | Developed using Python, Streamlit and RDKit")

elif page == "📘 Theory":
    st.markdown("""
    <div class="section-header-liquid">
        <span class="icon-glow">📘</span> Theory: Introduction to Cheminformatics
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="glass-panel">
        <h4 style="margin-top:0; color: #38bdf8; font-weight: 700;">What is Cheminformatics?</h4>
        <p style="color: #cbd5e1; margin-bottom: 0; line-height: 1.7;">
            Cheminformatics applies computational and information-science methods to represent, store, analyze, and compare chemical structures and related data across pharmacology, biochemistry, and materials research.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("SMILES notation")
    st.write("SMILES (Simplified Molecular Input Line Entry System) represents molecular connectivity as a text string. Atom symbols represent atoms, parentheses represent branches, digits close rings, and bond symbols can indicate bond types.")
    st.dataframe(pd.DataFrame([
        ["Methane", "C", "Simplest alkane with one carbon atom"],
        ["Toluene", "Cc1ccccc1", "Methyl group attached to an aromatic ring"],
        ["Paracetamol", "CC(=O)Nc1ccc(O)cc1", "Aromatic amide and phenol"],
        ["Glucose", "OCC1OC(O)C(O)C(O)C1O", "Polyhydroxylated six-membered ring"],
        ["Ibuprofen", EXAMPLES["Ibuprofen"], "Aromatic ring with a carboxylic acid"],
    ], columns=["Molecule", "Example SMILES", "Structural note"]), use_container_width=True, hide_index=True)
    
    st.subheader("Key descriptors")
    st.markdown("""
    - **Molecular weight:** average molecular mass calculated from atomic weights.
    - **Exact molecular weight:** monoisotopic mass based on the most abundant isotopes.
    - **Crippen LogP:** estimated octanol/water partition coefficient for the represented neutral structure.
    - **TPSA:** topological polar surface area calculated from molecular fragments.
    - **H-bond donors/acceptors:** counts under RDKit's implemented conventions.
    - **Rotatable bonds:** a rule-based estimate of conformational flexibility.
    - **Fraction Csp3:** fraction of carbon atoms classified as sp3.
    """)
    st.subheader("Important limitations")
    st.write("A descriptor is not a direct measurement. LogP is not the same as pH-dependent LogD. Ionization, tautomerism, stereochemistry, conformer choice, and experimental conditions can change how a molecule behaves.")

elif page == "🧬 Molecular Visualization":
    st.markdown("""
    <div class="section-header-liquid">
        <span class="icon-glow">🧬</span> Molecular Visualization
    </div>
    """, unsafe_allow_html=True)
    chosen = st.selectbox("Load an example molecule", ["(keep current input)"] + list(EXAMPLES.keys()))
    if chosen != "(keep current input)":
        st.session_state.active_smiles = EXAMPLES[chosen]
    smiles = st.text_input("Enter SMILES notation", key="active_smiles")
    mol = parse_smiles(smiles)
    if mol is None:
        st.error("The SMILES string is empty or could not be parsed. Check the notation and try again.")
    else:
        st.success("Valid SMILES: RDKit parsed the molecular graph.")
        left, right = st.columns(2)
        with left:
            st.subheader("2D structure")
            st.markdown('<div class="mol-microscope-card">', unsafe_allow_html=True)
            st.image(Draw.MolToImage(mol, size=(650, 430)), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with right:
            st.subheader("Molecular identity")
            st.markdown(f'<div class="neon-pill">Formula: {rdMolDescriptors.CalcMolFormula(mol)}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="neon-pill">Heavy Atoms: {mol.GetNumHeavyAtoms()}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="neon-pill">Rings: {Lipinski.RingCount(mol)}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="neon-pill">Rotatable: {Lipinski.NumRotatableBonds(mol)}</div>', unsafe_allow_html=True)
            st.write("**Canonical SMILES:**", f"`{Chem.MolToSmiles(mol)}`")
            st.write("**Formula:**", rdMolDescriptors.CalcMolFormula(mol))
            st.write("**Heavy atoms:**", mol.GetNumHeavyAtoms())
            st.write("**Ring count:**", Lipinski.RingCount(mol))
        
        st.divider()
        st.subheader("Interactive 3D conformer")
        style = st.selectbox("Representation", ["Ball and stick", "Stick", "Space filling", "Wireframe"])
        surface = st.checkbox("Show van der Waals surface")
        if st.button("Generate 3D structure"):
            with st.spinner("Generating and optimizing a conformer..."):
                mol3d, method = make_3d(mol)
            if mol3d is None:
                st.error(method)
            else:
                st.caption(f"3D conformer generated. Optimization method: {method}. This is a computational conformer, not necessarily the experimentally observed geometry.")
                show_3d(mol3d, style, surface)

elif page == "📊 Molecular Properties":
    st.markdown("""
    <div class="section-header-liquid">
        <span class="icon-glow">📊</span> Molecular Properties
    </div>
    """, unsafe_allow_html=True)
    chosen = st.selectbox("Load an example molecule", ["(keep current input)"] + list(EXAMPLES.keys()), key="prop_example")
    if chosen != "(keep current input)":
        st.session_state.active_smiles = EXAMPLES[chosen]
    smiles = st.text_input("SMILES notation", key="active_smiles")
    mol = parse_smiles(smiles)
    if mol is None:
        st.error("Enter a valid SMILES string to calculate properties.")
    else:
        props = calculate_properties(mol)
        st.success(f"Structure validated. Formula: {props['Molecular formula']}")
        cols = st.columns(4)
        cols[0].metric("Molecular weight", f"{props['Molecular weight (g/mol)']:.3f} g/mol")
        cols[1].metric("Crippen LogP", props["Crippen LogP"])
        cols[2].metric("TPSA", f"{props['TPSA (Å²)']:.2f} Å²")
        cols[3].metric("Ring count", props["Ring count (RDKit)"])
        
        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
        df = pd.DataFrame([{"Property": k, "Value": v} for k, v in props.items()])
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button("Download properties as CSV", make_csv_bytes(df),
                           file_name="molecular_properties.csv", mime="text/csv")
        st.info("Bond-count convention: sigma bonds count atom-to-atom connections after adding hydrogens. Pi components are counted from a Kekulé form (double = 1, triple = 2). Treat these as a structural teaching count, not a quantum-chemical bond-order calculation.")

elif page == "📈 Structure–Property Comparison":
    st.markdown("""
    <div class="section-header-liquid">
        <span class="icon-glow">📈</span> Structure–Property Comparison
    </div>
    """, unsafe_allow_html=True)
    st.write("Enter one molecule per line using `Name,SMILES`. Invalid lines are reported and skipped.")
    default = "\n".join(f"{n},{s}" for n, s in list(EXAMPLES.items())[:5])
    text = st.text_area("Molecules", value=default, height=190)
    rows, invalid = [], []
    for line_no, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        if "," not in line:
            invalid.append(f"Line {line_no}: missing comma separator")
            continue
        name, smiles = line.split(",", 1)
        mol = parse_smiles(smiles)
        if mol is None:
            invalid.append(f"Line {line_no}: invalid SMILES for {name.strip()}")
            continue
        p = calculate_properties(mol)
        rows.append({"Molecule": name.strip(), "SMILES": smiles.strip(),
                     "Formula": p["Molecular formula"], "MW (g/mol)": p["Molecular weight (g/mol)"],
                     "LogP": p["Crippen LogP"], "TPSA (Å²)": p["TPSA (Å²)"],
                     "HBD": p["H-bond donors"], "HBA": p["H-bond acceptors"],
                     "Rotatable bonds": p["Rotatable bonds"], "Rings": p["Ring count (RDKit)"]})
    for issue in invalid:
        st.warning(issue)
    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button("Download comparison CSV", make_csv_bytes(df),
                           file_name="structure_property_comparison.csv", mime="text/csv")
        numeric = ["MW (g/mol)", "LogP", "TPSA (Å²)", "HBD", "HBA", "Rotatable bonds", "Rings"]
        metric = st.selectbox("Plot descriptor", numeric)
        chart_df = df.set_index("Molecule")[[metric]]
        st.bar_chart(chart_df)
        st.caption("A difference between descriptors suggests a structural difference worth explaining. It does not, by itself, establish causation or biological activity.")
    else:
        st.info("Enter at least one valid molecule in the required format.")

elif page == "📝 Assessment":
    st.markdown("""
    <div class="section-header-liquid">
        <span class="icon-glow">📝</span> Knowledge Check
    </div>
    """, unsafe_allow_html=True)
    questions = [
        ("What does a SMILES string encode?", ["Molecular connectivity as text", "A measured melting point", "An experimental spectrum"], 0),
        ("In SMILES, parentheses usually indicate:", ["Branches", "Isotopes only", "Aromaticity only"], 0),
        ("Which is the example SMILES for toluene here?", ["Cc1ccccc1", "C", "OCC1OC(O)C(O)C(O)C1O"], 0),
        ("Which descriptor is commonly used as an estimate of lipophilicity?", ["Crippen LogP", "Number of rings only", "Atom count"], 0),
        ("TPSA is an estimate of:", ["Topological polar surface area", "Thermal pressure", "Total pi-bond count"], 0),
        ("A double bond contains how many pi-bond components in the usual introductory model?", ["One", "Two", "Zero"], 0),
        ("A triple bond contains how many pi-bond components in the usual introductory model?", ["Two", "One", "Three"], 0),
        ("What does a 3D conformer represent?", ["A possible spatial arrangement", "A guaranteed experimental geometry", "A measured solubility"], 0),
        ("What is a limitation of computed descriptors?", ["They do not automatically replace experiments", "They never depend on structure", "They always predict biological activity"], 0),
        ("What does HBD stand for?", ["Hydrogen-bond donor", "Heavy bond density", "Hydrocarbon bond distance"], 0),
        ("Which representation emphasizes approximate molecular volume?", ["Space filling", "SMILES text only", "A property table"], 0),
        ("Why compare several molecules?", ["To investigate structure–property patterns", "To prove causation from one value", "To avoid checking structures"], 0),
    ]
    with st.form("assessment_form"):
        answers = []
        for i, (q, opts, correct) in enumerate(questions):
            ans = st.radio(f"{i+1}. {q}", opts, index=None, key=f"quiz_{i}")
            answers.append(ans)
        submitted = st.form_submit_button("Submit assessment")
    if submitted:
        if any(a is None for a in answers):
            st.error("Please answer every question before submitting.")
        else:
            score = sum(1 for a, q in zip(answers, questions) if a == q[1][q[2]])
            pct = 100 * score / len(questions)
            st.session_state.assessment_result = {"score": score, "total": len(questions), "percentage": pct}
            st.metric("Score", f"{score}/{len(questions)}", f"{pct:.1f}%")
            if pct >= 80:
                st.success("Excellent. Review the explanations and continue to the final report.")
            elif pct >= 50:
                st.info("Good start. Revisit the theory and descriptor limitations.")
            else:
                st.warning("Review the Theory page and retry the activity.")
            with st.expander("Review answer key"):
                for i, (q, opts, correct) in enumerate(questions):
                    st.write(f"{i+1}. {q} **Answer:** {opts[correct]}")

elif page == "📄 Final Report":
    st.markdown("""
    <div class="section-header-liquid">
        <span class="icon-glow">📄</span> Final Student Report
    </div>
    """, unsafe_allow_html=True)
    st.write("Check the student information, choose the molecule, add your interpretation, and generate a PDF report.")
    name = st.text_input("Student name", key="report_name", value=st.session_state.get("student_name", ""))
    reg = st.text_input("Registration number", key="report_reg", value=st.session_state.get("registration_number", ""))
    chosen = st.selectbox("Load an example molecule", ["(keep current input)"] + list(EXAMPLES.keys()), key="report_example")
    if chosen != "(keep current input)":
        st.session_state.active_smiles = EXAMPLES[chosen]
    smiles = st.text_input("SMILES notation", key="active_smiles")
    notes = st.text_area("Your scientific interpretation", placeholder="Compare the molecular structure and descriptors. Explain what can and cannot be concluded.", height=140)
    mol = parse_smiles(smiles)
    if mol is None:
        st.error("Enter a valid SMILES string before generating the report.")
    else:
        props = calculate_properties(mol)
        st.write("**Molecular formula:**", props["Molecular formula"])
        st.write("**Canonical SMILES:**", Chem.MolToSmiles(mol))
        if st.button("Generate PDF report"):
            pdf = create_pdf(name, reg, smiles, mol, props, notes)
            st.download_button("Download final report (PDF)", pdf,
                               file_name="computational_chemistry_student_report.pdf",
                               mime="application/pdf")
        st.caption("The assessment score is displayed in the Assessment module. Keep a screenshot or record of the score if your instructor requires it.")
