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

st.set_page_config(page_title="Computational Chemistry Virtual Lab", page_icon="🧪", layout="wide")

st.markdown("""
<style>
.main-title {text-align:center;font-size:2.25rem;font-weight:800;color:#17365D;}
.sub-title {text-align:center;font-size:1.05rem;color:#315b7d;}
div[data-testid="stMetric"] {background:#f4f7fb;border:1px solid #dce5ef;padding:12px;border-radius:10px;}
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
    """Count sigma bonds including H; count pi components from a Kekule form.

    For ordinary closed-shell organic structures: each atom-to-atom bond has one
    sigma component. Each double bond contributes one pi component; each triple
    bond contributes two. Aromatic structures are Kekulized before counting.
    This is a teaching convention, not a quantum-chemical bond-order analysis.
    """
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
            elif bt == Chem.BondType.QUADRUPLE:
                pi_bonds += 3
    except Exception:
        # Aromatic fallback is an approximate count of 0.5 pi component per
        # aromatic edge. The interface flags this convention for interpretation.
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
        view.setStyle({"stick": {"radius": 0.14}, "sphere": {"scale": 0.28}})
    elif style == "Stick":
        view.setStyle({"stick": {"radius": 0.18}})
    elif style == "Space filling":
        view.setStyle({"sphere": {"scale": 0.9}})
    elif style == "Wireframe":
        view.setStyle({"line": {"linewidth": 2}})
    if show_surface:
        view.addSurface(py3Dmol.VDW, {"opacity": 0.45}, {"color": "white"})
    view.setBackgroundColor("white")
    view.zoomTo()
    components.html(view._make_html(), height=540, scrolling=False)

def make_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8-sig")

def create_pdf(student_name, reg_no, smiles, mol, props, notes=""):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=38, leftMargin=38, topMargin=38, bottomMargin=38)
    styles = getSampleStyleSheet()
    title = ParagraphStyle("LabTitle", parent=styles["Title"], fontSize=19, leading=23,
                           alignment=TA_CENTER, textColor=colors.HexColor("#17365D"))
    heading = ParagraphStyle("LabHeading", parent=styles["Heading2"], fontSize=12,
                             textColor=colors.HexColor("#17365D"), spaceBefore=10, spaceAfter=6)
    body = ParagraphStyle("LabBody", parent=styles["BodyText"], fontSize=9, leading=12)
    story = [Paragraph("COMPUTATIONAL CHEMISTRY VIRTUAL LABORATORY", title),
             Spacer(1, 8), Paragraph("Student Molecular Analysis Report", heading)]
    info = [
        ["Student name", student_name or "Not provided"],
        ["Registration number", reg_no or "Not provided"],
        ["Input SMILES", smiles],
        ["Canonical SMILES", Chem.MolToSmiles(mol)],
        ["Molecular formula", props["Molecular formula"]]
    ]
    t = Table(info, colWidths=[1.8*inch, 4.8*inch])
    t.setStyle(TableStyle([("GRID",(0,0),(-1,-1),0.4,colors.grey),
                           ("BACKGROUND",(0,0),(0,-1),colors.HexColor("#EAF0F8")),
                           ("VALIGN",(0,0),(-1,-1),"TOP"),("PADDING",(0,0),(-1,-1),5)]))
    story += [t, Spacer(1, 10), Paragraph("2D molecular structure", heading)]
    img = Draw.MolToImage(mol, size=(650, 420))
    img_buf = BytesIO()
    img.save(img_buf, format="PNG")
    img_buf.seek(0)
    story += [Image(img_buf, width=4.9*inch, height=3.15*inch), Paragraph("Calculated molecular descriptors", heading)]
    data = [["Property", "Value"]] + [[str(k), str(v)] for k, v in props.items()]
    pt = Table(data, colWidths=[3.1*inch, 3.5*inch], repeatRows=1)
    pt.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#2F5597")),
                            ("TEXTCOLOR",(0,0),(-1,0),colors.white),
                            ("GRID",(0,0),(-1,-1),0.35,colors.grey),
                            ("VALIGN",(0,0),(-1,-1),"TOP"),("PADDING",(0,0),(-1,-1),4)]))
    story += [pt, Spacer(1, 8), Paragraph("Interpretation and limitations", heading),
              Paragraph("Descriptors are computed from the submitted molecular graph using RDKit. They are useful for comparing structures, but do not independently establish measured solubility, permeability, toxicity, or biological activity. Results can depend on protonation, tautomerism, stereochemistry, and the molecular representation.", body)]
    if notes.strip():
        story += [Spacer(1, 6), Paragraph("Student interpretation", heading),
                  Paragraph(escape(notes).replace("\n", "<br/>"), body)]
    doc.build(story)
    return buffer.getvalue()

st.sidebar.title("🧪 Computational Chemistry Lab")
page = st.sidebar.radio("Navigation", [
    "🏠 Home", "📘 Theory", "🧬 Molecular Visualization",
    "📊 Molecular Properties", "📈 Structure–Property Comparison",
    "📝 Assessment", "📄 Final Report"
])
st.sidebar.divider()
st.sidebar.text_input("Student name", key="student_name")
st.sidebar.text_input("Registration number", key="registration_number")
st.sidebar.markdown("### Learning outcomes")
st.sidebar.caption("SMILES • 2D/3D structures • molecular descriptors • structure–property relationships • scientific reporting")

if page == "🏠 Home":
    st.markdown('<div class="main-title">🧪 Computational Chemistry Virtual Laboratory</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Undergraduate practical activity | Chemistry and cheminformatics</div>', unsafe_allow_html=True)
    st.divider()
    st.header("Welcome")
    st.write("Explore how molecular structures can be represented and analyzed using computational chemistry tools.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Learning modules", "7")
    c2.metric("Example molecules", len(EXAMPLES))
    c3.metric("Core approach", "Structure → properties")
    st.subheader("What you will do")
    st.markdown("""
    - Enter or select a SMILES string and validate it.
    - Generate a 2D molecular structure and an interactive 3D conformer.
    - Calculate molecular descriptors using RDKit.
    - Compare multiple molecules and explain differences.
    - Complete a short assessment and download a report.
    """)
    st.subheader("Example molecules")
    st.dataframe(pd.DataFrame([{"Molecule": k, "SMILES": v} for k, v in EXAMPLES.items()]),
                 use_container_width=True, hide_index=True)
    st.info("Begin with Theory, then complete Molecular Visualization, Molecular Properties, Comparison, Assessment, and Final Report.")
    st.warning("Educational use: computed descriptors are predictions or graph-derived values, not substitutes for experimental measurements.")

elif page == "📘 Theory":
    st.title("📘 Theory: Introduction to Cheminformatics")
    st.subheader("What is cheminformatics?")
    st.write("Cheminformatics applies computational and information-science methods to represent, store, analyze, and compare chemical structures and related data.")
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
    st.title("🧬 Molecular Visualization")
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
            st.image(Draw.MolToImage(mol, size=(650, 430)), use_container_width=True)
        with right:
            st.subheader("Molecular identity")
            st.write("**Canonical SMILES:**", Chem.MolToSmiles(mol))
            st.write("**Formula:**", rdMolDescriptors.CalcMolFormula(mol))
            st.write("**Heavy atoms:**", mol.GetNumHeavyAtoms())
            st.write("**Ring count:**", Lipinski.RingCount(mol))
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
    st.title("📊 Molecular Properties")
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
        df = pd.DataFrame([{"Property": k, "Value": v} for k, v in props.items()])
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button("Download properties as CSV", make_csv_bytes(df),
                           file_name="molecular_properties.csv", mime="text/csv")
        st.info("Bond-count convention: sigma bonds count atom-to-atom connections after adding hydrogens. Pi components are counted from a Kekulé form (double = 1, triple = 2). Treat these as a structural teaching count, not a quantum-chemical bond-order calculation.")

elif page == "📈 Structure–Property Comparison":
    st.title("📈 Structure–Property Comparison")
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
    st.title("📝 Knowledge Check")
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
    st.title("📄 Final Student Report")
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
