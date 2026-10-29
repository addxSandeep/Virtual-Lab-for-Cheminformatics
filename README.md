# Virtual Laboratory

An undergraduate teaching application based on the structure of the supplied reference `app.py`, with molecular input, 2D and 3D visualization, molecular descriptors, structure-property comparison, quiz assessment, and PDF/CSV export.

## Contents
- `app.py`: Streamlit application
- `requirements.txt`: Python dependencies
- `README.md`: setup, student activity, validation notes, and deployment guide

## 1. Run locally

Python 3.10 or 3.11 is a practical choice. RDKit installation can vary by operating system. If pip installation of RDKit fails, use a Conda environment or the official RDKit installation guidance.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed in the terminal.

## 2. Deploy to Streamlit Community Cloud

1. Create a GitHub repository for this project.
2. Upload `app.py`, `requirements.txt`, and `README.md` to the repository root.
3. Sign in to Streamlit Community Cloud using GitHub.
4. Choose **Create app**, select the repository, branch, and `app.py`.
5. Deploy and wait for the build to finish.
6. Open the public app URL and test every navigation page on desktop and mobile.
7. Share the URL with students. Do not include private student records in public repository files.

Deployment environments change. If RDKit or py3Dmol dependencies fail to build, use the current package installation guidance and pin compatible versions after testing.

## 3. Suggested student activity

**Title:** Computational Chemistry and Molecular Descriptor Analysis

**Aim:** Represent selected molecules using SMILES, visualize 2D/3D structures, calculate descriptors, and interpret structure-property patterns.

**Procedure**
1. Enter your name and registration number in the sidebar.
2. Review the Theory page.
3. Choose three molecules such as toluene, paracetamol, and ibuprofen.
4. Record SMILES, formula, 2D structure, and 3D model observations.
5. Record molecular weight, Crippen LogP, TPSA, H-bond donors/acceptors, rotatable bonds, and ring count.
6. Compare the molecules in the Structure–Property Comparison page.
7. Explain at least two differences using structural evidence.
8. Complete the knowledge check.
9. Generate and submit your PDF report and comparison CSV, if requested by the instructor.

**Submission:** PDF report, comparison table, 2D/3D figures, 150–250 word conclusion, and quiz score evidence.

## 4. Suggested marking rubric (100 marks)

- Molecular identity and valid input: 10
- 2D/3D visualization and observations: 20
- Descriptor calculations and table: 20
- Scientific comparison and interpretation: 25
- Conclusion and report quality: 15
- Quiz performance: 10

## 5. Scientific interpretation and validation notes

- RDKit calculates descriptors from the molecular graph. Results are not experimental measurements.
- Crippen LogP is a model-based estimate for the represented structure; it is not the same as pH-dependent LogD.
- TPSA and hydrogen-bond donor/acceptor counts depend on the toolkit's implemented definitions.
- Ionization state, tautomerism, stereochemistry, salts, and representation can affect comparisons. Students should record the exact SMILES used.
- 3D coordinates are generated conformers and are not guaranteed to match an experimental structure.
- The sigma/pi component count uses an introductory graph-based convention. Sigma count includes added hydrogen bonds; pi components are counted after Kekulization (double bond = 1, triple bond = 2). This is not a quantum-chemical bond-order calculation. Check aromatic and unusual valence systems carefully.
- The application is designed for teaching, not for clinical, toxicological, or regulatory decisions.

## 6. Instructor notes

- Test the application in the actual deployment environment before assigning it.
- Ask students to justify claims with structure and descriptor evidence rather than assuming a descriptor proves a biological outcome.
- This version does not provide a central instructor dashboard or securely collect class submissions. Students must submit reports through your chosen learning-management system or approved institutional channel.
- Do not ask students to enter sensitive personal information beyond what is needed for coursework.
