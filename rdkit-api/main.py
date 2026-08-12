from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rdkit import Chem
from rdkit.Chem import AllChem


app = FastAPI(
    title="RasaYantra 3D Geometry API"
)


# Allow requests from your GitHub Pages website
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://prashant-research.github.io"
    ],
    allow_credentials=False,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


class MolRequest(BaseModel):
    molfile: str


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "RasaYantra 3D Geometry API"
    }


@app.post("/generate-3d")
def generate_3d(request: MolRequest):

    molblock = request.molfile

    if not molblock.strip():
        raise HTTPException(
            status_code=400,
            detail="No molecular structure was supplied."
        )

    # Prevent excessively large requests
    if len(molblock) > 200000:
        raise HTTPException(
            status_code=400,
            detail="Molecular structure is too large."
        )

    try:

        mol = Chem.MolFromMolBlock(
            molblock,
            sanitize=True,
            removeHs=False
        )

    except Exception as exc:

        raise HTTPException(
            status_code=400,
            detail=f"Could not read molecular structure: {exc}"
        )


    if mol is None:

        raise HTTPException(
            status_code=400,
            detail="RDKit could not interpret the drawn molecule."
        )


    if mol.GetNumAtoms() > 300:

        raise HTTPException(
            status_code=400,
            detail="Version 1 currently limits structures to 300 atoms."
        )


    # Add explicit hydrogen atoms
    mol = Chem.AddHs(
        mol,
        addCoords=True
    )


    # Generate initial 3D structure
    params = AllChem.ETKDGv3()

    params.randomSeed = 2026

    params.clearConfs = True

    result = AllChem.EmbedMolecule(
        mol,
        params
    )


    # Second attempt using random coordinates
    if result == -1:

        params.useRandomCoords = True

        result = AllChem.EmbedMolecule(
            mol,
            params
        )


    if result == -1:

        raise HTTPException(
            status_code=422,
            detail=(
                "RDKit could not generate a 3D conformer "
                "for this molecular structure."
            )
        )


    optimization = "ETKDG only"
    warning = None


    # Try MMFF first
    if AllChem.MMFFHasAllMoleculeParams(mol):

        status = AllChem.MMFFOptimizeMolecule(
            mol,
            maxIters=500
        )

        optimization = "ETKDGv3 + MMFF94"

        if status != 0:
            warning = (
                "MMFF optimization did not fully converge. "
                "The geometry should be treated only as an "
                "initial structure."
            )


    # Otherwise try UFF
    elif AllChem.UFFHasAllMoleculeParams(mol):

        status = AllChem.UFFOptimizeMolecule(
            mol,
            maxIters=500
        )

        optimization = "ETKDGv3 + UFF"

        if status != 0:
            warning = (
                "UFF optimization did not fully converge. "
                "The geometry should be treated only as an "
                "initial structure."
            )


    else:

        warning = (
            "No complete MMFF or UFF parameter set was "
            "available for this molecule. The returned "
            "coordinates are therefore the ETKDG-generated "
            "geometry without force-field refinement."
        )


    xyz = Chem.MolToXYZBlock(
        mol,
        precision=8
    )


    return {
        "success": True,
        "atoms": mol.GetNumAtoms(),
        "method": optimization,
        "xyz": xyz,
        "warning": warning
    }
