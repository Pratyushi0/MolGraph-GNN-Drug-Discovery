# ============================================================
# api/app.py
# FastAPI REST API for MolGraph
#
# Endpoints:
#   GET  /               → Health check
#   POST /predict        → Single molecule prediction
#   POST /predict/batch  → Batch prediction
#   GET  /molecule/{smiles}/info → Molecular info
#   GET  /docs           → Auto-generated API docs
# ============================================================

import os
import sys
import io
import base64
import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional
import uvicorn

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.molecular_features import smiles_to_graph, get_molecular_descriptors
from src.model import build_model

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

# ─────────────────────────────────────────
# APP INITIALIZATION
# ─────────────────────────────────────────

app = FastAPI(
    title       = "🧬 MolGraph API",
    description = """
## MolGraph: GNN-Powered Drug Activity Predictor

This API uses a **Graph Attention Network (GATv2)** trained on molecular graphs
to predict drug activity (Blood-Brain Barrier Penetration).

### How it works:
1. You provide a **SMILES** string (molecular representation)
2. We convert it to an **atomic graph** (atoms=nodes, bonds=edges)
3. Our **GNN** processes the graph through attention layers
4. Returns: **activity probability** + **molecular properties**

### Example SMILES:
- Aspirin:    `CC(=O)Oc1ccccc1C(=O)O`
- Caffeine:   `CN1C=NC2=C1C(=O)N(C(=O)N2C)C`
- Ibuprofen:  `CC(C)Cc1ccc(cc1)C(C)C(=O)O`
    """,
    version     = "1.0.0",
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

# CORS middleware (allow all origins for demo)
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ─────────────────────────────────────────
# MODEL LOADING (on startup)
# ─────────────────────────────────────────

MODEL       = None
DEVICE      = torch.device("cpu")
MODEL_READY = False

@app.on_event("startup")
async def load_model():
    """Load model on API startup."""
    global MODEL, DEVICE, MODEL_READY

    checkpoint_path = os.environ.get(
        "MODEL_CHECKPOINT",
        "checkpoints/best_model.pt"
    )

    print("🚀 Starting MolGraph API...")
    print(f"   Device: {DEVICE}")

    # Build model architecture
    MODEL = build_model(
        model_type      = "GAT",
        in_channels     = 75,
        hidden_channels = 256,
        num_layers      = 4,
        num_heads       = 8,
        edge_dim        = 10,
        num_classes     = 1,
        dropout         = 0.0,  # No dropout at inference
        task_type       = "classification",
    ).to(DEVICE)

    # Load trained weights if available
    if os.path.exists(checkpoint_path):
        MODEL.load_state_dict(
            torch.load(checkpoint_path, map_location=DEVICE)
        )
        MODEL.eval()
        MODEL_READY = True
        print(f"✅ Model loaded from: {checkpoint_path}")
    else:
        print(f"⚠️  No checkpoint found at {checkpoint_path}")
        print(f"   Running with random weights (train first!)")
        MODEL.eval()
        MODEL_READY = False  # Will still work but with random predictions

    print("✅ API ready!")


# ─────────────────────────────────────────
# REQUEST / RESPONSE SCHEMAS
# ─────────────────────────────────────────

class PredictRequest(BaseModel):
    smiles: str = Field(
        ...,
        example      = "CC(=O)Oc1ccccc1C(=O)O",
        description  = "SMILES string of the molecule",
        min_length   = 2,
        max_length   = 1000,
    )
    include_descriptors: bool = Field(
        default     = True,
        description = "Include RDKit molecular descriptors in response",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "smiles":               "CC(=O)Oc1ccccc1C(=O)O",
                "include_descriptors":  True,
            }
        }


class BatchPredictRequest(BaseModel):
    smiles_list: List[str] = Field(
        ...,
        example     = ["CC(=O)Oc1ccccc1C(=O)O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"],
        description = "List of SMILES strings",
        max_items   = 100,
    )


class PredictionResult(BaseModel):
    smiles:              str
    is_valid:            bool
    activity_score:      Optional[float]    # Raw logit
    activity_prob:       Optional[float]    # Sigmoid probability
    predicted_class:     Optional[str]      # "Active" / "Inactive"
    confidence:          Optional[float]    # How confident is the model
    num_atoms:           Optional[int]
    num_bonds:           Optional[int]
    descriptors:         Optional[dict]
    interpretation:      Optional[str]
    model_ready:         bool


# ─────────────────────────────────────────
# PREDICTION HELPER
# ─────────────────────────────────────────

def predict_molecule(smiles: str, include_descriptors: bool = True):
    """Run GNN prediction on a single SMILES."""
    from torch_geometric.data import Data, Batch

    # Validate SMILES
    if RDKIT_AVAILABLE:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {
                'smiles':          smiles,
                'is_valid':        False,
                'activity_score':  None,
                'activity_prob':   None,
                'predicted_class': None,
                'confidence':      None,
                'num_atoms':       None,
                'num_bonds':       None,
                'descriptors':     None,
                'interpretation':  "Invalid SMILES string",
                'model_ready':     MODEL_READY,
            }
        num_atoms = mol.GetNumAtoms()
        num_bonds = mol.GetNumBonds()
    else:
        num_atoms = None
        num_bonds = None

    # Convert to graph
    graph_dict = smiles_to_graph(smiles)
    if graph_dict is None:
        raise HTTPException(
            status_code = 422,
            detail      = f"Failed to convert SMILES to graph: {smiles}"
        )

    # Create PyG Data object
    data = Data(
        x          = graph_dict['x'],
        edge_index = graph_dict['edge_index'],
        edge_attr  = graph_dict['edge_attr'],
    )
    batch = Batch.from_data_list([data]).to(DEVICE)

    # GNN prediction
    with torch.no_grad():
        logit = MODEL(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        prob  = torch.sigmoid(logit).item()
        score = logit.item()

    predicted_class = "Active" if prob >= 0.5 else "Inactive"
    confidence      = prob if prob >= 0.5 else (1 - prob)

    # Interpretation
    if prob >= 0.8:
        interpretation = "🟢 High probability of being active/permeable"
    elif prob >= 0.5:
        interpretation = "🟡 Moderate probability of being active/permeable"
    elif prob >= 0.2:
        interpretation = "🟠 Low probability of being active/permeable"
    else:
        interpretation = "🔴 Very unlikely to be active/permeable"

    # Molecular descriptors
    descriptors = None
    if include_descriptors and RDKIT_AVAILABLE:
        descriptors = get_molecular_descriptors(smiles)
        if descriptors:
            # Round values for clean output
            descriptors = {k: round(v, 3) for k, v in descriptors.items()}

            # Lipinski Rule of 5 check
            lipinski = {
                'MW_ok':       descriptors.get('MolWt', 0) <= 500,
                'LogP_ok':     descriptors.get('LogP', 0) <= 5,
                'HBD_ok':      descriptors.get('NumHDonors', 0) <= 5,
                'HBA_ok':      descriptors.get('NumHAcceptors', 0) <= 10,
                'Lipinski_ok': (
                    descriptors.get('MolWt', 0) <= 500 and
                    descriptors.get('LogP', 0) <= 5 and
                    descriptors.get('NumHDonors', 0) <= 5 and
                    descriptors.get('NumHAcceptors', 0) <= 10
                )
            }
            descriptors['Lipinski'] = lipinski

    return {
        'smiles':          smiles,
        'is_valid':        True,
        'activity_score':  round(score, 4),
        'activity_prob':   round(prob, 4),
        'predicted_class': predicted_class,
        'confidence':      round(confidence, 4),
        'num_atoms':       num_atoms,
        'num_bonds':       num_bonds,
        'descriptors':     descriptors,
        'interpretation':  interpretation,
        'model_ready':     MODEL_READY,
    }


# ─────────────────────────────────────────
# API ENDPOINTS
# ─────────────────────────────────────────

@app.get("/", tags=["Health"])
async def health_check():
    """API health check endpoint."""
    return {
        "status":      "🟢 Online",
        "model":       "MolGraph GATv2",
        "model_ready": MODEL_READY,
        "version":     "1.0.0",
        "description": "GNN-Powered Drug Activity Predictor",
    }


@app.post("/predict", response_model=PredictionResult, tags=["Prediction"])
async def predict_single(request: PredictRequest):
    """
    Predict drug activity for a single molecule.
    
    **Input:** SMILES string
    **Output:** Activity probability, class prediction, molecular properties
    
    **Example SMILES:**
    - Aspirin:   `CC(=O)Oc1ccccc1C(=O)O`
    - Caffeine:  `CN1C=NC2=C1C(=O)N(C(=O)N2C)C`
    - Ibuprofen: `CC(C)Cc1ccc(cc1)C(C)C(=O)O`
    """
    try:
        result = predict_molecule(
            smiles              = request.smiles.strip(),
            include_descriptors = request.include_descriptors,
        )
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", tags=["Prediction"])
async def predict_batch(request: BatchPredictRequest):
    """
    Predict drug activity for multiple molecules at once.
    Maximum 100 molecules per request.
    """
    results = []
    for smiles in request.smiles_list:
        try:
            result = predict_molecule(smiles.strip(), include_descriptors=True)
            results.append(result)
        except Exception as e:
            results.append({
                'smiles': smiles,
                'is_valid': False,
                'error': str(e),
            })

    return {
        "total":    len(results),
        "valid":    sum(1 for r in results if r.get('is_valid', False)),
        "invalid":  sum(1 for r in results if not r.get('is_valid', False)),
        "results":  results,
    }


@app.get("/molecule/{smiles}/info", tags=["Molecule Info"])
async def molecule_info(smiles: str):
    """
    Get detailed molecular information from a SMILES string.
    Returns: atom count, bond count, Lipinski properties, molecular weight, etc.
    """
    import urllib.parse
    smiles = urllib.parse.unquote(smiles)

    if not RDKIT_AVAILABLE:
        raise HTTPException(
            status_code = 503,
            detail      = "RDKit not available"
        )

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise HTTPException(
            status_code = 422,
            detail      = f"Invalid SMILES: {smiles}"
        )

    descriptors = get_molecular_descriptors(smiles)

    # Graph structure info
    graph = smiles_to_graph(smiles)

    return {
        "smiles":            smiles,
        "num_atoms":         mol.GetNumAtoms(),
        "num_bonds":         mol.GetNumBonds(),
        "num_node_features": graph['x'].shape[1] if graph else 0,
        "num_edge_features": graph['edge_attr'].shape[1] if graph else 0,
        "molecular_formula": Chem.rdMolDescriptors.CalcMolFormula(mol),
        "descriptors":       {k: round(v, 4) for k, v in descriptors.items()},
    }


@app.get("/examples", tags=["Examples"])
async def get_examples():
    """Get a list of example SMILES strings for testing."""
    return {
        "examples": [
            {"name": "Aspirin",       "smiles": "CC(=O)Oc1ccccc1C(=O)O",
             "class": "Pain reliever"},
            {"name": "Caffeine",      "smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
             "class": "Stimulant"},
            {"name": "Ibuprofen",     "smiles": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
             "class": "Anti-inflammatory"},
            {"name": "Paracetamol",   "smiles": "CC(=O)Nc1ccc(O)cc1",
             "class": "Analgesic"},
            {"name": "Penicillin G",  "smiles": "CC1(C)SC2C(NC(=O)Cc3ccccc3)C(=O)N2C1C(=O)O",
             "class": "Antibiotic"},
            {"name": "Morphine",      "smiles": "CN1CC[C@]23c4c5ccc(O)c4O[C@H]2[C@@H](O)C=C[C@@H]3[C@@H]1C5",
             "class": "Opioid analgesic"},
            {"name": "Glucose",       "smiles": "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O",
             "class": "Sugar (negative control)"},
        ]
    }


# ─────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("  🧬 MolGraph API Server")
    print("=" * 50)
    print("  URL:  http://localhost:8000")
    print("  Docs: http://localhost:8000/docs")
    print("=" * 50)

    uvicorn.run(
        "app:app",
        host    = "0.0.0.0",
        port    = 8000,
        reload  = False,
        workers = 1,
    )
