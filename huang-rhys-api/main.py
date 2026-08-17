import asyncio
import json
import re
import shutil
import tempfile
import time
import uuid
import zipfile

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile
)

from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="Huang-Rhys Factor Analysis API",
    version="1.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://prashant-research.github.io",
        "http://localhost:5500",
        "http://127.0.0.1:5500"
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"]
)


# ============================================================
# SETTINGS
# ============================================================

MAX_COMBINED_UPLOAD_BYTES = (
    200 * 1024 * 1024
)

UPLOAD_CHUNK_SIZE = (
    1024 * 1024
)

RESULT_LIFETIME_SECONDS = (
    60 * 60
)


RESULT_ROOT = (
    Path(tempfile.gettempdir())
    / "huang_rhys_jobs"
)

RESULT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)


# Only one large numerical analysis at once.
# This helps protect a small Render instance from
# several simultaneous memory-heavy calculations.

analysis_semaphore = asyncio.Semaphore(1)


# ============================================================
# CONSTANTS FROM NOTEBOOK
# ============================================================

bohr_to_ang = 0.529177210903

amu_to_me = 1822.888486209

cm_to_hartree = (
    1.0 / 219474.6313705
)

cm_to_eV = (
    1.0 / 8065.54429
)


# ============================================================
# GENERAL HELPERS
# ============================================================

def cleanup_old_results():

    now = time.time()

    for item in RESULT_ROOT.iterdir():

        try:

            age = (
                now -
                item.stat().st_mtime
            )

            if age > RESULT_LIFETIME_SECONDS:

                if item.is_dir():

                    shutil.rmtree(
                        item,
                        ignore_errors=True
                    )

                else:

                    item.unlink(
                        missing_ok=True
                    )

        except Exception:

            pass


def gaussian_float(value):

    return float(
        value
        .replace("D", "E")
        .replace("d", "e")
    )


async def save_upload(
    upload,
    destination,
    bytes_already_saved
):

    written = 0

    with open(
        destination,
        "wb"
    ) as handle:

        while True:

            chunk = await upload.read(
                UPLOAD_CHUNK_SIZE
            )

            if not chunk:

                break

            written += len(chunk)

            if (
                bytes_already_saved +
                written >
                MAX_COMBINED_UPLOAD_BYTES
            ):

                raise HTTPException(
                    status_code=413,
                    detail=(
                        "The combined input files exceed "
                        "the 200 MB limit."
                    )
                )

            handle.write(chunk)

    await upload.close()

    return written


# ============================================================
# GAUSSIAN LOG FREQUENCY PARSER
# ============================================================

def read_gaussian_frequencies(
    logfile
):

    frequencies = []

    pattern = re.compile(
        r"Frequencies\s+--\s+(.*)"
    )

    with open(
        logfile,
        "r",
        errors="ignore"
    ) as handle:

        for line in handle:

            match = pattern.search(
                line
            )

            if match:

                values = (
                    match
                    .group(1)
                    .split()
                )

                frequencies.extend(
                    gaussian_float(x)
                    for x in values
                )

    if len(frequencies) == 0:

        raise ValueError(
            "No Gaussian 'Frequencies --' "
            "lines were found."
        )

    return np.asarray(
        frequencies,
        dtype=np.float64
    )


# ============================================================
# MEMORY-CONSCIOUS FCHK PARSER
# ============================================================

def read_gaussian_fchk(
    filename
):

    wanted = {

        "Current cartesian coordinates":
            "coords",

        "Real atomic weights":
            "masses",

        "Vib-Modes":
            "modes"

    }

    results = {}


    with open(
        filename,
        "r",
        errors="ignore"
    ) as handle:

        while True:

            line = handle.readline()

            if not line:

                break


            found_keyword = None
            result_name = None


            for keyword, name in wanted.items():

                if (
                    keyword in line and
                    name not in results
                ):

                    found_keyword = keyword
                    result_name = name

                    break


            if found_keyword is None:

                continue


            count_match = re.search(
                r"N=\s*(\d+)",
                line
            )


            if count_match:

                number_values = int(
                    count_match.group(1)
                )

            else:

                parts = line.split()

                try:

                    number_values = int(
                        parts[-1]
                    )

                except Exception:

                    raise ValueError(
                        "Could not determine array size "
                        f"for '{found_keyword}'."
                    )


            data = []


            while (
                len(data) <
                number_values
            ):

                data_line = (
                    handle.readline()
                )

                if not data_line:

                    raise ValueError(
                        "Unexpected end of formatted "
                        "checkpoint file while reading "
                        f"'{found_keyword}'."
                    )


                for token in (
                    data_line.split()
                ):

                    data.append(
                        gaussian_float(
                            token
                        )
                    )


            results[result_name] = (
                np.asarray(
                    data[
                        :number_values
                    ],
                    dtype=np.float64
                )
            )


            if (
                len(results) ==
                len(wanted)
            ):

                break


    missing = [

        keyword

        for keyword, name
        in wanted.items()

        if name not in results

    ]


    if missing:

        raise ValueError(

            "Required fchk section(s) not found: "

            + ", ".join(
                missing
            )

        )


    return (
        results["coords"],
        results["masses"],
        results["modes"]
    )


# ============================================================
# LOAD GAUSSIAN STATE
# ============================================================

def load_gaussian_state(
    logfile,
    fchkfile
):

    freqs_cm = (
        read_gaussian_frequencies(
            logfile
        )
    )


    (
        coords,
        masses,
        modes_raw
    ) = read_gaussian_fchk(
        fchkfile
    )


    n_atoms = len(
        masses
    )

    N = (
        3 * n_atoms
    )


    if len(coords) != N:

        raise ValueError(
            "Coordinate array length does not "
            "match 3N."
        )


    if (
        len(modes_raw) %
        N != 0
    ):

        raise ValueError(
            "Vib-Modes array length is not "
            "divisible by 3N."
        )


    n_modes = (
        len(modes_raw) //
        N
    )


    modes = (
        modes_raw
        .reshape(
            (n_modes, N)
        )
        .T
    )


    if (
        len(freqs_cm) !=
        n_modes
    ):

        raise ValueError(

            "Frequency count from Gaussian log "

            f"({len(freqs_cm)}) "

            "does not match Vib-Modes count "

            f"from fchk ({n_modes})."

        )


    return {

        "freqs_cm_all":
            freqs_cm,

        "coords_bohr":
            coords,

        "masses_amu":
            masses,

        "modes_cart":
            modes,

        "n_atoms":
            n_atoms,

        "N":
            N,

        "n_modes":
            n_modes

    }


# ============================================================
# WEIGHTED KABSCH ALIGNMENT
# ============================================================

def kabsch_align(
    P,
    Q,
    weights=None
):

    if weights is None:

        weights = np.ones(
            P.shape[0],
            dtype=float
        )


    w = (
        weights /
        np.sum(weights)
    )


    Pc = (
        P -
        np.sum(
            P * w[:, None],
            axis=0
        )
    )


    Qc = (
        Q -
        np.sum(
            Q * w[:, None],
            axis=0
        )
    )


    C = (
        (Qc * w[:, None]).T
        @ Pc
    )


    V, singular_values, Wt = (
        np.linalg.svd(C)
    )


    d = np.sign(
        np.linalg.det(
            V @ Wt
        )
    )


    D = np.diag(
        [
            1.0,
            1.0,
            d
        ]
    )


    R = (
        V @ D @ Wt
    )


    Q_aligned = (
        Qc @ R
    )


    return (
        Pc.flatten(),
        Q_aligned.flatten(),
        R
    )


# ============================================================
# MODE FILTERING
# ============================================================

def build_mode_mask(
    freqs_cm_all,
    low_freq_cutoff_cm
):

    return (
        freqs_cm_all >
        low_freq_cutoff_cm
    )


# ============================================================
# SYMMETRIC INVERSE SQUARE ROOT
# ============================================================

def symm_inv_sqrt(
    matrix,
    eps=1e-12
):

    eigvals, eigvecs = (
        np.linalg.eigh(
            matrix
        )
    )


    smallest = float(
        np.min(eigvals)
    )


    if smallest < eps:

        raise ValueError(

            "Mode-overlap matrix is not "
            "sufficiently positive definite. "

            "Smallest eigenvalue = "

            f"{smallest:.3e}"

        )


    return (

        eigvecs

        @ np.diag(
            1.0 /
            np.sqrt(eigvals)
        )

        @ eigvecs.T

    )


# ============================================================
# MASS-WEIGHTED BASIS
# ============================================================

def build_mass_weighted_basis(
    modes_cart,
    masses_amu
):

    mass_au = np.repeat(
        masses_amu *
        amu_to_me,
        3
    )


    sqrtM = np.sqrt(
        mass_au
    )


    A = (
        sqrtM[:, None] *
        modes_cart
    )


    overlap = (
        A.T @ A
    )


    inverse_sqrt = (
        symm_inv_sqrt(
            overlap
        )
    )


    U = (
        A @ inverse_sqrt
    )


    return (
        U,
        overlap
    )


# ============================================================
# STRONGEST DUSCHINSKY MIXING
# ============================================================

def strongest_mixing_table(
    J,
    orig_i,
    orig_f,
    top_final_indices,
    top_n=5
):

    rows = []


    for final_idx in (
        top_final_indices
    ):

        weights = np.abs(
            J[final_idx, :]
        )


        indices = (

            np.argsort(weights)
            [::-1][:top_n]

        )


        for initial_idx in indices:

            rows.append({

                "final_mode_filtered_idx":
                    int(final_idx + 1),

                "final_mode_gaussian":
                    int(
                        orig_f[
                            final_idx
                        ]
                    ),

                "initial_mode_filtered_idx":
                    int(
                        initial_idx + 1
                    ),

                "initial_mode_gaussian":
                    int(
                        orig_i[
                            initial_idx
                        ]
                    ),

                "J_element":
                    float(
                        J[
                            final_idx,
                            initial_idx
                        ]
                    ),

                "abs_J_element":
                    float(
                        abs(
                            J[
                                final_idx,
                                initial_idx
                            ]
                        )
                    )

            })


    return pd.DataFrame(
        rows
    )


# ============================================================
# MAIN ANALYSIS
# ============================================================

def perform_analysis(
    initial_log,
    initial_fchk,
    final_log,
    final_fchk,
    low_freq_cutoff_cm,
    top_n_modes
):

    # --------------------------------------------------------
    # LOAD STATES
    # --------------------------------------------------------

    state_i = load_gaussian_state(
        initial_log,
        initial_fchk
    )


    state_f = load_gaussian_state(
        final_log,
        final_fchk
    )


    if (
        state_i["n_atoms"] !=
        state_f["n_atoms"]
    ):

        raise ValueError(

            "Atom count mismatch: "

            f"initial state has "
            f"{state_i['n_atoms']} atoms, "

            f"final state has "
            f"{state_f['n_atoms']} atoms."

        )


    if not np.allclose(
        state_i["masses_amu"],
        state_f["masses_amu"],
        atol=1e-6
    ):

        raise ValueError(

            "Atomic masses differ between "
            "initial and final states. "

            "Check atom ordering."

        )


    n_atoms = (
        state_i["n_atoms"]
    )


    N = (
        state_i["N"]
    )


    masses_amu = (
        state_i["masses_amu"]
    )


    # --------------------------------------------------------
    # FREQUENCY FILTERING
    # --------------------------------------------------------

    freqs_i_all = (
        state_i[
            "freqs_cm_all"
        ]
    )


    freqs_f_all = (
        state_f[
            "freqs_cm_all"
        ]
    )


    mask_i = build_mode_mask(
        freqs_i_all,
        low_freq_cutoff_cm
    )


    mask_f = build_mode_mask(
        freqs_f_all,
        low_freq_cutoff_cm
    )


    n_imag_i = int(
        np.sum(
            freqs_i_all < 0.0
        )
    )


    n_imag_f = int(
        np.sum(
            freqs_f_all < 0.0
        )
    )


    n_near_i = int(
        np.sum(
            (
                freqs_i_all >= 0.0
            )
            &
            (
                freqs_i_all < 20.0
            )
        )
    )


    n_near_f = int(
        np.sum(
            (
                freqs_f_all >= 0.0
            )
            &
            (
                freqs_f_all < 20.0
            )
        )
    )


    freqs_i = (
        freqs_i_all[
            mask_i
        ]
    )


    freqs_f = (
        freqs_f_all[
            mask_f
        ]
    )


    modes_i = (
        state_i[
            "modes_cart"
        ][:, mask_i]
    )


    modes_f = (
        state_f[
            "modes_cart"
        ][:, mask_f]
    )


    orig_i = (
        np.arange(
            1,
            len(freqs_i_all) + 1
        )[mask_i]
    )


    orig_f = (
        np.arange(
            1,
            len(freqs_f_all) + 1
        )[mask_f]
    )


    if (
        modes_i.shape[1] !=
        modes_f.shape[1]
    ):

        raise ValueError(

            "Initial and final states do not "
            "have the same number of retained "
            "vibrational modes after filtering."

        )


    n_vib = int(
        modes_i.shape[1]
    )


    if n_vib == 0:

        raise ValueError(
            "No vibrational modes remain "
            "after applying the frequency cutoff."
        )


    # --------------------------------------------------------
    # GEOMETRY ALIGNMENT
    # --------------------------------------------------------

    coords_i = (
        state_i[
            "coords_bohr"
        ]
        .reshape(-1, 3)
    )


    coords_f = (
        state_f[
            "coords_bohr"
        ]
        .reshape(-1, 3)
    )


    (
        coords_i_aligned,
        coords_f_aligned,
        rotation
    ) = kabsch_align(
        coords_i,
        coords_f,
        weights=masses_amu
    )


    delta_R_bohr = (
        coords_f_aligned -
        coords_i_aligned
    )


    rmsd_ang = float(
        np.sqrt(
            np.mean(
                (
                    delta_R_bohr *
                    bohr_to_ang
                ) ** 2
            )
        )
    )


    mass_au = np.repeat(
        masses_amu *
        amu_to_me,
        3
    )


    sqrtM = np.sqrt(
        mass_au
    )


    delta_mw = (
        sqrtM *
        delta_R_bohr
    )


    delta_mw_norm = float(
        np.linalg.norm(
            delta_mw
        )
    )


    # --------------------------------------------------------
    # MASS-WEIGHTED BASES
    # --------------------------------------------------------

    U_i, S_i_raw = (
        build_mass_weighted_basis(
            modes_i,
            masses_amu
        )
    )


    U_f, S_f_raw = (
        build_mass_weighted_basis(
            modes_f,
            masses_amu
        )
    )


    initial_basis_error = float(

        np.max(
            np.abs(
                U_i.T @ U_i -
                np.eye(n_vib)
            )
        )

    )


    final_basis_error = float(

        np.max(
            np.abs(
                U_f.T @ U_f -
                np.eye(n_vib)
            )
        )

    )


    initial_overlap_min = float(
        np.min(
            np.linalg.eigvalsh(
                S_i_raw
            )
        )
    )


    final_overlap_min = float(
        np.min(
            np.linalg.eigvalsh(
                S_f_raw
            )
        )
    )


    # --------------------------------------------------------
    # DUSCHINSKY MATRIX AND SHIFT
    # --------------------------------------------------------

    J = (
        U_f.T @ U_i
    )


    K = (
        U_f.T @ delta_mw
    )


    orth_err = float(

        np.max(
            np.abs(
                J.T @ J -
                np.eye(
                    J.shape[0]
                )
            )
        )

    )


    detJ = float(
        np.linalg.det(J)
    )


    q_i_proj = (
        U_i.T @ delta_mw
    )


    q_f_proj = (
        U_f.T @ delta_mw
    )


    rec_i = (
        U_i @ q_i_proj
    )


    rec_f = (
        U_f @ q_f_proj
    )


    if delta_mw_norm > 0:

        err_i = float(

            np.linalg.norm(
                delta_mw -
                rec_i
            )
            /
            delta_mw_norm

        )


        err_f = float(

            np.linalg.norm(
                delta_mw -
                rec_f
            )
            /
            delta_mw_norm

        )

    else:

        err_i = 0.0
        err_f = 0.0


    consistency = float(

        np.linalg.norm(
            q_f_proj -
            K
        )

    )


    # --------------------------------------------------------
    # HUANG-RHYS FACTORS
    # --------------------------------------------------------

    omega_i_au = (
        freqs_i *
        cm_to_hartree
    )


    omega_f_au = (
        freqs_f *
        cm_to_hartree
    )


    d_f = (
        np.sqrt(
            omega_f_au
        )
        *
        K
    )


    S_f = (
        0.5 *
        d_f ** 2
    )


    d_i = (
        np.sqrt(
            omega_i_au
        )
        *
        q_i_proj
    )


    S_i = (
        0.5 *
        d_i ** 2
    )


    total_S_f = float(
        np.sum(S_f)
    )


    total_S_i = float(
        np.sum(S_i)
    )


    if total_S_f > 0:

        omega_eff_f_cm = float(

            np.sum(
                S_f *
                freqs_f
            )
            /
            total_S_f

        )


        hbar_omega_f_eV = float(

            omega_eff_f_cm *
            cm_to_eV

        )


        lambda_in_f_eV = float(

            total_S_f *
            hbar_omega_f_eV

        )

    else:

        omega_eff_f_cm = None

        hbar_omega_f_eV = None

        lambda_in_f_eV = None


    if total_S_i > 0:

        omega_eff_i_cm = float(

            np.sum(
                S_i *
                freqs_i
            )
            /
            total_S_i

        )

    else:

        omega_eff_i_cm = None


    # --------------------------------------------------------
    # DATA TABLES
    # --------------------------------------------------------

    df_final = pd.DataFrame({

        "filt_idx_final":
            np.arange(
                1,
                n_vib + 1
            ),

        "gauss_mode_final":
            orig_f,

        "freq_final_cm^-1":
            freqs_f,

        "K_final":
            K,

        "d_final":
            d_f,

        "S_final":
            S_f

    })


    df_final = (

        df_final
        .sort_values(
            "S_final",
            ascending=False
        )
        .reset_index(
            drop=True
        )

    )


    df_final[
        "hbar_omega_eV"
    ] = (

        df_final[
            "freq_final_cm^-1"
        ]
        *
        cm_to_eV

    )


    df_final[
        "lambda_in_k_eV"
    ] = (

        df_final[
            "S_final"
        ]
        *
        df_final[
            "hbar_omega_eV"
        ]

    )


    df_initial = pd.DataFrame({

        "filt_idx_initial":
            np.arange(
                1,
                n_vib + 1
            ),

        "gauss_mode_initial":
            orig_i,

        "freq_initial_cm^-1":
            freqs_i,

        "q_initial_proj":
            q_i_proj,

        "d_initial":
            d_i,

        "S_initial":
            S_i

    })


    df_initial = (

        df_initial
        .sort_values(
            "S_initial",
            ascending=False
        )
        .reset_index(
            drop=True
        )

    )


    actual_top_n = min(
        top_n_modes,
        len(df_final)
    )


    top_modes_df = (
        df_final
        .head(actual_top_n)
        .copy()
    )


    top_final_indices = (

        df_final
        .head(
            min(
                5,
                len(df_final)
            )
        )[
            "filt_idx_final"
        ]
        .astype(int)
        .to_numpy()

        - 1

    )


    mixing_df = (
        strongest_mixing_table(
            J,
            orig_i,
            orig_f,
            top_final_indices,
            top_n=5
        )
    )


    # --------------------------------------------------------
    # CREATE RESULT DIRECTORY
    # --------------------------------------------------------

    job_id = uuid.uuid4().hex


    job_dir = (
        RESULT_ROOT /
        job_id
    )


    job_dir.mkdir(
        parents=True,
        exist_ok=False
    )


    # --------------------------------------------------------
    # EXPORT CSV FILES
    # --------------------------------------------------------

    df_final.to_csv(
        job_dir /
        "duschinsky_hr_final_basis.csv",
        index=False
    )


    df_initial.to_csv(
        job_dir /
        "duschinsky_hr_initial_basis.csv",
        index=False
    )


    np.savetxt(
        job_dir /
        "duschinsky_matrix_J.csv",
        J,
        delimiter=","
    )


    np.savetxt(
        job_dir /
        "duschinsky_shift_K.csv",
        K,
        delimiter=","
    )


    mixing_df.to_csv(
        job_dir /
        "strongest_mode_mixing.csv",
        index=False
    )


    # --------------------------------------------------------
    # DUSCHINSKY MATRIX PLOT
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(7, 6)
    )


    image = ax.imshow(
        J,
        aspect="auto"
    )


    fig.colorbar(
        image,
        ax=ax,
        label="J_kl"
    )


    ax.set_xlabel(
        "Initial-state mode index"
    )


    ax.set_ylabel(
        "Final-state mode index"
    )


    ax.set_title(
        "Duschinsky rotation matrix J"
    )


    fig.tight_layout()


    duschinsky_plot = (
        job_dir /
        "duschinsky_matrix.png"
    )


    fig.savefig(
        duschinsky_plot,
        dpi=180,
        bbox_inches="tight"
    )


    plt.close(fig)


    # --------------------------------------------------------
    # TOP HR MODES PLOT
    # --------------------------------------------------------

    plot_df = (

        top_modes_df
        .iloc[::-1]

    )


    fig, ax = plt.subplots(
        figsize=(8, 6)
    )


    ax.barh(

        plot_df[
            "gauss_mode_final"
        ].astype(str),

        plot_df[
            "S_final"
        ]

    )


    ax.set_xlabel(
        "Huang–Rhys factor S"
    )


    ax.set_ylabel(
        "Final Gaussian mode number"
    )


    ax.set_title(

        f"Top {actual_top_n} "
        "final-state Huang–Rhys factors"

    )


    fig.tight_layout()


    hr_plot = (
        job_dir /
        "top_huang_rhys_modes.png"
    )


    fig.savefig(
        hr_plot,
        dpi=180,
        bbox_inches="tight"
    )


    plt.close(fig)


    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    summary = {

        "atoms":
            int(n_atoms),

        "cartesian_dimensions":
            int(N),

        "initial_modes_from_files":
            int(
                state_i[
                    "n_modes"
                ]
            ),

        "final_modes_from_files":
            int(
                state_f[
                    "n_modes"
                ]
            ),

        "retained_modes":
            int(n_vib),

        "low_frequency_cutoff_cm":
            float(
                low_freq_cutoff_cm
            ),

        "initial_imaginary_modes":
            int(n_imag_i),

        "final_imaginary_modes":
            int(n_imag_f),

        "initial_near_zero_modes":
            int(n_near_i),

        "final_near_zero_modes":
            int(n_near_f),

        "aligned_rmsd_ang":
            rmsd_ang,

        "mass_weighted_displacement_norm":
            delta_mw_norm,

        "initial_basis_orthogonality_error":
            initial_basis_error,

        "final_basis_orthogonality_error":
            final_basis_error,

        "initial_raw_overlap_smallest_eigenvalue":
            initial_overlap_min,

        "final_raw_overlap_smallest_eigenvalue":
            final_overlap_min,

        "duschinsky_orthogonality_error":
            orth_err,

        "duschinsky_determinant":
            detJ,

        "initial_basis_reconstruction_error":
            err_i,

        "final_basis_reconstruction_error":
            err_f,

        "shift_consistency_error":
            consistency,

        "total_hr_final_basis":
            total_S_f,

        "total_hr_initial_basis":
            total_S_i,

        "S_eff":
            total_S_f,

        "omega_eff_cm":
            omega_eff_f_cm,

        "hbar_omega_eV":
            hbar_omega_f_eV,

        "lambda_in_eV":
            lambda_in_f_eV,

        "omega_eff_initial_cm":
            omega_eff_i_cm

    }


    # --------------------------------------------------------
    # SUMMARY TEXT
    # --------------------------------------------------------

    with open(
        job_dir /
        "summary.txt",
        "w"
    ) as handle:

        handle.write(
            "Duschinsky-enabled Huang-Rhys analysis\n"
        )

        handle.write(
            "=====================================\n\n"
        )

        handle.write(
            f"Atoms: {n_atoms}\n"
        )

        handle.write(
            f"Retained vibrational modes: {n_vib}\n"
        )

        handle.write(

            "Low-frequency cutoff [cm^-1]: "

            f"{low_freq_cutoff_cm:.6f}\n"

        )

        handle.write(

            "Aligned RMSD [Angstrom]: "

            f"{rmsd_ang:.12e}\n"

        )

        handle.write(

            "Total HR factor in final-state basis: "

            f"{total_S_f:.12e}\n"

        )

        handle.write(

            "Total HR factor in initial-state basis: "

            f"{total_S_i:.12e}\n"

        )


        if omega_eff_f_cm is not None:

            handle.write(

                "Effective frequency "
                "[cm^-1]: "

                f"{omega_eff_f_cm:.12e}\n"

            )


            handle.write(

                "hbar omega [eV]: "

                f"{hbar_omega_f_eV:.12e}\n"

            )


            handle.write(

                "lambda_in [eV]: "

                f"{lambda_in_f_eV:.12e}\n"

            )


        handle.write(

            "max |J^T J - I|: "

            f"{orth_err:.12e}\n"

        )


        handle.write(

            "det(J): "

            f"{detJ:.12e}\n"

        )


        handle.write(

            "\nInitial imaginary modes: "

            f"{n_imag_i}\n"

        )


        handle.write(

            "Final imaginary modes: "

            f"{n_imag_f}\n"

        )


        handle.write(

            "Initial near-zero modes (<20 cm^-1): "

            f"{n_near_i}\n"

        )


        handle.write(

            "Final near-zero modes (<20 cm^-1): "

            f"{n_near_f}\n"

        )


    # --------------------------------------------------------
    # JSON COPY
    # --------------------------------------------------------

    with open(
        job_dir /
        "results.json",
        "w"
    ) as handle:

        json.dump(
            summary,
            handle,
            indent=2
        )


    # --------------------------------------------------------
    # ZIP EVERYTHING
    # --------------------------------------------------------

    zip_path = (
        job_dir /
        "Huang_Rhys_detailed_results.zip"
    )


    files_to_zip = [

        "summary.txt",

        "results.json",

        "duschinsky_hr_final_basis.csv",

        "duschinsky_hr_initial_basis.csv",

        "duschinsky_matrix_J.csv",

        "duschinsky_shift_K.csv",

        "strongest_mode_mixing.csv",

        "duschinsky_matrix.png",

        "top_huang_rhys_modes.png"

    ]


    with zipfile.ZipFile(
        zip_path,
        "w",
        compression=zipfile.ZIP_DEFLATED
    ) as archive:

        for filename in files_to_zip:

            archive.write(

                job_dir /
                filename,

                arcname=filename

            )


    # --------------------------------------------------------
    # TOP MODES FOR WEBSITE
    # --------------------------------------------------------

    top_modes = []


    for _, row in (
        top_modes_df.iterrows()
    ):

        top_modes.append({

            "gaussian_mode":
                int(
                    row[
                        "gauss_mode_final"
                    ]
                ),

            "frequency_cm":
                float(
                    row[
                        "freq_final_cm^-1"
                    ]
                ),

            "S":
                float(
                    row[
                        "S_final"
                    ]
                ),

            "hbar_omega_eV":
                float(
                    row[
                        "hbar_omega_eV"
                    ]
                ),

            "lambda_in_k_eV":
                float(
                    row[
                        "lambda_in_k_eV"
                    ]
                )

        })


    return {

        "success":
            True,

        "job_id":
            job_id,

        "summary":
            summary,

        "top_modes":
            top_modes,

        "download_path":
            f"/download/{job_id}",

        "duschinsky_plot_path":
            f"/plot/{job_id}/duschinsky",

        "hr_plot_path":
            f"/plot/{job_id}/huang-rhys"

    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {

        "status":
            "ok",

        "service":
            "Huang-Rhys Factor Analysis API",

        "version":
            "1.0"

    }


# ============================================================
# ANALYSIS ENDPOINT
# ============================================================

@app.post("/analyze")
async def analyze(

    initial_log: UploadFile =
        File(...),

    initial_fchk: UploadFile =
        File(...),

    final_log: UploadFile =
        File(...),

    final_fchk: UploadFile =
        File(...),

    low_freq_cutoff_cm: float =
        Form(100.0),

    top_n_modes: int =
        Form(20)

):


    cleanup_old_results()


    if (
        low_freq_cutoff_cm < 0
    ):

        raise HTTPException(

            status_code=400,

            detail=(
                "Low-frequency cutoff cannot "
                "be negative."
            )

        )


    if (
        top_n_modes < 1 or
        top_n_modes > 100
    ):

        raise HTTPException(

            status_code=400,

            detail=(
                "Top modes must be between "
                "1 and 100."
            )

        )


    with tempfile.TemporaryDirectory() as temp_name:

        temp_dir = Path(
            temp_name
        )


        files = [

            (
                initial_log,
                temp_dir /
                "initial.log"
            ),

            (
                initial_fchk,
                temp_dir /
                "initial.fchk"
            ),

            (
                final_log,
                temp_dir /
                "final.log"
            ),

            (
                final_fchk,
                temp_dir /
                "final.fchk"
            )

        ]


        total_bytes = 0


        for upload, path in files:

            total_bytes += (
                await save_upload(

                    upload,

                    path,

                    total_bytes

                )
            )


        try:

            async with analysis_semaphore:

                result = await run_in_threadpool(

                    perform_analysis,

                    temp_dir /
                    "initial.log",

                    temp_dir /
                    "initial.fchk",

                    temp_dir /
                    "final.log",

                    temp_dir /
                    "final.fchk",

                    float(
                        low_freq_cutoff_cm
                    ),

                    int(
                        top_n_modes
                    )

                )


        except ValueError as error:

            raise HTTPException(

                status_code=422,

                detail=str(error)

            )


        except np.linalg.LinAlgError as error:

            raise HTTPException(

                status_code=422,

                detail=(
                    "Numerical linear-algebra "
                    "failure: "
                    + str(error)
                )

            )


        except Exception as error:

            raise HTTPException(

                status_code=500,

                detail=(
                    "Unexpected analysis failure: "
                    + str(error)
                )

            )


        result[
            "uploaded_size_mb"
        ] = round(

            total_bytes /
            (1024 * 1024),

            2

        )


        return result


# ============================================================
# ZIP DOWNLOAD
# ============================================================

@app.get(
    "/download/{job_id}"
)
def download_results(
    job_id: str
):


    cleanup_old_results()


    if not re.fullmatch(
        r"[a-f0-9]{32}",
        job_id
    ):

        raise HTTPException(
            status_code=404,
            detail="Result not found."
        )


    zip_path = (

        RESULT_ROOT /

        job_id /

        "Huang_Rhys_detailed_results.zip"

    )


    if not zip_path.exists():

        raise HTTPException(

            status_code=404,

            detail=(
                "Result archive has expired "
                "or does not exist."
            )

        )


    return FileResponse(

        path=zip_path,

        media_type="application/zip",

        filename=(
            "Huang_Rhys_detailed_results.zip"
        )

    )


# ============================================================
# PLOT DOWNLOAD / DISPLAY
# ============================================================

@app.get(
    "/plot/{job_id}/{plot_name}"
)
def result_plot(
    job_id: str,
    plot_name: str
):


    cleanup_old_results()


    if not re.fullmatch(
        r"[a-f0-9]{32}",
        job_id
    ):

        raise HTTPException(
            status_code=404,
            detail="Plot not found."
        )


    filenames = {

        "duschinsky":
            "duschinsky_matrix.png",

        "huang-rhys":
            "top_huang_rhys_modes.png"

    }


    if plot_name not in filenames:

        raise HTTPException(
            status_code=404,
            detail="Plot not found."
        )


    plot_path = (

        RESULT_ROOT /

        job_id /

        filenames[
            plot_name
        ]

    )


    if not plot_path.exists():

        raise HTTPException(
            status_code=404,
            detail="Plot has expired."
        )


    return FileResponse(
        plot_path,
        media_type="image/png"
    )
