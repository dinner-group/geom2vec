import math
import os
import json
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
from tqdm.auto import tqdm

mass_mapping = {
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "P": 30.974,
    "H": 1.008,
    "S": 32.06,
    "F": 18.998,
    "Cl": 35.453,
}
atomic_mapping = {"H": 1, "C": 6, "N": 7, "O": 8, "P": 15, "S": 16, "F": 9, "Cl": 17}


def _require_torch():
    try:
        import torch  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("PyTorch is required for trajectory inference. Install it via `pip install torch`.") from exc
    return torch


def _require_torch_scatter():
    try:
        from torch_scatter import scatter  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "torch-scatter is required for coarse-graining. Install it via `pip install torch-scatter`."
        ) from exc
    return scatter


def _require_mdtraj():
    try:
        import mdtraj as _md  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("mdtraj is required for mdtraj-based extraction. Install it via `pip install mdtraj`.") from exc
    return _md


def _require_mdanalysis():
    try:
        import MDAnalysis as _mda  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "MDAnalysis is required for MDAnalysis-based extraction. Install it via `pip install MDAnalysis`."
        ) from exc
    return _mda





def _resolve_hidden_channels(model, hidden_channels: Optional[int], _visited: Optional[set] = None) -> int:
    if hidden_channels is not None:
        return hidden_channels

    if _visited is None:
        _visited = set()
    obj_id = id(model)
    if obj_id in _visited:
        raise ValueError(
            "Unable to determine `hidden_channels`. Pass it explicitly or ensure the model exposes "
            "`hidden_channels` or `embedding_dimension`."
        )
    _visited.add(obj_id)

    for attr in ("hidden_channels", "embedding_dimension"):
        value = getattr(model, attr, None)
        if isinstance(value, int):
            return value

    module = getattr(model, "module", None)
    if module is not None and module is not model:
        try:
            return _resolve_hidden_channels(module, None, _visited)
        except ValueError:
            pass

    for child in getattr(model, "children", lambda: [])():
        try:
            return _resolve_hidden_channels(child, None, _visited)
        except ValueError:
            continue

    raise ValueError(
        "Unable to determine `hidden_channels`. Pass it explicitly or ensure the model exposes "
        "`hidden_channels` or `embedding_dimension`."
    )




def _resolve_device(model, device, _visited: Optional[set] = None):
    if device is not None:
        return device

    try:
        import torch
    except ModuleNotFoundError:
        return "cpu"

    if _visited is None:
        _visited = set()
    obj_id = id(model)
    if obj_id in _visited:
        return torch.device("cpu")
    _visited.add(obj_id)

    try:
        first_param = next(model.parameters())
    except (AttributeError, StopIteration, TypeError):
        module = getattr(model, "module", None)
        if module is not None and module is not model:
            return _resolve_device(module, device, _visited)
        for child in getattr(model, "children", lambda: [])():
            resolved = _resolve_device(child, device, _visited)
            if resolved is not None:
                return resolved
        return torch.device("cpu")
    else:
        return first_param.device




def infer_traj(
    model,
    hidden_channels: int,
    data: Sequence[np.ndarray],
    atomic_numbers: np.ndarray,
    device,
    saving_path: str,
    batch_size: int = 32,
    reduction: str = "sum",
    cg_mapping: Optional[np.ndarray] = None,
    file_name_list: Optional[Sequence[str]] = None,
    *,
    show_progress: bool = True,
    log_saves: bool = True,
) -> None:
    """Run batched inference over trajectories and persist representations to disk."""

    torch = _require_torch()
    scatter = _require_torch_scatter()

    if reduction not in {"sum", "mean"}:
        raise ValueError("`reduction` must be either 'sum' or 'mean'.")

    device = torch.device(device)
    model = model.to(device=device).eval()

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
        torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]

    output_dir = Path(saving_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    if file_name_list is not None and len(file_name_list) != len(data):
        raise ValueError("file_name_list must match length of data sequences")

    atomic_numbers = np.asarray(atomic_numbers)
    num_atoms = int(atomic_numbers.shape[0])
    torch_atomic = torch.from_numpy(atomic_numbers).to(device)

    cg_map = None
    if cg_mapping is not None:
        cg_counts = torch.from_numpy(np.asarray(cg_mapping)).to(device)
        if cg_counts.sum().item() != num_atoms:
            raise ValueError("Sum of `cg_mapping` entries must equal number of atoms.")
        cg_map = torch.repeat_interleave(torch.arange(cg_counts.shape[0], device=device), cg_counts, dim=0)

    with torch.no_grad():
        for traj_index, traj in enumerate(data):
            traj_tensor = torch.from_numpy(np.asarray(traj)).float().to(device)

            outputs: List[torch.Tensor] = []
            num_frames = traj_tensor.shape[0]
            num_batches = max(1, math.ceil(num_frames / batch_size))
            batch_iter = range(num_batches)
            if show_progress:
                batch_iter = tqdm(batch_iter, desc=f"Inference {traj_index}", total=num_batches)

            for batch_idx in batch_iter:
                start = batch_idx * batch_size
                end = min(start + batch_size, num_frames)
                pos_batch = traj_tensor[start:end]

                n_samples, n_atoms, _ = pos_batch.shape
                z_batch = torch_atomic.expand(n_samples, -1).reshape(-1)
                batch_batch = torch.arange(n_samples, device=device).unsqueeze(1).expand(-1, n_atoms).reshape(-1)

                x_rep, v_rep, _ = model(
                    z=z_batch,
                    pos=pos_batch.reshape(-1, 3).contiguous(),
                    batch=batch_batch,
                )

                x_rep = x_rep.reshape(-1, num_atoms, 1, hidden_channels)
                v_rep = v_rep.reshape(-1, num_atoms, 3, hidden_channels)
                atom_rep = torch.cat([x_rep, v_rep], dim=-2)

                if cg_map is not None:
                    cg_rep = scatter(atom_rep, cg_map, dim=1, reduce=reduction)
                    outputs.append(cg_rep.detach().cpu())
                    continue

                atom_rep = atom_rep.detach().cpu()
                if reduction == "mean":
                    outputs.append(atom_rep.mean(dim=1))
                else:
                    outputs.append(atom_rep.sum(dim=1))

            traj_rep = torch.cat(outputs, dim=0)
            file_name = file_name_list[traj_index] if file_name_list is not None else f"traj_{traj_index}"
            save_path = output_dir / f"{file_name}.pt"
            torch.save(traj_rep, save_path)
            if log_saves:
                print(f"Trajectory {traj_index} saved to {save_path}.")


def count_segments(numbers: Sequence[int]) -> np.ndarray:
    """Compress consecutive integers into counts per contiguous segment."""

    numbers = list(numbers)
    if not numbers:
        return np.asarray([], dtype=int)

    segments = []
    current_segment = [numbers[0]]

    for i in range(1, len(numbers)):
        if numbers[i] == numbers[i - 1]:
            current_segment.append(numbers[i])
        else:
            segments.append(current_segment)
            current_segment = [numbers[i]]
    segments.append(current_segment)

    return np.asarray([len(segment) for segment in segments], dtype=int)


def extract_mda_info(
    protein,
    stride: int = 1,
    selection: Optional[str] = None,
    return_ca: bool = False,
    env_config=None,
):
    """Extract atomic positions, atomic numbers, and segment counts from an MDAnalysis Universe.

    When ``return_ca`` is True, this function also returns centered Cα
    coordinates of shape ``(frames, residues, 3)`` sampled with the same
    stride, using the selection ``\"protein and name CA\"`` (with a fallback to
    ``\"name CA\"`` for topologies without protein annotations).
    """

    _require_mdanalysis()
    env_enabled = bool(getattr(env_config, "enabled", False))
    if env_enabled and not return_ca:
        raise ValueError("env_config requires return_ca=True (environment descriptors are per-token).")

    protein_residues = protein.select_atoms("prop mass > 1.5 ")
    if selection is not None:
        protein_residues = protein.select_atoms(selection)
    atomic_masses = protein_residues.masses
    atomic_masses = np.round(atomic_masses, 3)

    atomic_types = [
        list(mass_mapping.keys())[list(mass_mapping.values()).index(mass)]
        for mass in atomic_masses
    ]
    atomic_numbers = [atomic_mapping[atom] for atom in atomic_types]

    if not return_ca:
        positions = [protein_residues.positions.copy() for _ in protein.trajectory]
        positions = np.asarray(positions)[::stride]
        segment_counts = count_segments(protein_residues.resids)
        return positions, np.array(atomic_numbers), np.array(segment_counts)

    ca = None
    for ca_sel in ("protein and name CA", "name CA"):
        try:
            ca = protein.select_atoms(ca_sel)
        except Exception:
            ca = None
        if ca is not None and getattr(ca, "n_atoms", 0) > 0:
            break

    if env_enabled and (ca is None or getattr(ca, "n_atoms", 0) <= 0):
        raise ValueError("Environment features require Cα atoms, but no Cα atoms were found in this topology.")

    water = None
    lipid_head = None
    lipid_tail = None
    env_scalar_frames = []
    env_vector_frames = []
    ema_alpha = getattr(env_config, "ema_alpha", None) if env_enabled else None
    if ema_alpha is not None:
        ema_alpha = float(ema_alpha)
        if not (0.0 <= ema_alpha < 1.0):
            raise ValueError("EnvironmentConfig.ema_alpha must be in the range [0, 1).")
    ema_scalar_prev = None
    ema_vector_prev = None
    want_water = env_enabled and bool(
        getattr(env_config, "compute_water", True) or getattr(env_config, "compute_water_vectors", False)
    )
    want_water_vectors = env_enabled and bool(getattr(env_config, "compute_water_vectors", False))

    want_lipid_head = env_enabled and bool(
        getattr(env_config, "compute_lipids", False)
        or getattr(env_config, "compute_lipid_vectors", False)
        or getattr(env_config, "compute_membrane_frame", False)
    )
    want_lipid_tail = env_enabled and bool(
        getattr(env_config, "compute_lipid_tail", False) or getattr(env_config, "compute_lipid_tail_vectors", False)
    )

    if env_enabled and not (want_water or want_lipid_head or want_lipid_tail):
        raise ValueError(
            "env_config.enabled=True but no environment features were requested. "
            "Enable at least one of: compute_water, compute_water_vectors, compute_lipids, "
            "compute_lipid_vectors, compute_lipid_tail, compute_membrane_frame."
        )

    if want_water:
        try:
            water = protein.select_atoms(getattr(env_config, "water_selection_mda", "resname SOL and name O"))
        except Exception as exc:
            raise ValueError(
                "Failed to build MDAnalysis selection for water atoms. "
                "Pass a valid `EnvironmentConfig.water_selection_mda`."
            ) from exc
    if want_lipid_head:
        lipid_head_sel = getattr(env_config, "lipid_head_selection_mda", None)
        if lipid_head_sel is None:
            raise ValueError("lipid_head_selection_mda must be set when computing lipid or membrane-frame features.")
        try:
            lipid_head = protein.select_atoms(lipid_head_sel)
        except Exception as exc:
            raise ValueError(
                "Failed to build MDAnalysis selection for lipid headgroup atoms. "
                "Pass a valid `EnvironmentConfig.lipid_head_selection_mda`."
            ) from exc
    if want_lipid_tail:
        lipid_tail_sel = getattr(env_config, "lipid_tail_selection_mda", None)
        if lipid_tail_sel is None:
            raise ValueError("lipid_tail_selection_mda must be set when compute_lipid_tail is enabled.")
        try:
            lipid_tail = protein.select_atoms(lipid_tail_sel)
        except Exception as exc:
            raise ValueError(
                "Failed to build MDAnalysis selection for lipid tail atoms. "
                "Pass a valid `EnvironmentConfig.lipid_tail_selection_mda`."
            ) from exc

    if env_enabled and (want_water or want_lipid_head or want_lipid_tail):
        try:
            from geom2vec.data.environment import compute_membrane_depth, compute_membrane_frame, compute_rbf_features
        except Exception as exc:
            raise ImportError("Failed to import mean-field environment feature helpers.") from exc

    positions_frames = []
    ca_frames = []
    segment_counts = count_segments(protein_residues.resids)
    num_tokens = int(np.asarray(segment_counts).shape[0])

    for _ts in protein.trajectory[::stride]:
        positions_frames.append(protein_residues.positions.copy())
        if ca is not None and getattr(ca, "n_atoms", 0) > 0:
            ca_pos = ca.positions.copy()
            if env_enabled and ca_pos.shape[0] != num_tokens:
                raise ValueError(
                    "Environment features require residue-level tokens aligned to Cα atoms, but "
                    f"token count ({num_tokens}) != number of selected Cα atoms ({ca_pos.shape[0]}). "
                    "Ensure `selection` selects protein atoms only (e.g., 'protein and prop mass > 1.5')."
                )
            ca_frames.append(ca_pos)
            if env_enabled:
                box_A = getattr(_ts, "dimensions", None)
                scalar_parts = []
                vector_parts = []

                if want_water:
                    water_pos = np.zeros((0, 3), dtype=np.float32)
                    if water is not None and getattr(water, "n_atoms", 0) > 0:
                        water_pos = water.positions.copy()
                    rho, moment = compute_rbf_features(ca_pos, water_pos, config=env_config, box_A=box_A)
                    if getattr(env_config, "compute_water", True):
                        scalar_parts.append(rho)
                    if want_water_vectors:
                        vector_parts.append(moment)

                if want_lipid_head:
                    head_pos = np.zeros((0, 3), dtype=np.float32)
                    if lipid_head is not None and getattr(lipid_head, "n_atoms", 0) > 0:
                        head_pos = lipid_head.positions.copy()
                    rho_h, moment_h = compute_rbf_features(ca_pos, head_pos, config=env_config, box_A=box_A)
                    if getattr(env_config, "compute_lipids", False):
                        scalar_parts.append(rho_h)
                    if getattr(env_config, "compute_lipid_vectors", False):
                        vector_parts.append(moment_h)

                if want_lipid_tail:
                    tail_pos = np.zeros((0, 3), dtype=np.float32)
                    if lipid_tail is not None and getattr(lipid_tail, "n_atoms", 0) > 0:
                        tail_pos = lipid_tail.positions.copy()
                    rho_t, moment_t = compute_rbf_features(ca_pos, tail_pos, config=env_config, box_A=box_A)
                    if getattr(env_config, "compute_lipid_tail", False):
                        scalar_parts.append(rho_t)
                    if getattr(env_config, "compute_lipid_tail_vectors", False):
                        vector_parts.append(moment_t)

                if getattr(env_config, "compute_membrane_frame", False):
                    head_pos = np.zeros((0, 3), dtype=np.float32)
                    if lipid_head is not None and getattr(lipid_head, "n_atoms", 0) > 0:
                        head_pos = lipid_head.positions.copy()
                    if head_pos.size == 0:
                        raise ValueError("compute_membrane_frame=True requires lipid headgroup atoms, but none were selected.")
                    normal, center, thickness = compute_membrane_frame(head_pos, config=env_config, box_A=box_A)
                    depth = compute_membrane_depth(
                        ca_pos,
                        membrane_center_A=center,
                        membrane_normal=normal,
                        membrane_thickness_A=thickness,
                        normalize=bool(getattr(env_config, "normalize_membrane_depth", True)),
                    )
                    scalar_parts.append(depth)
                    normal_field = np.broadcast_to(normal.reshape(1, 3, 1), (num_tokens, 3, 1)).astype(np.float32)
                    vector_parts.append(normal_field)

                frame_scalar = None
                if scalar_parts:
                    frame_scalar = np.concatenate(scalar_parts, axis=-1).astype(np.float32, copy=False)
                frame_vector = None
                if vector_parts:
                    frame_vector = np.concatenate(vector_parts, axis=-1).astype(np.float32, copy=False)

                if ema_alpha is not None:
                    if frame_scalar is not None:
                        if ema_scalar_prev is None:
                            ema_scalar_prev = frame_scalar
                        else:
                            ema_scalar_prev = (ema_alpha * ema_scalar_prev) + ((1.0 - ema_alpha) * frame_scalar)
                        frame_scalar = ema_scalar_prev
                    if frame_vector is not None:
                        if ema_vector_prev is None:
                            ema_vector_prev = frame_vector
                        else:
                            ema_vector_prev = (ema_alpha * ema_vector_prev) + ((1.0 - ema_alpha) * frame_vector)
                        frame_vector = ema_vector_prev

                if frame_scalar is not None:
                    env_scalar_frames.append(frame_scalar)
                if frame_vector is not None:
                    env_vector_frames.append(frame_vector)

    positions = np.asarray(positions_frames)

    ca_array = None
    if ca_frames:
        ca_array = np.asarray(ca_frames, dtype=np.float32)
        ca_array = ca_array - ca_array.mean(axis=1, keepdims=True)

    if not env_enabled:
        return positions, np.array(atomic_numbers), np.array(segment_counts), ca_array

    env_scalar = None
    if env_scalar_frames:
        env_scalar = np.asarray(env_scalar_frames, dtype=np.float32)
    env_vector = None
    if env_vector_frames:
        env_vector = np.asarray(env_vector_frames, dtype=np.float32)
    return positions, np.array(atomic_numbers), np.array(segment_counts), ca_array, env_scalar, env_vector


def extract_mda_info_folder(
    folder: str,
    top_file: str,
    stride: int = 1,
    selection: Optional[str] = None,
    file_postfix: str = ".dcd",
    sorting: bool = True,
):
    """Extract MDAnalysis-based metadata for all trajectories within a folder."""

    mda = _require_mdanalysis()

    dcd_files = [f for f in os.listdir(folder) if f.endswith(file_postfix)]
    if sorting:
        dcd_files.sort()

    position_list: List[np.ndarray] = []
    universes: List = []
    file_paths: List[str] = []
    atomic_numbers = None
    segment_counts = None

    for traj in dcd_files:
        path = os.path.join(folder, traj)
        print(f"Processing {traj}")
        universe = mda.Universe(top_file, path)
        positions, atomic_numbers, segment_counts = extract_mda_info(
            universe,
            stride=stride,
            selection=selection,
        )
        position_list.append(positions)
        universes.append(universe)
        file_paths.append(path)

    return position_list, atomic_numbers, segment_counts, file_paths, universes


def extract_mdtraj_info(md_traj_object, exclude_hydrogens: bool = True):
    """Extract positions and metadata from an mdtraj trajectory."""

    md = _require_mdtraj()
    if not isinstance(md_traj_object, md.Trajectory):  # type: ignore[attr-defined]
        raise TypeError("md_traj_object must be an mdtraj.Trajectory")

    atomic_numbers = np.array([atom.element.atomic_number for atom in md_traj_object.top.atoms])
    residue_indices = np.array([atom.residue.index for atom in md_traj_object.top.atoms])
    positions = md_traj_object.xyz * 10.0

    if exclude_hydrogens:
        mask = atomic_numbers != 1
        positions = positions[:, mask]
        atomic_numbers = atomic_numbers[mask]
        residue_indices = residue_indices[mask]

    segment_counts = count_segments(residue_indices)
    return positions, atomic_numbers, segment_counts


def extract_mdtraj_info_folder(
    folder: str,
    top_file: str,
    stride: int = 1,
    selection: str = "protein",
    file_postfix: str = ".dcd",
    num_trajs: Optional[int] = None,
    exclude_hydrogens: bool = True,
):
    """Load mdtraj trajectories from a directory and extract metadata."""

    md = _require_mdtraj()

    dcd_files = [f for f in os.listdir(folder) if f.endswith(file_postfix)]
    dcd_files.sort()

    if num_trajs is not None:
        dcd_files = dcd_files[:num_trajs]

    positions_list: List[np.ndarray] = []
    file_paths: List[str] = []
    trajectories: List = []
    atomic_numbers = None
    segment_counts = None

    for traj_file in dcd_files:
        print(f"Processing {traj_file}")
        path = os.path.join(folder, traj_file)
        try:
            traj = md.load(path, top=top_file, stride=stride)
        except Exception as exc:
            print(f"Error loading file {traj_file}: {exc}")
            continue

        if selection == "protein":
            traj = traj.atom_slice(traj.top.select("protein"))
        elif selection == "backbone":
            traj = traj.atom_slice(traj.top.select("backbone"))
        elif selection == "heavy":
            traj = traj.atom_slice(traj.top.select("not water and not hydrogen"))
        elif selection != "all":
            raise ValueError("Invalid selection type")

        pos, atomic_numbers, segment_counts = extract_mdtraj_info(traj, exclude_hydrogens=exclude_hydrogens)
        positions_list.append(pos)
        file_paths.append(path)
        trajectories.append(traj)

    return positions_list, atomic_numbers, segment_counts, file_paths, trajectories


def infer_mdanalysis_folder(
    model,
    topology_file: str,
    trajectory_folder: str,
    output_dir: str,
    *,
    hidden_channels: Optional[int] = None,
    device=None,
    stride: int = 1,
    selection: Optional[str] = None,
    file_postfix: str = ".dcd",
    sorting: bool = True,
    batch_size: int = 32,
    reduction: str = "sum",
    cg_mapping: Optional[np.ndarray] = None,
    overwrite: bool = False,
    env_config=None,
) -> dict:
    """High-level helper that extracts MDAnalysis trajectories and runs inference.

    Parameters
    ----------
    model :
        Pretrained geom2vec representation model.
    topology_file : str
        Path to the topology file (e.g., `.pdb`).
    trajectory_folder : str
        Directory containing trajectory files (default: `.dcd` files).
    output_dir : str
        Destination directory where `.pt` embeddings will be stored.
    hidden_channels : int, optional
        Representation width. If omitted, attempts to infer from the model.
    device : Optional[torch.device or str], optional
        Target device for inference. Defaults to the model parameter device or CPU.
    stride : int, optional
        Subsampling stride applied when loading coordinates.
    selection : str, optional
        MDAnalysis selection string to filter atoms.
    file_postfix : str, optional
        Extension pattern for trajectory files (default: `.dcd`).
    sorting : bool, optional
        Whether to sort trajectory filenames before processing.
    batch_size : int, optional
        Number of frames processed per inference batch.
    reduction : {"sum", "mean"}, optional
        Aggregation mode when coarse-graining is not requested.
    cg_mapping : np.ndarray, optional
        Custom coarse-grained mapping. If omitted, uses the residue-derived mapping.
    overwrite : bool, optional
        If False (default), skip trajectories whose outputs already exist.

    Returns
    -------
    dict
        A dictionary containing:
        - "computed": A list of dictionaries, each with "source_path" (str),
          "output_path" (Path), and "frames" (int) for processed trajectories.
        - "skipped": A list of `Path` objects for trajectories that were skipped.
    """

    mda = _require_mdanalysis()

    traj_files = []
    for root, _, files in os.walk(trajectory_folder):
        for file in files:
            if file.endswith(file_postfix):
                traj_files.append(os.path.join(root, file))

    if sorting:
        traj_files.sort()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    computed = []
    skipped = []
    resolved_hidden = _resolve_hidden_channels(model, hidden_channels)
    resolved_device = _resolve_device(model, device)

    atomic_numbers: Optional[np.ndarray] = None
    default_cg = np.asarray(cg_mapping) if cg_mapping is not None else None

    progress_iter = tqdm(traj_files, desc="Inference", total=len(traj_files), leave=True)

    for traj_path in progress_iter:
        relative_path = os.path.relpath(traj_path, trajectory_folder)
        name = Path(relative_path).with_suffix("").as_posix().replace("/", "_")
        
        target_dir = output_path / os.path.dirname(relative_path)
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"{Path(traj_path).stem}.pt"

        if target.exists() and not overwrite:
            skipped.append(target)
            continue

        universe = mda.Universe(topology_file, traj_path)
        try:
            env_enabled = bool(getattr(env_config, "enabled", False))
            if env_enabled:
                positions, numbers, segment_counts, ca_array, env_scalar, env_vector = extract_mda_info(
                    universe,
                    stride=stride,
                    selection=selection,
                    return_ca=True,
                    env_config=env_config,
                )
            else:
                positions, numbers, segment_counts, ca_array = extract_mda_info(
                    universe,
                    stride=stride,
                    selection=selection,
                    return_ca=True,
                )
                env_scalar = None
                env_vector = None

            try:
                torch = _require_torch()
            except ImportError:
                ca_target = None
                env_target = None
            else:
                if ca_array is not None:
                    ca_target = target_dir / f"{Path(traj_path).stem}_ca.pt"
                    if overwrite or not ca_target.exists():
                        torch.save(torch.from_numpy(ca_array), ca_target)
                if env_scalar is not None:
                    env_target = target_dir / f"{Path(traj_path).stem}_env_scalar.pt"
                    if overwrite or not env_target.exists():
                        torch.save(torch.from_numpy(env_scalar), env_target)
                if env_vector is not None:
                    env_vec_target = target_dir / f"{Path(traj_path).stem}_env_vector.pt"
                    if overwrite or not env_vec_target.exists():
                        torch.save(torch.from_numpy(env_vector), env_vec_target)
                if env_enabled and (env_scalar is not None or env_vector is not None):
                    meta_target = target_dir / f"{Path(traj_path).stem}_env_meta.json"
                    if overwrite or not meta_target.exists():
                        try:
                            from geom2vec.data.environment import rbf_centers_A
                        except Exception:
                            centers = None
                        else:
                            centers = rbf_centers_A(env_config).tolist()  # type: ignore[arg-type]

                        meta = {
                            "enabled": True,
                            "backend": "mdanalysis",
                            "water_selection_mda": getattr(env_config, "water_selection_mda", None),
                            "water_selection_mdtraj": getattr(env_config, "water_selection_mdtraj", None),
                            "lipid_head_selection_mda": getattr(env_config, "lipid_head_selection_mda", None),
                            "lipid_tail_selection_mda": getattr(env_config, "lipid_tail_selection_mda", None),
                            "lipid_head_selection_mdtraj": getattr(env_config, "lipid_head_selection_mdtraj", None),
                            "lipid_tail_selection_mdtraj": getattr(env_config, "lipid_tail_selection_mdtraj", None),
                            "r_max_A": getattr(env_config, "r_max_A", None),
                            "n_rbf": getattr(env_config, "n_rbf", None),
                            "rbf_width_A": getattr(env_config, "rbf_width_A", None),
                            "rbf_centers_A": centers,
                            "use_pbc": getattr(env_config, "use_pbc", None),
                            "compute_water": getattr(env_config, "compute_water", None),
                            "compute_water_vectors": getattr(env_config, "compute_water_vectors", None),
                            "compute_lipids": getattr(env_config, "compute_lipids", None),
                            "compute_lipid_vectors": getattr(env_config, "compute_lipid_vectors", None),
                            "compute_lipid_tail": getattr(env_config, "compute_lipid_tail", None),
                            "compute_lipid_tail_vectors": getattr(env_config, "compute_lipid_tail_vectors", None),
                            "compute_membrane_frame": getattr(env_config, "compute_membrane_frame", None),
                            "normalize_membrane_depth": getattr(env_config, "normalize_membrane_depth", None),
                            "ema_alpha": getattr(env_config, "ema_alpha", None),
                        }
                        with open(meta_target, "w") as f:
                            json.dump(meta, f, indent=4)
        finally:
            trajectory = getattr(universe, "trajectory", None)
            close = getattr(trajectory, "close", None)
            if callable(close):
                close()

        numbers = np.asarray(numbers)
        if atomic_numbers is None:
            atomic_numbers = numbers
        elif not np.array_equal(atomic_numbers, numbers):
            raise ValueError("Atomic numbers differ across trajectories; cannot batch inference.")

        mapping = default_cg if default_cg is not None else segment_counts

        infer_traj(
            model=model,
            hidden_channels=resolved_hidden,
            data=[positions],
            atomic_numbers=atomic_numbers,
            device=resolved_device,
            saving_path=str(target_dir),
            batch_size=batch_size,
            reduction=reduction,
            cg_mapping=mapping,
            file_name_list=[Path(traj_path).stem],
            show_progress=False,
            log_saves=False,
        )
        computed.append(
            {"source_path": traj_path, "output_path": target, "frames": positions.shape[0]}
        )
        if hasattr(progress_iter, "set_postfix"):
            progress_iter.set_postfix({"file": name})

    # Prepare data for JSON serialization
    summary_data = {
        "computed": [
            {
                "source_path": item["source_path"],
                "output_path": str(item["output_path"]),
                "frames": item["frames"],
            }
            for item in computed
        ],
        "skipped": [str(p) for p in skipped],
    }

    # Save summary to a JSON file
    summary_file = output_path / "inference_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary_data, f, indent=4)

    return {"computed": computed, "skipped": skipped}


def infer_mdtraj_folder(
    model,
    topology_file: str,
    trajectory_folder: str,
    output_dir: str,
    *,
    hidden_channels: Optional[int] = None,
    device=None,
    stride: int = 1,
    selection: str = "protein",
    file_postfix: str = ".dcd",
    sorting: bool = True,
    batch_size: int = 32,
    reduction: str = "sum",
    cg_mapping: Optional[np.ndarray] = None,
    exclude_hydrogens: bool = True,
    overwrite: bool = False,
    env_config=None,
) -> dict:
    """High-level helper that loads MDTraj trajectories and runs inference.

    Mirrors :func:`infer_mdanalysis_folder` but uses MDTraj for I/O.

    Notes
    -----
    - Coordinates are assumed to be in **nm** in MDTraj and are converted to Å.
    - When environment features are enabled, the input selection must correspond to
      residue-level protein tokens aligned to Cα atoms.
    """

    md = _require_mdtraj()
    torch = _require_torch()

    traj_files = []
    for root, _, files in os.walk(trajectory_folder):
        for file in files:
            if file.endswith(file_postfix):
                traj_files.append(os.path.join(root, file))

    if sorting:
        traj_files.sort()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    computed = []
    skipped = []
    resolved_hidden = _resolve_hidden_channels(model, hidden_channels)
    resolved_device = _resolve_device(model, device)

    atomic_numbers: Optional[np.ndarray] = None
    default_cg = np.asarray(cg_mapping) if cg_mapping is not None else None

    env_enabled = bool(getattr(env_config, "enabled", False))

    for traj_path in tqdm(traj_files, desc="Inference", total=len(traj_files), leave=True):
        relative_path = os.path.relpath(traj_path, trajectory_folder)
        target_dir = output_path / os.path.dirname(relative_path)
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"{Path(traj_path).stem}.pt"

        if target.exists() and not overwrite:
            skipped.append(target)
            continue

        traj = md.load(traj_path, top=topology_file, stride=stride)

        if selection == "protein":
            atom_indices = traj.top.select("protein")
        elif selection == "backbone":
            atom_indices = traj.top.select("backbone")
        elif selection == "heavy":
            atom_indices = traj.top.select("not water and not hydrogen")
        elif selection == "all":
            atom_indices = np.arange(traj.n_atoms, dtype=int)
        else:
            # Treat as MDTraj selection query.
            atom_indices = traj.top.select(selection)

        atom_indices = np.asarray(atom_indices, dtype=int)
        if atom_indices.size == 0:
            raise ValueError(f"Selection '{selection}' produced zero atoms for {traj_path}.")

        atom_numbers = np.asarray([traj.top.atom(int(i)).element.atomic_number for i in atom_indices], dtype=int)
        residue_indices = np.asarray([traj.top.atom(int(i)).residue.index for i in atom_indices], dtype=int)

        if exclude_hydrogens:
            heavy_mask = atom_numbers != 1
            atom_indices = atom_indices[heavy_mask]
            atom_numbers = atom_numbers[heavy_mask]
            residue_indices = residue_indices[heavy_mask]

        positions_A = traj.xyz[:, atom_indices, :] * 10.0
        segment_counts = count_segments(residue_indices)

        # Determine residue ordering and map to CA indices for token coordinates.
        ca_array = None
        if segment_counts.size > 0:
            starts = np.concatenate([[0], np.cumsum(segment_counts)[:-1]])
            residue_ids = residue_indices[starts]

            ca_by_res = {}
            for atom in traj.top.atoms:
                if atom.name == "CA":
                    ca_by_res[atom.residue.index] = atom.index

            try:
                ca_indices = np.asarray([ca_by_res[int(rid)] for rid in residue_ids], dtype=int)
            except KeyError as exc:
                if env_enabled:
                    raise ValueError(
                        "Environment features require Cα atoms aligned to residue tokens, but a residue "
                        "in the selected atoms is missing a CA atom."
                    ) from exc
                ca_indices = np.asarray([], dtype=int)
            else:
                ca_array = traj.xyz[:, ca_indices, :] * 10.0
                ca_array = ca_array.astype(np.float32)
                ca_array = ca_array - ca_array.mean(axis=1, keepdims=True)

        if env_enabled:
            if ca_array is None or ca_array.size == 0:
                raise ValueError("env_config.enabled=True requires residue-aligned Cα coordinates, but none were found.")

            from geom2vec.data.environment import compute_membrane_depth, compute_membrane_frame, compute_rbf_features, rbf_centers_A

            # Environment selections (evaluated on the full system topology).
            water_idx = np.asarray(traj.top.select(getattr(env_config, "water_selection_mdtraj", "water and name O")), dtype=int)
            head_sel = getattr(env_config, "lipid_head_selection_mdtraj", None)
            tail_sel = getattr(env_config, "lipid_tail_selection_mdtraj", None)
            head_idx = np.asarray(traj.top.select(head_sel), dtype=int) if head_sel else np.asarray([], dtype=int)
            tail_idx = np.asarray(traj.top.select(tail_sel), dtype=int) if tail_sel else np.asarray([], dtype=int)

            want_water = bool(getattr(env_config, "compute_water", True) or getattr(env_config, "compute_water_vectors", False))
            want_water_vectors = bool(getattr(env_config, "compute_water_vectors", False))
            want_lipid_head = bool(
                getattr(env_config, "compute_lipids", False)
                or getattr(env_config, "compute_lipid_vectors", False)
                or getattr(env_config, "compute_membrane_frame", False)
            )
            want_lipid_tail = bool(getattr(env_config, "compute_lipid_tail", False) or getattr(env_config, "compute_lipid_tail_vectors", False))

            if want_lipid_head and head_idx.size == 0:
                raise ValueError("lipid_head_selection_mdtraj must be set (and match atoms) when lipid/membrane features are enabled.")
            if want_lipid_tail and tail_idx.size == 0:
                raise ValueError("lipid_tail_selection_mdtraj must be set (and match atoms) when tail features are enabled.")

            ema_alpha = getattr(env_config, "ema_alpha", None)
            if ema_alpha is not None:
                ema_alpha = float(ema_alpha)
                if not (0.0 <= ema_alpha < 1.0):
                    raise ValueError("EnvironmentConfig.ema_alpha must be in the range [0, 1).")
            ema_scalar_prev = None
            ema_vector_prev = None

            unitcell = traj.unitcell_lengths
            env_scalar_frames = []
            env_vector_frames = []

            # Use uncentered CA coordinates for env computation.
            ca_uncentered = (traj.xyz[:, ca_indices, :] * 10.0).astype(np.float32)
            num_tokens = int(ca_uncentered.shape[1])

            for frame in range(traj.n_frames):
                box_A = None
                if unitcell is not None and unitcell.shape[0] == traj.n_frames:
                    box_A = (unitcell[frame] * 10.0).astype(np.float32)

                token_xyz = ca_uncentered[frame]
                scalar_parts = []
                vector_parts = []

                if want_water:
                    water_xyz = (traj.xyz[frame, water_idx, :] * 10.0).astype(np.float32) if water_idx.size else np.zeros((0, 3), dtype=np.float32)
                    rho_w, moment_w = compute_rbf_features(token_xyz, water_xyz, config=env_config, box_A=box_A)
                    if getattr(env_config, "compute_water", True):
                        scalar_parts.append(rho_w)
                    if want_water_vectors:
                        vector_parts.append(moment_w)

                if want_lipid_head:
                    head_xyz = (traj.xyz[frame, head_idx, :] * 10.0).astype(np.float32) if head_idx.size else np.zeros((0, 3), dtype=np.float32)
                    rho_h, moment_h = compute_rbf_features(token_xyz, head_xyz, config=env_config, box_A=box_A)
                    if getattr(env_config, "compute_lipids", False):
                        scalar_parts.append(rho_h)
                    if getattr(env_config, "compute_lipid_vectors", False):
                        vector_parts.append(moment_h)

                if want_lipid_tail:
                    tail_xyz = (traj.xyz[frame, tail_idx, :] * 10.0).astype(np.float32) if tail_idx.size else np.zeros((0, 3), dtype=np.float32)
                    rho_t, moment_t = compute_rbf_features(token_xyz, tail_xyz, config=env_config, box_A=box_A)
                    if getattr(env_config, "compute_lipid_tail", False):
                        scalar_parts.append(rho_t)
                    if getattr(env_config, "compute_lipid_tail_vectors", False):
                        vector_parts.append(moment_t)

                if getattr(env_config, "compute_membrane_frame", False):
                    head_xyz = (traj.xyz[frame, head_idx, :] * 10.0).astype(np.float32) if head_idx.size else np.zeros((0, 3), dtype=np.float32)
                    normal, center, thickness = compute_membrane_frame(head_xyz, config=env_config, box_A=box_A)
                    depth = compute_membrane_depth(
                        token_xyz,
                        membrane_center_A=center,
                        membrane_normal=normal,
                        membrane_thickness_A=thickness,
                        normalize=bool(getattr(env_config, "normalize_membrane_depth", True)),
                    )
                    scalar_parts.append(depth)
                    normal_field = np.broadcast_to(normal.reshape(1, 3, 1), (num_tokens, 3, 1)).astype(np.float32)
                    vector_parts.append(normal_field)

                frame_scalar = np.concatenate(scalar_parts, axis=-1).astype(np.float32, copy=False) if scalar_parts else None
                frame_vector = np.concatenate(vector_parts, axis=-1).astype(np.float32, copy=False) if vector_parts else None

                if ema_alpha is not None:
                    if frame_scalar is not None:
                        if ema_scalar_prev is None:
                            ema_scalar_prev = frame_scalar
                        else:
                            ema_scalar_prev = (ema_alpha * ema_scalar_prev) + ((1.0 - ema_alpha) * frame_scalar)
                        frame_scalar = ema_scalar_prev
                    if frame_vector is not None:
                        if ema_vector_prev is None:
                            ema_vector_prev = frame_vector
                        else:
                            ema_vector_prev = (ema_alpha * ema_vector_prev) + ((1.0 - ema_alpha) * frame_vector)
                        frame_vector = ema_vector_prev

                if frame_scalar is not None:
                    env_scalar_frames.append(frame_scalar)
                if frame_vector is not None:
                    env_vector_frames.append(frame_vector)

            env_scalar = np.asarray(env_scalar_frames, dtype=np.float32) if env_scalar_frames else None
            env_vector = np.asarray(env_vector_frames, dtype=np.float32) if env_vector_frames else None

            if env_scalar is not None:
                torch.save(torch.from_numpy(env_scalar), target_dir / f"{Path(traj_path).stem}_env_scalar.pt")
            if env_vector is not None:
                torch.save(torch.from_numpy(env_vector), target_dir / f"{Path(traj_path).stem}_env_vector.pt")

            meta_target = target_dir / f"{Path(traj_path).stem}_env_meta.json"
            centers = rbf_centers_A(env_config).tolist() if env_config is not None else None
            meta = {
                "enabled": True,
                "backend": "mdtraj",
                "water_selection_mdtraj": getattr(env_config, "water_selection_mdtraj", None),
                "lipid_head_selection_mdtraj": getattr(env_config, "lipid_head_selection_mdtraj", None),
                "lipid_tail_selection_mdtraj": getattr(env_config, "lipid_tail_selection_mdtraj", None),
                "r_max_A": getattr(env_config, "r_max_A", None),
                "n_rbf": getattr(env_config, "n_rbf", None),
                "rbf_width_A": getattr(env_config, "rbf_width_A", None),
                "rbf_centers_A": centers,
                "use_pbc": getattr(env_config, "use_pbc", None),
                "compute_water": getattr(env_config, "compute_water", None),
                "compute_water_vectors": getattr(env_config, "compute_water_vectors", None),
                "compute_lipids": getattr(env_config, "compute_lipids", None),
                "compute_lipid_vectors": getattr(env_config, "compute_lipid_vectors", None),
                "compute_lipid_tail": getattr(env_config, "compute_lipid_tail", None),
                "compute_lipid_tail_vectors": getattr(env_config, "compute_lipid_tail_vectors", None),
                "compute_membrane_frame": getattr(env_config, "compute_membrane_frame", None),
                "normalize_membrane_depth": getattr(env_config, "normalize_membrane_depth", None),
                "ema_alpha": getattr(env_config, "ema_alpha", None),
            }
            with open(meta_target, "w") as f:
                json.dump(meta, f, indent=4)

        # Save CA coordinates if available.
        if ca_array is not None and ca_array.size:
            torch.save(torch.from_numpy(ca_array), target_dir / f"{Path(traj_path).stem}_ca.pt")

        atom_numbers = np.asarray(atom_numbers)
        if atomic_numbers is None:
            atomic_numbers = atom_numbers
        elif not np.array_equal(atomic_numbers, atom_numbers):
            raise ValueError("Atomic numbers differ across trajectories; cannot batch inference.")

        mapping = default_cg if default_cg is not None else segment_counts

        infer_traj(
            model=model,
            hidden_channels=resolved_hidden,
            data=[positions_A.astype(np.float32)],
            atomic_numbers=atomic_numbers,
            device=resolved_device,
            saving_path=str(target_dir),
            batch_size=batch_size,
            reduction=reduction,
            cg_mapping=mapping,
            file_name_list=[Path(traj_path).stem],
            show_progress=False,
            log_saves=False,
        )

        computed.append({"source_path": traj_path, "output_path": target, "frames": positions_A.shape[0]})

    summary_data = {
        "computed": [
            {"source_path": item["source_path"], "output_path": str(item["output_path"]), "frames": item["frames"]}
            for item in computed
        ],
        "skipped": [str(p) for p in skipped],
    }

    summary_file = output_path / "inference_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary_data, f, indent=4)

    return {"computed": computed, "skipped": skipped}
