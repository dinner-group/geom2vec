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


def extract_mda_info(protein, stride: int = 1, selection: Optional[str] = None):
    """Extract atomic positions, atomic numbers, and segment counts from an MDAnalysis Universe."""

    _require_mdanalysis()

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

    positions = [protein_residues.positions.copy() for _ in protein.trajectory]
    positions = np.asarray(positions)[::stride]
    segment_counts = count_segments(protein_residues.resids)

    return positions, np.array(atomic_numbers), np.array(segment_counts)


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
        positions, atomic_numbers, segment_counts = extract_mda_info(universe, stride=stride, selection=selection)
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
            positions, numbers, segment_counts = extract_mda_info(
                universe,
                stride=stride,
                selection=selection,
            )
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
