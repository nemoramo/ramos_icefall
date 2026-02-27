from __future__ import annotations

import argparse
import contextlib
import hashlib
import inspect
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union


@dataclass(frozen=True)
class RecipeLocator:
    """Where to run/import a recipe entrypoint from."""

    work_dir: Path
    entrypoint: Path  # relative to work_dir


def _as_cli_flag(key: str) -> str:
    k = str(key).strip()
    if k.startswith("-"):
        return k
    return "--" + k.replace("_", "-")


def _stringify_cli_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _extend_argv_from_mapping(argv: List[str], items: Iterable[Tuple[str, Any]]) -> None:
    for key, value in items:
        if value is None:
            continue
        flag = _as_cli_flag(key)
        if isinstance(value, (list, tuple)):
            for v in value:
                if v is None:
                    continue
                argv.extend([flag, _stringify_cli_value(v)])
        else:
            argv.extend([flag, _stringify_cli_value(value)])


def _build_recipe_argv(cfg: Any) -> List[str]:
    """Build argv for the recipe entrypoint from a merged OmegaConf config.

    Supported config layouts:
      - ddp: {world_size, dist_backend, master_port, ...}  -> converted to flags
      - exp: {exp_dir, ...}                                -> converted to flags
      - cli: a dict of extra flags, or a list of raw argv strings
    """
    argv: List[str] = []

    ddp = getattr(cfg, "ddp", None)
    if ddp is not None:
        _extend_argv_from_mapping(argv, list(ddp.items()))

    exp = getattr(cfg, "exp", None)
    if exp is not None:
        _extend_argv_from_mapping(argv, list(exp.items()))

    cli = getattr(cfg, "cli", None)
    if cli is None:
        return argv

    if isinstance(cli, (list, tuple)):
        argv.extend([str(x) for x in cli])
        return argv

    if isinstance(cli, dict):
        for k in sorted(cli.keys(), key=str):
            _extend_argv_from_mapping(argv, [(k, cli[k])])
        return argv

    # OmegaConf DictConfig/ListConfig are not plain dict/list; fall back to items().
    if hasattr(cli, "items"):
        for k, v in sorted(list(cli.items()), key=lambda kv: str(kv[0])):
            _extend_argv_from_mapping(argv, [(k, v)])
        return argv
    if hasattr(cli, "__iter__"):
        argv.extend([str(x) for x in list(cli)])
        return argv

    raise TypeError(f"Unsupported cli section type: {type(cli)}")


def _find_flag_value(argv: Sequence[str], flag: str) -> Optional[str]:
    """Return the last value for `flag` in argv (supports --flag v and --flag=v)."""
    out: Optional[str] = None
    for i, a in enumerate(argv):
        if a == flag and i + 1 < len(argv):
            out = str(argv[i + 1])
        elif a.startswith(flag + "="):
            out = a.split("=", 1)[1]
    return out


def _best_effort_git_rev(cwd: Path) -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(cwd), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
        return out or None
    except Exception:
        return None


@contextlib.contextmanager
def _temporary_cwd(path: Path):
    old = Path.cwd()
    os.chdir(str(path))
    try:
        yield
    finally:
        os.chdir(str(old))


@contextlib.contextmanager
def _temporary_sys_path(paths: Sequence[Path]):
    old = list(sys.path)
    try:
        for p in reversed(paths):
            sys.path.insert(0, str(p))
        yield
    finally:
        sys.path[:] = old


def _import_module_from_path(py_file: Path) -> Any:
    import importlib.util

    h = hashlib.sha1(str(py_file).encode("utf-8")).hexdigest()[:12]
    mod_name = f"icefall_recipe_entry_{h}"
    spec = importlib.util.spec_from_file_location(mod_name, str(py_file))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to create import spec for: {py_file}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _call_recipe_main(main_fn: Any, *, argv: Sequence[str], prog: str) -> None:
    """Call a recipe's main() with argv if supported, else via sys.argv."""
    try:
        sig = inspect.signature(main_fn)
        if len(sig.parameters) >= 1:
            main_fn(list(argv))
            return
    except Exception:
        pass

    old_argv = list(sys.argv)
    sys.argv = [prog] + list(argv)
    try:
        main_fn()
    finally:
        sys.argv = old_argv


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Unified Zipformer YAML entrypoint (OmegaConf-based).",
    )
    parser.add_argument(
        "--config",
        action="append",
        required=True,
        help="YAML config path (repeatable; later files override earlier ones).",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="OmegaConf dotlist override, e.g. ddp.world_size=8 or cli.max_duration=1800.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print resolved config + computed recipe argv; do not start training.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def _load_omegaconf(config_paths: Sequence[str], overrides: Sequence[str]) -> Any:
    try:
        from omegaconf import OmegaConf
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Missing dependency: omegaconf. Install it via `pip install omegaconf` "
            "(or `pip install -r requirements.txt`)."
        ) from e

    cfg = OmegaConf.create()
    for p in config_paths:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(p))
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    return cfg


def _get_recipe_locator(cfg: Any) -> RecipeLocator:
    recipe = getattr(cfg, "recipe", None)
    if recipe is None:
        raise ValueError("Missing required 'recipe' section in YAML config.")
    if not hasattr(recipe, "get"):
        raise TypeError(
            "Expected config.recipe to be a mapping with keys: work_dir, entrypoint."
        )

    work_dir = Path(str(recipe.get("work_dir", "")).strip())
    if not str(work_dir):
        raise ValueError("Missing required 'recipe.work_dir' in YAML config.")

    entrypoint = str(recipe.get("entrypoint", "zipformer/train.py")).strip()
    if not entrypoint:
        entrypoint = "zipformer/train.py"
    return RecipeLocator(work_dir=work_dir, entrypoint=Path(entrypoint))


def _maybe_write_config_dumps(
    *,
    cfg: Any,
    resolved_cfg_text: str,
    merged_cfg_text: str,
    work_dir: Path,
    recipe_argv: Sequence[str],
) -> None:
    exp_dir_str = _find_flag_value(recipe_argv, "--exp-dir")
    if not exp_dir_str:
        return
    exp_dir = Path(exp_dir_str)
    if not exp_dir.is_absolute():
        exp_dir = work_dir / exp_dir
    exp_dir.mkdir(parents=True, exist_ok=True)

    (exp_dir / "config.merged.yaml").write_text(merged_cfg_text, encoding="utf-8")
    (exp_dir / "config.resolved.yaml").write_text(resolved_cfg_text, encoding="utf-8")
    (exp_dir / "argv.txt").write_text(" ".join(recipe_argv) + "\n", encoding="utf-8")

    rev = _best_effort_git_rev(work_dir)
    if rev is not None:
        (exp_dir / "git_rev.txt").write_text(rev + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)

    cfg = _load_omegaconf(args.config, args.overrides)
    from omegaconf import OmegaConf

    merged_cfg_text = OmegaConf.to_yaml(cfg, resolve=False)
    resolved_cfg_text = OmegaConf.to_yaml(cfg, resolve=True)

    locator = _get_recipe_locator(cfg)
    work_dir = locator.work_dir.resolve()
    entry_path = (work_dir / locator.entrypoint).resolve()
    if not entry_path.is_file():
        raise FileNotFoundError(f"Recipe entrypoint not found: {entry_path}")

    recipe_argv = _build_recipe_argv(cfg)
    _maybe_write_config_dumps(
        cfg=cfg,
        resolved_cfg_text=resolved_cfg_text,
        merged_cfg_text=merged_cfg_text,
        work_dir=work_dir,
        recipe_argv=recipe_argv,
    )

    if args.dry_run:
        print("=== Resolved config ===")
        print(resolved_cfg_text.rstrip())
        print("\n=== Recipe entrypoint ===")
        print(str(entry_path))
        print("\n=== Recipe argv ===")
        print(" ".join(recipe_argv))
        return

    # Most recipes assume:
    #   - CWD is the recipe's ASR directory (for relative paths),
    #   - sys.path includes the directory of the entry script (for local imports).
    with _temporary_cwd(work_dir), _temporary_sys_path([entry_path.parent, work_dir]):
        mod = _import_module_from_path(entry_path)
        if not hasattr(mod, "main"):
            raise AttributeError(f"Recipe entrypoint has no main(): {entry_path}")
        _call_recipe_main(mod.main, argv=recipe_argv, prog=str(entry_path))


if __name__ == "__main__":
    main()
