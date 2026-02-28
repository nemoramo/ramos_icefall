from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, List, Optional, Sequence

from icefall.training.zipformer import yaml_entrypoint as _yaml


def _parse_args(argv: Optional[Sequence[str]] = None) -> tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Unified Zipformer Hydra entrypoint (Hydra compose -> argv -> recipe main).",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default=None,
        help=(
            "Hydra config directory (contains config groups). "
            "Either provide this + --config-name, or use --config <path/to/config.yaml>."
        ),
    )
    parser.add_argument(
        "--config-name",
        type=str,
        default=None,
        help="Hydra config name (file stem without .yaml), e.g. 'config'.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Convenience: path to a single Hydra config YAML file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print resolved config + computed recipe argv; do not start training.",
    )
    args, overrides = parser.parse_known_args(list(argv) if argv is not None else None)

    # Hydra-specific CLI flags (e.g. -m/--multirun) are not supported in this entrypoint.
    if "-m" in overrides or "--multirun" in overrides:
        raise ValueError(
            "Hydra multirun (-m/--multirun) is not supported by this entrypoint. "
            "Use an external loop, or extend the entrypoint to use Hydra launchers/sweepers."
        )
    return args, [str(x) for x in overrides]


def _load_hydra_cfg(
    *, config_path: Path, config_name: str, overrides: Sequence[str]
) -> Any:
    try:
        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Missing dependency: hydra-core. Install it via `pip install hydra-core` "
            "(or `pip install -r requirements.txt`)."
        ) from e

    try:
        from omegaconf import OmegaConf
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Missing dependency: omegaconf. Install it via `pip install omegaconf` "
            "(or `pip install -r requirements.txt`)."
        ) from e

    # Ensure Hydra global state is clean (important when calling from notebooks/tests).
    try:
        GlobalHydra.instance().clear()
    except Exception:
        pass

    cfg_dir = Path(config_path).resolve()
    if not cfg_dir.is_dir():
        raise FileNotFoundError(f"Hydra config directory not found: {cfg_dir}")

    with initialize_config_dir(config_dir=str(cfg_dir), version_base=None):
        cfg = compose(config_name=str(config_name), overrides=list(overrides))

    # Best-effort: resolve interpolations early so we fail fast on invalid refs.
    # We still keep the original cfg for dumping with resolve=False.
    _ = OmegaConf.to_container(cfg, resolve=True)
    return cfg


def main(argv: Optional[Sequence[str]] = None) -> None:
    args, overrides = _parse_args(argv)

    config_path = args.config_path
    config_name = args.config_name

    if args.config is not None:
        p = Path(str(args.config)).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f"--config file not found: {p}")
        config_path = str(p.parent)
        config_name = p.stem

    if not config_path or not config_name:
        raise ValueError(
            "Missing Hydra config. Provide --config-path <dir> --config-name <name>, "
            "or --config <path/to/config.yaml>."
        )

    cfg = _load_hydra_cfg(
        config_path=Path(config_path), config_name=str(config_name), overrides=overrides
    )
    from omegaconf import OmegaConf

    merged_cfg_text = OmegaConf.to_yaml(cfg, resolve=False)
    resolved_cfg_text = OmegaConf.to_yaml(cfg, resolve=True)

    locator = _yaml._get_recipe_locator(cfg)
    work_dir = locator.work_dir.resolve()
    entry_path = (work_dir / locator.entrypoint).resolve()
    if not entry_path.is_file():
        raise FileNotFoundError(f"Recipe entrypoint not found: {entry_path}")

    recipe_argv = _yaml._build_recipe_argv(cfg)
    _yaml._maybe_write_config_dumps(
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
        print("\n=== Hydra overrides ===")
        print(" ".join(overrides))
        return

    with _yaml._temporary_cwd(work_dir), _yaml._temporary_sys_path(
        [entry_path.parent, work_dir]
    ):
        mod = _yaml._import_module_from_path(entry_path)
        if not hasattr(mod, "main"):
            raise AttributeError(f"Recipe entrypoint has no main(): {entry_path}")
        _yaml._call_recipe_main(mod.main, argv=recipe_argv, prog=str(entry_path))


if __name__ == "__main__":
    main()

