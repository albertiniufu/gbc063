"""Valida e empacota a submissão do time.

    python submit.py
    python submit.py --weights checkpoints/policy.pt --dry-run

O script faz o possível para antecipar problemas: confere as versões
obrigatórias, o `equipe.json`, importa o `agent.py`, carrega os pesos e
roda um episódio curto. A validação oficial do torneio é feita de novo,
em subprocesso isolado; se algo falhar, o time é eliminado.

O resultado é `gbc063_submission.zip`, que deve ser commitado na **raiz**
do repositório GitHub do time.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import signal
import sys
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
REQUIRED = json.loads((ROOT / "versions.json").read_text(encoding="utf-8"))
EXCLUDE_PY = {"train.py", "play.py", "submit.py", "conftest.py"}
MAX_WEIGHTS_MB = 50
MAX_ZIP_MB = 60


class Timeout(Exception):
    pass


def _alarm(seconds: int):
    if not hasattr(signal, "SIGALRM"):
        return None

    def handler(signum, frame):
        raise Timeout()

    previous = signal.signal(signal.SIGALRM, handler)
    signal.alarm(int(seconds))
    return previous


def _cancel_alarm(previous) -> None:
    if previous is not None:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def check_versions(strict: bool = True) -> None:
    problems = []
    python = platform.python_version()
    if python != REQUIRED["python"]:
        problems.append(f"Python {python} (obrigatório {REQUIRED['python']})")
    for package in ("torch", "numpy", "gymnasium", "pygame"):
        try:
            version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            problems.append(f"{package} não instalado (obrigatório "
                            f"{REQUIRED[package]})")
            continue
        base = version.split("+", 1)[0]
        if base != REQUIRED[package]:
            problems.append(f"{package} {version} "
                            f"(obrigatório {REQUIRED[package]})")
    if not problems:
        print("versões: ok")
        return
    message = "versões obrigatórias divergentes:\n  - " + "\n  - ".join(
        problems)
    if strict:
        raise SystemExit(
            message + "\n\nUse o ambiente do README (Python 3.13.18 e "
            "requirements.txt).\nSe for só uma diferença de build CUDA do "
            "torch, rode com --allow-version-mismatch.")
    print("AVISO: " + message)


def load_team(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(
            f"{path} não encontrado. Copie equipe.example.json para "
            "equipe.json e preencha os dados do time.")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{path} não é um JSON válido: {exc}") from exc
    if not isinstance(data, dict):
        raise SystemExit(f"{path} deve conter um objeto JSON")

    team = data.get("time")
    if not isinstance(team, str) or not team.strip() or \
            "NOME DO TIME" in team.upper():
        raise SystemExit(f"{path}: preencha o campo 'time' com o nome do time")

    members = data.get("integrantes")
    if not isinstance(members, list) or not members:
        raise SystemExit(
            f"{path}: 'integrantes' deve ser uma lista com ao menos uma "
            "pessoa")
    for index, member in enumerate(members, start=1):
        if not isinstance(member, dict):
            raise SystemExit(f"{path}: integrante {index} inválido")
        name = member.get("nome")
        matricula = member.get("matricula")
        if not isinstance(name, str) or not name.strip() or \
                "NOME COMPLETO" in name.upper():
            raise SystemExit(f"{path}: nome do integrante {index} vazio")
        if isinstance(matricula, int):
            matricula = str(matricula)
        if not isinstance(matricula, str) or not matricula.strip() or \
                matricula.strip() == "000000":
            raise SystemExit(
                f"{path}: matrícula do integrante {index} vazia")

    changes = data.get("alteracoes")
    if not isinstance(changes, str) or len(changes.strip()) < 10:
        raise SystemExit(
            f"{path}: descreva em 'alteracoes' o que você mudou no agente "
            "e no algoritmo de treino")
    return data


def smoke_test(weights: Path) -> None:
    if not weights.exists():
        raise SystemExit(
            f"pesos não encontrados em {weights}. Treine primeiro:\n"
            "  python train.py --episodes 300")
    size_mb = weights.stat().st_size / (1024 * 1024)
    if size_mb > MAX_WEIGHTS_MB:
        raise SystemExit(f"{weights} tem {size_mb:.1f} MB; o limite é "
                         f"{MAX_WEIGHTS_MB} MB")

    try:
        from agent import Agent
        from tankgrid import TankGridConfig
        from tankgrid.env import TankGridEnv
    except Exception as exc:
        raise SystemExit(f"não foi possível importar agent.py: {exc}") from exc

    previous = _alarm(90)
    try:
        agent = Agent(str(weights), device="cpu")
        config = TankGridConfig(seed=0, opponent="idle", max_steps=12,
                                control="team", auto_upgrade=True,
                                view_size=15)
        env = TankGridEnv(config=config)
        obs, _ = env.reset(seed=0)
        if hasattr(agent, "reset"):
            agent.reset(0)
        for _ in range(12):
            action = agent.act(obs)
            if isinstance(action, bool) or not isinstance(
                    action, (int, np.integer)):
                raise SystemExit(
                    f"Agent.act devolveu {type(action).__name__}; o contrato "
                    "exige um int de 0 a 80")
            action = int(action)
            if not 0 <= action < 81:
                raise SystemExit(
                    f"Agent.act devolveu {action}; o contrato exige 0..80")
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
        env.close()
    except Timeout as exc:
        raise SystemExit(
            "o episódio de validação passou de 90 s; verifique o tempo de "
            "cada ação do agente") from exc
    except SystemExit:
        raise
    except Exception as exc:
        raise SystemExit(f"o agente falhou no episódio de validação: "
                         f"{exc!r}") from exc
    finally:
        _cancel_alarm(previous)
    print("agente: ok (importou, carregou os pesos e agiu)")


def package(team_file: Path, weights: Path, out: Path) -> None:
    if out.exists():
        out.unlink()
    py_files = sorted(
        path for path in ROOT.glob("*.py") if path.name not in EXCLUDE_PY)
    if not any(path.name == "agent.py" for path in py_files):
        raise SystemExit("agent.py não encontrado na raiz do projeto")

    nested = []
    for path in ROOT.rglob("*.py"):
        if path.parent == ROOT or "__pycache__" in path.parts:
            continue
        top = path.relative_to(ROOT).parts[0]
        if top in ("tankgrid", "tests", ".venv", "venv", "env",
                   "checkpoints", ".git", ".pytest_cache", "__pycache__"):
            continue
        nested.append(path)
    if nested:
        print("AVISO: módulos fora da raiz não entram no zip:")
        for path in nested[:8]:
            print(f"  - {path.relative_to(ROOT)}")
        print("  mantenha os arquivos do agente no nível raiz (layout plano).")

    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(team_file, "equipe.json")
        zf.write(weights, "policy.pt")
        for path in py_files:
            zf.write(path, path.name)

    size_mb = out.stat().st_size / (1024 * 1024)
    if size_mb > MAX_ZIP_MB:
        out.unlink()
        raise SystemExit(f"o zip ficou com {size_mb:.1f} MB; o limite é "
                         f"{MAX_ZIP_MB} MB")

    print(f"zip: {out} ({size_mb:.2f} MB)")
    with zipfile.ZipFile(out) as zf:
        for name in zf.namelist():
            print(f"  {name}")
    print("\nColoque gbc063_submission.zip na RAIZ do repositório GitHub do "
          "time e faça commit/push.\n"
          "Checklist:\n"
          "  [ ] modifiquei agent.py e o algoritmo de treino\n"
          "  [ ] equipe.json tem o nome do time e todos os integrantes\n"
          "  [ ] usei as versões obrigatórias do README\n"
          "  [ ] o zip está na raiz do repositório")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--team-file", default="equipe.json")
    parser.add_argument("--weights", default="checkpoints/policy.pt")
    parser.add_argument("--out", default="gbc063_submission.zip")
    parser.add_argument("--dry-run", action="store_true",
                        help="valida sem gerar o zip")
    parser.add_argument("--allow-version-mismatch", action="store_true",
                        help="só avisa sobre versões divergentes")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    print(f"python {platform.python_version()} | projeto {ROOT}")
    check_versions(strict=not args.allow_version_mismatch)
    team_file = Path(args.team_file)
    weights = Path(args.weights)
    team = load_team(team_file)
    print(f"time: {team['time']} ({len(team['integrantes'])} integrante(s))")
    smoke_test(weights)
    if args.dry_run:
        print("dry-run: zip não gerado")
        return
    package(team_file, weights, Path(args.out))


if __name__ == "__main__":
    main()
