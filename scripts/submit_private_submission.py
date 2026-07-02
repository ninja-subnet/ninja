#!/usr/bin/env python3
"""Submit agent.py or a multi-file harness to the Subnet 66 private submission API."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import re
import sys
import requests
from pathlib import Path
from typing import Any


DEFAULT_API_URL = "https://ninja66.ai/submissions/create"
USER_AGENT = "ninja66-private-submission/3.0"
MAX_TOTAL_BYTES = 5_000_000
MAX_AGENT_FILES = 32
PRIVATE_SUBMISSION_RE = re.compile(r"^private-submission:[A-Za-z0-9_.-]{1,128}:[0-9a-f]{64}$")
ENTRYPOINT = "agent.py"
MANIFEST_FILENAME = "tau_agent_files.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Submit a private Subnet 66 ninja harness.")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--agent", type=Path, help="Path to a single submitted agent.py (legacy single-file).")
    source.add_argument("--bundle", type=Path,
        help=(
            "Path to a harness directory containing agent.py. Every *.py file under it is "
            f"submitted; if a {MANIFEST_FILENAME} manifest is present, exactly those files are "
            "used. Defaults to this repository's root."
        ),
    )
    parser.add_argument("--api-url", default=os.getenv("NINJA_SUBMISSION_API", DEFAULT_API_URL))
    parser.add_argument("--submission-id", help="Optional stable submission id. Defaults to hotkey/hash derived id.")
    parser.add_argument("--hotkey", help="Expected miner hotkey SS58 address. Defaults to loaded wallet hotkey.")
    parser.add_argument("--wallet-name", default=os.getenv("BT_WALLET_NAME", "test66"))
    parser.add_argument("--wallet-hotkey", default=os.getenv("BT_WALLET_HOTKEY", "test66_hot"))
    parser.add_argument("--wallet-path", default=os.getenv("BT_WALLET_PATH"))
    parser.add_argument("--agent-username", help="Optional public display username for this miner agent.")
    parser.add_argument("--coldkey", help="Coldkey SS58 that owns the submitting hotkey. Defaults to the loaded wallet coldkey.")
    parser.add_argument("--coldkey-signature",
        help="Coldkey signature over tau-agent-submission-username:<agent-username>. Defaults to signing with the loaded wallet coldkey.",)
    parser.add_argument("--dry-run", action="store_true", help="Build and print the request without sending it.")
    args = parser.parse_args()
    if args.agent is None and args.bundle is None:
        args.bundle = Path(__file__).resolve().parents[1]
    return args

def main() -> int:
    args = parse_args()
    try:
        agent_files = load_agent_files(args)
        validate_agent_files(agent_files)

        wallet = load_wallet(args)
        hotkey = wallet.hotkey.ss58_address
        if args.hotkey and args.hotkey != hotkey:
            raise ValueError(f"loaded wallet hotkey {hotkey} does not match --hotkey {args.hotkey}")

        agent_sha256 = agent_bundle_sha256(agent_files)
        submission_id = args.submission_id or derive_submission_id(hotkey=hotkey, agent_sha256=agent_sha256)
        signature_payload = private_submission_signature_payload(
            hotkey=hotkey,
            submission_id=submission_id,
            agent_sha256=agent_sha256,
        )
        signature = sign_payload(wallet, signature_payload)
        identity = build_username_identity(args=args, wallet=wallet)

        print_request_summary(
            source_label=str(args.agent or args.bundle),
            agent_files=agent_files,
            hotkey=hotkey,
            submission_id=submission_id,
            agent_sha256=agent_sha256,
            signature_payload=signature_payload,
            identity=identity,
            signature=signature
        )

        if args.dry_run:
            print("dry_run: true")
            return 0

        req_json = build_request_json(
            source_label=str(args.agent or args.bundle),
            agent_files=agent_files,
            hotkey=hotkey,
            submission_id=submission_id,
            agent_sha256=agent_sha256,
            signature_payload=signature_payload,
            identity=identity,
            signature = signature
        )

        response = post_submission(
            api_url=args.api_url,
            req_json=req_json
        )

        if not bool(response.get("accepted")):
            return 1

        commitment = str(response.get("commitment") or "")
        validate_private_commitment(commitment)
        return 0

    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

def load_agent_files(args: argparse.Namespace) -> dict[str, str]:
    if args.bundle is not None:
        return collect_harness_from_directory(args.bundle)
    agent_path = args.agent.expanduser().resolve()
    return {ENTRYPOINT: agent_path.read_text(encoding="utf-8")}

def collect_harness_from_directory(path: Path) -> dict[str, str]:
    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise ValueError(f"--bundle must be a directory: {resolved}")
    manifest_path = resolved / MANIFEST_FILENAME
    if manifest_path.is_file():
        relative_paths = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(relative_paths, list) or not all(isinstance(p, str) for p in relative_paths):
            raise ValueError(f"{MANIFEST_FILENAME} must be a JSON array of relative file paths")
        return {
            relative: (resolved / relative).read_text(encoding="utf-8")
            for relative in sorted(relative_paths)
        }
    files: dict[str, str] = {}
    for file_path in sorted(resolved.rglob("*.py")):
        relative = file_path.relative_to(resolved)
        if any(part.startswith(".") for part in relative.parts):
            continue
        if "scripts" in relative.parts:
            continue
        files[relative.as_posix()] = file_path.read_text(encoding="utf-8")
    return files

def validate_agent_files(files: dict[str, str]) -> None:
    if ENTRYPOINT not in files:
        raise ValueError(f"submission must include `{ENTRYPOINT}` as the agent entrypoint")
    if len(files) > MAX_AGENT_FILES:
        raise ValueError(f"submission has {len(files)} files; the maximum is {MAX_AGENT_FILES}")
    for path in files:
        if path.startswith("/") or "\\" in path or any(seg in {"", ".", ".."} for seg in path.split("/")):
            raise ValueError(f"agent file path `{path}` must be a clean relative POSIX path")
        if not path.endswith(".py"):
            raise ValueError(f"agent file `{path}` must be a Python module")
    total = sum(len(path.encode("utf-8")) + len(content.encode("utf-8")) for path, content in files.items())
    if total > MAX_TOTAL_BYTES:
        raise ValueError(f"submission is {total} bytes; maximum is {MAX_TOTAL_BYTES} bytes")

def agent_bundle_sha256(files: dict[str, str]) -> str:
    """Deterministic submission hash, identical to the validator's.

    Single-file submissions keep the historical sha256-of-agent.py value;
    multi-file submissions hash the sorted path/content-sha lines.
    """
    if set(files) == {ENTRYPOINT}:
        return hashlib.sha256(files[ENTRYPOINT].encode("utf-8")).hexdigest()
    digest = hashlib.sha256()
    for path in sorted(files):
        content_sha = hashlib.sha256(files[path].encode("utf-8")).hexdigest()
        digest.update(f"{path}\0{content_sha}\n".encode())
    return digest.hexdigest()

def load_wallet(args: argparse.Namespace) -> Any:
    try:
        bt = importlib.import_module("bittensor")
    except ImportError as exc:
        raise RuntimeError("bittensor is not installed in this Python environment") from exc
    wallet_kwargs = {"name": args.wallet_name, "hotkey": args.wallet_hotkey}
    if args.wallet_path:
        wallet_kwargs["path"] = args.wallet_path
    return bt.Wallet(**wallet_kwargs)


def derive_submission_id(*, hotkey: str, agent_sha256: str) -> str:
    safe_hotkey = re.sub(r"[^A-Za-z0-9_.-]", "-", hotkey)[:16] or "hotkey"
    return f"{safe_hotkey}-{agent_sha256[:16]}"

def private_submission_signature_payload(*, hotkey: str, submission_id: str, agent_sha256: str) -> bytes:
    return f"tau-private-submission-v1:{hotkey}:{submission_id}:{agent_sha256.lower()}".encode("utf-8")

def sign_payload(wallet: Any, payload: bytes) -> str:
    signature = wallet.hotkey.sign(payload)
    if isinstance(signature, str):
        return signature.removeprefix("0x")
    if isinstance(signature, bytes):
        return signature.hex()
    if hasattr(signature, "hex"):
        return signature.hex()
    raise TypeError(f"unsupported signature type: {type(signature).__name__}")


def username_signature_payload(username: str) -> bytes:
    return f"tau-agent-submission-username:{username}".encode("utf-8")


def wallet_coldkey_address(wallet: Any) -> str | None:
    for attr in ("coldkeypub", "coldkey"):
        value = getattr(wallet, attr, None)
        ss58 = getattr(value, "ss58_address", None)
        if ss58:
            return str(ss58)
    return None


def sign_with_coldkey(wallet: Any, payload: bytes) -> str:
    coldkey = getattr(wallet, "coldkey", None)
    if coldkey is None or not hasattr(coldkey, "sign"):
        raise RuntimeError("loaded wallet coldkey is not available; pass --coldkey-signature")
    signature = coldkey.sign(payload)
    if isinstance(signature, str):
        return signature.removeprefix("0x")
    if isinstance(signature, bytes):
        return signature.hex()
    if hasattr(signature, "hex"):
        return signature.hex()
    raise TypeError(f"unsupported coldkey signature type: {type(signature).__name__}")


def build_username_identity(*, args: argparse.Namespace, wallet: Any) -> dict[str, str]:
    username = (args.agent_username or "").strip()
    coldkey = (args.coldkey or "").strip()
    if not username:
        return {}
    if not coldkey:
        coldkey = wallet_coldkey_address(wallet) or ""
    if not coldkey:
        raise ValueError("--coldkey is required because the loaded wallet coldkey address was not available")
    return {
        "agent_username": username,
        "coldkey": coldkey,
    }


def print_request_summary(
    *,
    source_label: str,
    agent_files: dict[str, str],
    hotkey: str,
    submission_id: str,
    agent_sha256: str,
    signature_payload: bytes,
    identity: dict[str, str],
    signature: str
) -> None:
    print(f"source: {source_label}")
    print(f"files: {', '.join(sorted(agent_files))}")
    print(f"hotkey: {hotkey}")
    print(f"submission_id: {submission_id}")
    print(f"agent_sha256: {agent_sha256}")
    print(f"signature_payload: {signature_payload.decode('utf-8')}")
    print(f"signature: {signature}")

    if identity:
        print(f"agent_username: {identity['agent_username']}")
        print(f"coldkey: {identity['coldkey']}")
        print(f"username_signature_payload: {username_signature_payload(identity['agent_username']).decode('utf-8')}")

def build_request_json(
    source_label: str,
    agent_files: dict[str, str],
    hotkey: str,
    submission_id: str,
    agent_sha256: str,
    signature_payload: bytes,
    identity: dict[str, str],
    signature: str
) -> dict:

    req = {
        "source": source_label,
        "files": agent_files,
        "hotkey": hotkey,
        "submission_id": submission_id,
        "agent_sha256": agent_sha256,
        "signature_payload": signature_payload.decode('utf-8'),
        "hotkey_signature": signature
        }

    if identity:
        req["identity"] = {
            "agent_username": identity['agent_username'],
            "coldkey": identity['coldkey'],
            }
    return req


def post_submission(
    api_url: str,
    req_json: dict
) -> dict[str, Any]:

    headers={
        "Content-Type": "application/json; charset=utf-8",
        "accept":       "application/json",
        "agent":        USER_AGENT
    }

    return requests.post(api_url, json={"submission": req_json}, headers=headers).json()


def validate_private_commitment(commitment: str) -> None:
    if not PRIVATE_SUBMISSION_RE.fullmatch(commitment):
        raise ValueError("accepted API response did not include a valid private-submission commitment")

if __name__ == "__main__":
    sys.exit(main())
