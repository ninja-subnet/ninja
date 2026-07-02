#!/usr/bin/env python3
"""Test-only submission script: fakes wallet, hotkey, coldkey and signatures.

Same request shape as the real submitter, but no bittensor wallet is loaded.
Pass any --hotkey / --coldkey; signatures are deterministic dummies.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import requests
from pathlib import Path
from typing import Any


DEFAULT_API_URL = "https://api.ninja66.ai/submissions/create"
USER_AGENT = "ninja66-private-submission/3.0-test"
MAX_TOTAL_BYTES = 5_000_000
MAX_AGENT_FILES = 32
PRIVATE_SUBMISSION_RE = re.compile(r"^private-submission:[A-Za-z0-9_.-]{1,128}:[0-9a-f]{64}$")
ENTRYPOINT = "agent.py"
MANIFEST_FILENAME = "tau_agent_files.json"

DUMMY_HOTKEY = "alskdfjainfoaieno"
DUMMY_COLDKEY = "kaboiafea"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TEST submit (dummy signatures, no wallet).")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--agent", type=Path, help="Path to a single agent.py.")
    source.add_argument("--bundle", type=Path, help="Harness directory containing agent.py. Defaults to repo root.")
    parser.add_argument("--api-url", default=os.getenv("NINJA_SUBMISSION_API", DEFAULT_API_URL))
    parser.add_argument("--submission-id", help="Optional stable submission id.")
    parser.add_argument("--hotkey", default=DUMMY_HOTKEY, help="Any hotkey SS58 string.")
    parser.add_argument("--coldkey", default=DUMMY_COLDKEY, help="Any coldkey SS58 string.")
    parser.add_argument("--agent-username", help="Optional public display username.")
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

        hotkey = args.hotkey
        agent_sha256 = agent_bundle_sha256(agent_files)
        submission_id = args.submission_id or derive_submission_id(hotkey=hotkey, agent_sha256=agent_sha256)
        signature_payload = private_submission_signature_payload(
            hotkey=hotkey, submission_id=submission_id, agent_sha256=agent_sha256,
        )
        identity = build_username_identity(args=args)

        print_request_summary(
            source_label=str(args.agent or args.bundle),
            agent_files=agent_files, hotkey=hotkey, submission_id=submission_id,
            agent_sha256=agent_sha256, signature_payload=signature_payload, identity=identity,
        )

        if args.dry_run:
            print("dry_run: true")
            return 0

        req_json = build_request_json(
            source_label=str(args.agent or args.bundle),
            agent_files=agent_files, hotkey=hotkey, submission_id=submission_id,
            agent_sha256=agent_sha256, signature_payload=signature_payload, identity=identity,
        )
        response = post_submission(api_url=args.api_url, req_json=req_json)
        print(json.dumps(response, indent=2, sort_keys=True))
        if not bool(response.get("accepted")):
            return 1
        validate_private_commitment(str(response.get("commitment") or ""))
        return 0

    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def dummy_signature(payload: bytes) -> str:
    """Deterministic 64-byte (128 hex) fake signature."""
    return (hashlib.sha256(b"sig0:" + payload).hexdigest()
            + hashlib.sha256(b"sig1:" + payload).hexdigest())


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
        return {rel: (resolved / rel).read_text(encoding="utf-8") for rel in sorted(relative_paths)}
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
    total = sum(len(p.encode("utf-8")) + len(c.encode("utf-8")) for p, c in files.items())
    if total > MAX_TOTAL_BYTES:
        raise ValueError(f"submission is {total} bytes; maximum is {MAX_TOTAL_BYTES} bytes")


def agent_bundle_sha256(files: dict[str, str]) -> str:
    if set(files) == {ENTRYPOINT}:
        return hashlib.sha256(files[ENTRYPOINT].encode("utf-8")).hexdigest()
    digest = hashlib.sha256()
    for path in sorted(files):
        content_sha = hashlib.sha256(files[path].encode("utf-8")).hexdigest()
        digest.update(f"{path}\0{content_sha}\n".encode())
    return digest.hexdigest()


def derive_submission_id(*, hotkey: str, agent_sha256: str) -> str:
    safe_hotkey = re.sub(r"[^A-Za-z0-9_.-]", "-", hotkey)[:16] or "hotkey"
    return f"{safe_hotkey}-{agent_sha256[:16]}"


def private_submission_signature_payload(*, hotkey: str, submission_id: str, agent_sha256: str) -> bytes:
    return f"tau-private-submission-v1:{hotkey}:{submission_id}:{agent_sha256.lower()}".encode("utf-8")


def username_signature_payload(username: str) -> bytes:
    return f"tau-agent-submission-username:{username}".encode("utf-8")


def build_username_identity(*, args: argparse.Namespace) -> dict[str, str]:
    username = (args.agent_username or "").strip()
    if not username:
        return {}
    coldkey = (args.coldkey or "").strip() or DUMMY_COLDKEY
    return {
        "agent_username": username,
        "coldkey": coldkey,
        "coldkey_signature": dummy_signature(username_signature_payload(username)),
    }


def print_request_summary(*, source_label, agent_files, hotkey, submission_id,
                          agent_sha256, signature_payload, identity) -> None:
    print(f"source: {source_label}")
    print(f"files: {', '.join(sorted(agent_files))}")
    print(f"hotkey: {hotkey}")
    print(f"submission_id: {submission_id}")
    print(f"agent_sha256: {agent_sha256}")
    print(f"signature_payload: {signature_payload.decode('utf-8')}")
    print(f"signature (dummy): {dummy_signature(signature_payload)}")
    if identity:
        print(f"agent_username: {identity['agent_username']}")
        print(f"coldkey: {identity['coldkey']}")


def build_request_json(source_label, agent_files, hotkey, submission_id,
                       agent_sha256, signature_payload, identity) -> dict:
    req = {
        "source": source_label,
        "files": agent_files,
        "hotkey": hotkey,
        "submission_id": submission_id,
        "agent_sha256": agent_sha256,
        "signature_payload": signature_payload.decode("utf-8"),
        "signature": dummy_signature(signature_payload),
    }
    if identity:
        req["identity"] = {
            "agent_username": identity["agent_username"],
            "coldkey": identity["coldkey"],
            "username_signature_payload": username_signature_payload(identity["agent_username"]).decode("utf-8"),
            "coldkey_signature": identity["coldkey_signature"],
        }
    return req


def post_submission(api_url: str, req_json: dict) -> dict[str, Any]:
    headers = {"Content-Type": "application/json", "accept": "application/json", "agent": USER_AGENT}
    return requests.post(api_url, json={"submission": req_json}, headers=headers).json()


def validate_private_commitment(commitment: str) -> None:
    if not PRIVATE_SUBMISSION_RE.fullmatch(commitment):
        raise ValueError("accepted API response did not include a valid private-submission commitment")


if __name__ == "__main__":
    sys.exit(main())
