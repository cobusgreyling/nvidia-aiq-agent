"""Environment validation — checks API keys, dependencies, and data paths before first run."""

import importlib
import os
import sys
from pathlib import Path


def check(label: str, ok: bool, detail: str = "") -> bool:
    status = "OK" if ok else "FAIL"
    msg = f"  [{status}] {label}"
    if detail:
        msg += f"  — {detail}"
    print(msg)
    return ok


def main() -> int:
    print("NeMo AgentIQ — Environment Check\n")
    passed = 0
    failed = 0

    # ── API keys ────────────────────────────────────────────────────────────
    print("API Keys:")
    nvidia_key = os.getenv("NVIDIA_API_KEY", "")
    ok = check(
        "NVIDIA_API_KEY",
        bool(nvidia_key) and nvidia_key != "your-nvidia-nim-api-key-here",
        "required for NIM model calls",
    )
    if ok:
        passed += 1
    else:
        failed += 1

    tavily_key = os.getenv("TAVILY_API_KEY", "")
    ok = check(
        "TAVILY_API_KEY",
        bool(tavily_key) and tavily_key != "your-tavily-api-key-for-web-search",
        "optional — needed only for web search agent",
    )
    if ok:
        passed += 1
    else:
        failed += 1

    # ── Python dependencies ─────────────────────────────────────────────────
    print("\nDependencies:")
    required = [
        ("langchain", "langchain"),
        ("langgraph", "langgraph"),
        ("langchain_nvidia_ai_endpoints", "langchain-nvidia-ai-endpoints"),
        ("langchain_community", "langchain-community"),
        ("faiss", "faiss-cpu"),
        ("streamlit", "streamlit"),
        ("sqlalchemy", "sqlalchemy"),
        ("dotenv", "python-dotenv"),
        ("requests", "requests"),
        ("yaml", "PyYAML"),
    ]
    for module_name, pip_name in required:
        try:
            importlib.import_module(module_name)
            ok = check(pip_name, True)
        except ImportError:
            ok = check(pip_name, False, f"pip install {pip_name}")
        if ok:
            passed += 1
        else:
            failed += 1

    optional = [
        ("tavily", "tavily-python"),
        ("nemoguardrails", "nemoguardrails"),
        ("pypdf", "pypdf"),
    ]
    for module_name, pip_name in optional:
        try:
            importlib.import_module(module_name)
            ok = check(f"{pip_name} (optional)", True)
        except ImportError:
            ok = check(f"{pip_name} (optional)", False, f"pip install {pip_name}")
        if ok:
            passed += 1
        else:
            failed += 1

    # ── Config files ────────────────────────────────────────────────────────
    print("\nConfig Files:")
    for cfg in ["config/models.yaml", "config/guardrails.yaml"]:
        exists = Path(cfg).exists()
        ok = check(cfg, exists, "missing" if not exists else "")
        if ok:
            passed += 1
        else:
            failed += 1

    env_file = Path(".env").exists()
    ok = check(".env", env_file, "copy .env.example to .env and fill in keys" if not env_file else "")
    if ok:
        passed += 1
    else:
        failed += 1

    # ── Data paths ──────────────────────────────────────────────────────────
    print("\nData Paths:")
    docs_dir = Path("data/documents")
    doc_detail = "will be created on first ingest"
    if docs_dir.exists():
        doc_detail = f"{len(list(docs_dir.iterdir()))} file(s)"
    ok = check("data/documents/", docs_dir.exists(), doc_detail)
    if ok:
        passed += 1
    else:
        failed += 1

    vs_path = Path("data/vectorstore")
    ok = check(
        "data/vectorstore/", vs_path.exists(),
        "run 'python ingest.py' to create" if not vs_path.exists() else "",
    )
    if ok:
        passed += 1
    else:
        failed += 1

    db_path = Path("data/sample.db")
    ok = check(
        "data/sample.db", db_path.exists(),
        "run 'python seed_sample_db.py' to create" if not db_path.exists() else "",
    )
    if ok:
        passed += 1
    else:
        failed += 1

    # ── Summary ─────────────────────────────────────────────────────────────
    print(f"\n{'=' * 50}")
    print(f"  {passed} passed, {failed} failed")
    if failed:
        print("  Fix the items above and re-run: python check_env.py")
    else:
        print("  All checks passed! Run: streamlit run app.py")
    return 1 if failed else 0


if __name__ == "__main__":
    # Load .env if available
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    sys.exit(main())
