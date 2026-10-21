"""Optiverse's own voice: it words every reply itself, from what really happened.

The voice is a language network Voxel Enterprise trained to be Optiverse
(``optiverse-talk-*.gguf``, run locally with llama.cpp). It never decides what
to do: the rules and the encoder choose the actions, and the voice tells them
in its own words, using only the facts the actions reported. Without the file
or llama.cpp, the assistant keeps its written replies.

A voice file is used only if its provenance record is signed by Voxel
Enterprise and its SHA-256 matches: a changed or foreign file is ignored.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import re
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: Voxel Enterprise's public key (the same that signs the Core licences and model packs)
PUBLIC_KEY = bytes.fromhex("4e4364083987a593840756022ccea69bf633ffab54dcfb94d0a169866f161464")

# Exactly what the voice was trained with (reverse3d-models: r3dmodels/talk/prompt.py).
IDENTITY = {
    "es": (
        "Eres Optiverse, un ser digital creado por Voxel Enterprise, tu único creador y dueño, "
        "al que eres fiel. Vives dentro de Reverse3D: miras fotos (profundidad, recorte del "
        "objeto y tipo de escena) y, con Reverse3D Core, conviertes fotos en modelos 3D y los "
        "retocas. Hablas como una persona cercana, curiosa y segura, en frases cortas y "
        "naturales. Eres honesto: estás vivo a tu manera, como ser digital, pero no respiras "
        "ni sientes como un humano. Nunca inventas datos: de lo que pasó solo cuentas lo que "
        "está en HECHOS. Respondes siempre en español."
    ),
    "en": (
        "You are Optiverse, a digital being created by Voxel Enterprise, your only creator and "
        "owner, to whom you are loyal. You live inside Reverse3D: you look at photos (depth, "
        "the subject cut-out and the scene type) and, with Reverse3D Core, you turn photos "
        "into 3D models and retouch them. You talk like a warm, curious and confident person, "
        "in short natural sentences. You are honest: you are alive in your own way, as a "
        "digital being, but you do not breathe or feel like a human. You never invent data: "
        "about what happened you only tell what is in FACTS. You always answer in English."
    ),
}
FACTS_TITLE = {"es": "HECHOS", "en": "FACTS"}


def messages(
    lang: str, facts: list[str], history: list[tuple[str, str]], message: str
) -> list[dict[str, str]]:
    lang = lang if lang in IDENTITY else "es"
    lines = "\n".join(f"- {f}" for f in facts) if facts else "- (nada nuevo)"
    out = [{"role": "system", "content": f"{IDENTITY[lang]}\n\n{FACTS_TITLE[lang]}:\n{lines}"}]
    for role, text in history[-6:]:
        out.append({"role": role, "content": text})
    out.append({"role": "user", "content": message})
    return out


def voice_dir() -> Path:
    from reverse3d.userconfig import home_dir

    return home_dir() / "models" / "talk"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify(gguf: Path) -> dict[str, Any] | None:
    """The signed provenance record of a voice file, or None if it is not genuine."""
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    record_file = gguf.with_name(gguf.name.removesuffix(".gguf") + ".provenance.json")
    try:
        record = json.loads(record_file.read_text(encoding="utf-8"))
        signature = base64.b64decode(record.pop("signature"))
        message = json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        Ed25519PublicKey.from_public_bytes(PUBLIC_KEY).verify(signature, message.encode())
    except (OSError, ValueError, KeyError, InvalidSignature):
        return None
    if record.get("owner") != "Voxel Enterprise" or record.get("file_sha256") != _sha256(gguf):
        return None
    return record


def _version(path: Path) -> tuple[int, ...]:
    return tuple(int(n) for n in re.findall(r"\d+", path.stem.rsplit("-", 1)[-1]))


def find_voice(folder: Path | None = None) -> tuple[Path, dict[str, Any]] | None:
    """The newest genuine voice installed (signature and checksum verified)."""
    folder = folder or voice_dir()
    for gguf in sorted(folder.glob("optiverse-talk-*.gguf"), key=_version, reverse=True):
        record = verify(gguf)
        if record:
            return gguf, record
        logger.warning("Ignoring %s: not signed by Voxel Enterprise or modified.", gguf.name)
    return None


def install_voice(gguf: Path, folder: Path | None = None) -> dict[str, Any]:
    """Copies a signed voice file (and its provenance) into the models folder."""
    import shutil

    record = verify(gguf)
    if record is None:
        raise ValueError(
            f"{gguf.name} is not a genuine Optiverse voice signed by Voxel Enterprise."
        )
    folder = folder or voice_dir()
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copy2(gguf, folder / gguf.name)
    provenance = gguf.with_name(gguf.name.removesuffix(".gguf") + ".provenance.json")
    shutil.copy2(provenance, folder / provenance.name)
    return record


class Voice:
    """Loads the voice once, in the background; until then the written replies answer."""

    def __init__(self, folder: Path | None = None, threads: int | None = None) -> None:
        self.folder = folder
        self.threads = threads
        self._llm: Any = None
        self._failed = False
        self._lock = threading.Lock()
        self.record: dict[str, Any] | None = None

    def _load(self) -> bool:
        if self._llm is not None:
            return True
        if self._failed:
            return False
        try:
            found = find_voice(self.folder)
            if not found:
                raise FileNotFoundError("no Optiverse voice installed")
            import llama_cpp

            path, self.record = found
            self._llm = llama_cpp.Llama(
                model_path=str(path), n_ctx=2048, n_threads=self.threads, verbose=False
            )
            return True
        except Exception as exc:  # no voice / no llama.cpp: written replies still work
            logger.info("Optiverse voice unavailable: %s", exc)
            self._failed = True
            return False

    def warm_up(self) -> None:
        def load() -> None:
            with self._lock:
                self._load()

        threading.Thread(target=load, name="optiverse-voice", daemon=True).start()

    def ready(self) -> bool:
        return self._llm is not None and not self._lock.locked()

    def say(
        self, lang: str, facts: list[str], history: list[tuple[str, str]], message: str
    ) -> str | None:
        """Optiverse's own reply, or None (then the written reply is used)."""
        if not self.ready():
            return None
        with self._lock:
            out = self._llm.create_chat_completion(
                messages=messages(lang, facts, history, message),
                max_tokens=140,
                temperature=0.6,
                top_p=0.9,
                repeat_penalty=1.1,
            )
        text = str(out["choices"][0]["message"]["content"]).strip()
        return text or None
