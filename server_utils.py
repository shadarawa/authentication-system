# server_utils.py
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())

def ensure_parent(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)

class FileLock:
    def __init__(self, target_path: str, timeout: float = 5.0, poll: float = 0.05):
        self.target_path = str(target_path)
        self.lock_path = self.target_path + ".lock"
        self.timeout = timeout
        self.poll = poll
        self._acquired = False

    def acquire(self) -> None:
        start = time.time()
        while True:
            try:
                fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, str(os.getpid()).encode("utf-8"))
                os.close(fd)
                self._acquired = True
                return
            except FileExistsError:
                if time.time() - start > self.timeout:
                    raise TimeoutError("Could not acquire lock for %s" % self.target_path)
                time.sleep(self.poll)

    def release(self) -> None:
        if self._acquired:
            try:
                os.remove(self.lock_path)
            except Exception:
                pass
            self._acquired = False

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.release()

def read_json(path: str, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return default if default is not None else {}
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def write_json_atomic(path: str, data: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(p)

def append_jsonl(path: str, obj: Dict[str, Any]) -> None:
    ensure_parent(path)
    line = json.dumps(obj, ensure_ascii=False)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def gen_request_id() -> str:
    return uuid.uuid4().hex
