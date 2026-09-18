"""JSONL episode log, results export, and the MCAP bridge."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, Iterator, Optional

from robotruth.schema.episode import EpisodeRecord

MCAP_TOPIC = "/robotruth/episode"
MCAP_SCHEMA_NAME = "robotruth.EpisodeRecord"


class EpisodeLog:
    """Append-only JSONL file, one EpisodeRecord per line."""

    def __init__(self, path: Path | str):
        self.path = Path(path)

    def append(self, rec: EpisodeRecord) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(rec.model_dump_json() + "\n")

    def extend(self, recs: Iterable[EpisodeRecord]) -> None:
        for r in recs:
            self.append(r)

    def __iter__(self) -> Iterator[EpisodeRecord]:
        if not self.path.exists():
            return iter(())
        return (EpisodeRecord.model_validate_json(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip())

    def records(self) -> list[EpisodeRecord]:
        return list(self)

    def validate(self) -> tuple[int, list[tuple[int, str]]]:
        """Return (n_ok, [(line_no, error)])."""
        ok, errors = 0, []
        if not self.path.exists():
            return 0, [(0, f"{self.path} does not exist")]
        for i, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                EpisodeRecord.model_validate_json(line)
                ok += 1
            except Exception as e:  # noqa: BLE001
                errors.append((i, str(e).splitlines()[0][:200]))
        return ok, errors

    def to_results_csv(self, out: Path | str) -> Path:
        out = Path(out)
        rows = [r.to_result_row() for r in self if r.outcome.success is not None]
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as f:
            if not rows:
                f.write("episode_id,policy,task,success\n")
                return out
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        return out

    # MCAP bridge -----------------------------------------------------------------
    def to_mcap(self, out: Path | str, t0_ns: Optional[int] = None) -> Path:
        """Write records as JSON messages on /robotruth/episode. Requires the `mcap` package."""
        try:
            from mcap.writer import Writer
        except ImportError as e:  # pragma: no cover
            raise ImportError("pip install mcap") from e
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        schema_json = json.dumps(EpisodeRecord.model_json_schema()).encode()
        with out.open("wb") as f:
            w = Writer(f)
            w.start(profile="", library="robotruth")
            schema_id = w.register_schema(name=MCAP_SCHEMA_NAME, encoding="jsonschema", data=schema_json)
            chan_id = w.register_channel(topic=MCAP_TOPIC, message_encoding="json", schema_id=schema_id)
            for i, rec in enumerate(self):
                ts = _record_time_ns(rec) if t0_ns is None else t0_ns + i
                w.add_message(channel_id=chan_id, log_time=ts, publish_time=ts, data=rec.model_dump_json().encode(), sequence=i)
            w.finish()
        return out

    @classmethod
    def from_mcap(cls, mcap_path: Path | str, out: Path | str) -> "EpisodeLog":
        try:
            from mcap.reader import make_reader
        except ImportError as e:  # pragma: no cover
            raise ImportError("pip install mcap") from e
        log = cls(out)
        with Path(mcap_path).open("rb") as f:
            reader = make_reader(f)
            for _schema, channel, message in reader.iter_messages(topics=[MCAP_TOPIC]):
                log.append(EpisodeRecord.model_validate_json(message.data))
        return log


def _record_time_ns(rec: EpisodeRecord) -> int:
    from datetime import datetime
    src = rec.timing.t_start_utc or rec.created_at
    try:
        dt = datetime.fromisoformat(src.replace("Z", "+00:00"))
        return int(dt.timestamp() * 1e9)
    except Exception:  # noqa: BLE001
        return 0
