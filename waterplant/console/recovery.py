"""Recovery status surfaced to operators through the console."""

from __future__ import annotations

from waterplant.store import Store

from .runtime import Runtime


def collect(store: Store) -> dict[str, object]:
    """Return the current phase and the full audit history of recovery runs."""

    report = store.recovery_report()
    payload: dict[str, object] = {
        "phase": store.phase,
        "ready": store.ready(),
        "last_sequence": store.last_sequence,
        "failure_reason": store.failure_reason,
        "policy": {
            "flush": "every operation is fsynced before acknowledgement; writes are never batched",
            "recovery": "automatic on startup, no manual confirmation required",
            "discard_rule": "only the torn tail of the final segment is discarded automatically",
        },
    }
    if report is not None:
        details = report.as_dict()
        payload.update(
            {
                "outcome": details["outcome"],
                "detail": details["detail"],
                "at": details["at"],
                "recovered_to_sequence": details["recovered_to_sequence"],
                "segments": details["segments"],
                "applied_operations": details["applied_operations"],
                "discarded": details["discarded"],
                "migrated_from_legacy": details["migrated_from_legacy"],
                "migrated_keys": details["migrated_keys"],
                "history": details["history"],
            }
        )
        payload["last_recovery"] = details
    return payload


def collect_runtime(rt: Runtime) -> dict[str, object]:
    return collect(rt.store)
