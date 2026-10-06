"""Command line entry point for the control service."""

from __future__ import annotations

import argparse
import logging
from socketserver import ThreadingMixIn
from wsgiref.simple_server import WSGIServer, make_server

from .console.history import EVENT_KEY
from .console.seed import seed_defaults
from .console.server import Server
from .console.version import version_string
from .store import Store, append_event

LOGGER = logging.getLogger("waterplant")


class ThreadingWSGIServer(ThreadingMixIn, WSGIServer):
    """Serve concurrent control requests from a pool of worker threads."""

    daemon_threads = True


def parse_address(value: str) -> tuple[str, int]:
    host, separator, port = value.rpartition(":")
    if not separator or not port.isdigit():
        raise argparse.ArgumentTypeError("address must look like host:port")
    return host or "0.0.0.0", int(port)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="waterplant", description="control service")
    parser.add_argument("--store", default="state.json", help="file path for the persistent store")
    parser.add_argument("--addr", default=":8080", help="listen address as host:port")
    parser.add_argument("--version", action="store_true", help="print the build version and exit")
    return parser


def report_recovery(store: Store) -> None:
    """Log the startup recovery outcome and audit any discarded tail."""

    report = store.recovery_report()
    torn = report.get("torn")
    if isinstance(torn, dict):
        LOGGER.warning(
            "recovered to seq %s; discarded incomplete tail of segment %s at offset %s "
            "(%s bytes: %s); quarantined to %s",
            report.get("recovered_seq"),
            torn.get("segment"),
            torn.get("offset"),
            torn.get("discarded_bytes"),
            torn.get("detail"),
            torn.get("quarantine"),
        )
        append_event(
            store,
            EVENT_KEY,
            "recovery",
            f"discarded torn tail of {torn.get('segment')} at offset {torn.get('offset')}",
        )
    else:
        LOGGER.info(
            "store recovered: status=%s seq=%s replayed=%s",
            report.get("status"),
            report.get("recovered_seq"),
            report.get("frames_replayed"),
        )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.version:
        print(version_string())
        return 0

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    store = Store.open(args.store)
    try:
        report_recovery(store)
        seed_defaults(store)
        host, port = parse_address(args.addr)
        application = Server(store)
        with make_server(host, port, application, server_class=ThreadingWSGIServer) as httpd:
            LOGGER.info("control console listening on %s:%d", host, port)
            httpd.serve_forever()
    except KeyboardInterrupt:
        LOGGER.info("control console stopped")
    finally:
        store.close()
    return 0
