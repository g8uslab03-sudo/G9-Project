"""Autonomous AI Private System - 獨立全自主 AI 私有系統。"""

from __future__ import annotations

import argparse
import signal
import sys
import time

from autonomous_ai.config import AutonomousConfig
from autonomous_ai.core import AutonomousCore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Autonomous AI Private System")
    parser.add_argument("--cycles", type=int, default=None,
                        help="執行週期數，預設由設定決定（None = 無限）")
    parser.add_argument("--interval", type=float, default=None,
                        help="週期間隔秒數，覆蓋設定")
    parser.add_argument("--no-learning", action="store_true", help="停用自主學習")
    parser.add_argument("--no-detection", action="store_true", help="停用自主檢測")
    parser.add_argument("--no-healing", action="store_true", help="停用自主修復")
    parser.add_argument("--no-evolution", action="store_true", help="停用自主進化")
    parser.add_argument("--data-dir", type=str, default="./data",
                        help="資料目錄")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = AutonomousConfig(data_dir=args.data_dir,
                              memory_path=f"{args.data_dir}/memory.json",
                              state_path=f"{args.data_dir}/state.json",
                              log_path=f"{args.data_dir}/autonomous.log")
    if args.interval is not None:
        config.cycle_interval_seconds = args.interval
    config.learning_enabled = not args.no_learning
    config.detection_enabled = not args.no_detection
    config.healing_enabled = not args.no_healing
    config.evolution_enabled = not args.no_evolution

    core = AutonomousCore(config)

    stop_requested = False

    def handle_signal(signum, frame):  # noqa: ARG001
        nonlocal stop_requested
        stop_requested = True
        core.stop()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    print("Autonomous AI Private System 啟動 (全自主，無人工介入)")
    print(f"設定: {config.to_dict()}")

    max_cycles = args.cycles if args.cycles is not None else config.max_cycles
    core.run(max_cycles=max_cycles)

    print("\n系統停止。最終狀態:")
    for key, value in core.status().items():
        print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
