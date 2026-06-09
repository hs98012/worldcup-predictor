import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
STATUS_PATH = PROJECT_ROOT / "data/processed/update_status.json"
REQUIRED_DERIVED_FILES = [
    PROJECT_ROOT / "data/processed/completed_matches.csv",
    PROJECT_ROOT / "data/processed/upcoming_fixtures.csv",
]

UPDATE_SCRIPT = "00_update_results_csv.py"
PIPELINE_SCRIPTS = [
    "01_load_matches.py",
    "02_calculate_elo.py",
    "03_predict_match.py",
    "04_create_worldcup_matches.py",
    "04_predict_matches.py",
    "05_train_sklearn_model.py",
    "08_predict_upcoming_fixtures.py",
    "06_adjust_draw_probabilities.py",
    "07_calculate_group_standings.py",
    "08_simulate_group_stage.py",
    "09_simulate_tournament.py",
    "10_create_dashboard_summary.py",
    "09_sync_frontend_data.py",
]


def run_script(script_name):
    script_path = SCRIPTS_DIR / script_name

    if not script_path.exists():
        print(f"파일 없음, 스킵: {script_path.relative_to(PROJECT_ROOT)}")
        return

    print(f"\n실행: {script_path.relative_to(PROJECT_ROOT)}", flush=True)
    subprocess.run(
        [sys.executable, str(script_path)],
        cwd=PROJECT_ROOT,
        check=True,
    )


def load_update_status():
    if not STATUS_PATH.exists():
        raise FileNotFoundError(
            f"업데이트 상태 파일을 찾을 수 없습니다: {STATUS_PATH}"
        )

    with STATUS_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def main():
    run_script(UPDATE_SCRIPT)
    status = load_update_status()

    missing_derived_files = [
        path
        for path in REQUIRED_DERIVED_FILES
        if not path.exists()
    ]

    if not status.get("changed", False) and not missing_derived_files:
        print("\n원본 데이터 변경 없음. 이후 파이프라인을 실행하지 않습니다.")
        return

    if status.get("changed", False):
        print("\n원본 데이터 변경 감지. 전체 파이프라인을 실행합니다.")
    else:
        print("\n필수 처리 파일 없음. 전체 파이프라인을 실행합니다.")

    for script_name in PIPELINE_SCRIPTS:
        run_script(script_name)

    print("\n일일 데이터 파이프라인 완료")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        print(
            f"\n스크립트 실행 실패 (종료 코드 {error.returncode}): "
            f"{Path(error.cmd[-1]).name}",
            file=sys.stderr,
        )
        raise SystemExit(error.returncode)
    except Exception as error:
        print(f"\n일일 파이프라인 실패: {error}", file=sys.stderr)
        raise SystemExit(1)
