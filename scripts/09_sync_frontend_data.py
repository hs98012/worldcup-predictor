import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = PROJECT_ROOT / "data/processed"
FRONTEND_DATA_DIR = PROJECT_ROOT / "frontend/public/data"
SYNC_FILES = [
    "dashboard_summary.json",
    "tournament_simulation.json",
    "fixture_predictions.json",
    "model_metrics.json",
    "draw_calibration.json",
    "group_simulation.json",
    "worldcup_completed_results.json",
    "worldcup_group_standings.json",
]


def main():
    missing_files = [
        filename
        for filename in SYNC_FILES
        if not (SOURCE_DIR / filename).is_file()
    ]

    if missing_files:
        missing_list = ", ".join(missing_files)
        raise FileNotFoundError(
            f"프론트엔드 동기화 원본 파일이 없습니다: {missing_list}"
        )

    FRONTEND_DATA_DIR.mkdir(parents=True, exist_ok=True)

    for filename in SYNC_FILES:
        source_path = SOURCE_DIR / filename
        destination_path = FRONTEND_DATA_DIR / filename
        shutil.copy2(source_path, destination_path)
        print(f"프론트엔드 데이터 복사 완료: {filename}")

    print(f"동기화 경로: {FRONTEND_DATA_DIR.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
