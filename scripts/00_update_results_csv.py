import hashlib
import io
import json
import os
import tempfile
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = PROJECT_ROOT / "data/raw/results.csv"
STATUS_PATH = PROJECT_ROOT / "data/processed/update_status.json"
RESULTS_URL = os.environ.get(
    "RESULTS_CSV_URL",
    "https://raw.githubusercontent.com/martj42/"
    "international_results/refs/heads/master/results.csv",
)


def calculate_hash(content):
    return hashlib.sha256(content).hexdigest()


def read_csv_metadata(content):
    dataframe = pd.read_csv(io.BytesIO(content))

    if "date" not in dataframe.columns:
        raise ValueError("다운로드한 CSV에 date 컬럼이 없습니다.")

    dates = pd.to_datetime(dataframe["date"], errors="coerce")
    latest_date = dates.max()

    if pd.isna(latest_date):
        raise ValueError("다운로드한 CSV의 date 컬럼에 유효한 날짜가 없습니다.")

    return {
        "hash": calculate_hash(content),
        "row_count": int(len(dataframe)),
        "latest_date": latest_date.date().isoformat(),
    }


def download_results_csv():
    request = Request(
        RESULTS_URL,
        headers={"User-Agent": "worldcup-predictor-daily-pipeline"},
    )

    with urlopen(request, timeout=60) as response:
        return response.read()


def replace_results_csv(content):
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=RESULTS_PATH.parent,
        prefix="results-",
        suffix=".csv.tmp",
        delete=False,
    ) as temp_file:
        temp_path = Path(temp_file.name)
        temp_file.write(content)

    temp_path.replace(RESULTS_PATH)


def write_status(status):
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)

    with STATUS_PATH.open("w", encoding="utf-8") as file:
        json.dump(status, file, ensure_ascii=False, indent=2)
        file.write("\n")


def main():
    remote_content = download_results_csv()
    remote = read_csv_metadata(remote_content)

    local = None
    if RESULTS_PATH.exists():
        local = read_csv_metadata(RESULTS_PATH.read_bytes())

    changed_fields = [
        field
        for field in ("hash", "row_count", "latest_date")
        if local is None or local[field] != remote[field]
    ]
    changed = bool(changed_fields)

    if changed:
        replace_results_csv(remote_content)

    status = {
        "changed": changed,
        "changed_fields": changed_fields,
        "source_url": RESULTS_URL,
        "row_count": remote["row_count"],
        "latest_date": remote["latest_date"],
        "hash": remote["hash"],
        "previous": local,
    }
    write_status(status)

    if changed:
        print(f"results.csv 업데이트 완료: {RESULTS_PATH}")
        print(f"변경 기준: {', '.join(changed_fields)}")
    else:
        print("results.csv 변경 없음")

    print(f"행 수: {remote['row_count']}")
    print(f"최신 날짜: {remote['latest_date']}")
    print(f"상태 저장: {STATUS_PATH}")


if __name__ == "__main__":
    main()
