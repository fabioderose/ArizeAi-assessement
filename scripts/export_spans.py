# Export spans (step 4 in instructions: "Step 4: Tracing Experiments")
import json
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from phoenix.client import Client

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tracing import collector_endpoint, project_name

ROOT = Path(__file__).resolve().parent.parent
EXPORT_DIR = ROOT / "exports"


def is_missing(value) -> bool:
    if isinstance(value, (list, dict)):
        return False
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def main() -> None:
    client = Client(base_url=collector_endpoint())
    project = project_name()
    print(f"Exporting spans from project {project!r} at {collector_endpoint()}")
    spans = client.spans.get_spans_dataframe(project_identifier=project, limit=10000)
    if spans.empty:
        sys.exit("No spans found. Run `make queries` first.")
    EXPORT_DIR.mkdir(exist_ok=True)
    csv_path = EXPORT_DIR / "spans.csv"
    jsonl_path = EXPORT_DIR / "spans.jsonl"
    spans.to_csv(csv_path)
    with jsonl_path.open("w") as handle:
        for span_id, row in spans.iterrows():
            record = {"context.span_id": span_id, **{key: value for key, value in row.items() if not is_missing(value)}}
            handle.write(json.dumps(record, default=str, ensure_ascii=False) + "\n")
    traces = spans["context.trace_id"].nunique()
    print(f"\n{len(spans)} spans across {traces} traces")
    print("\nSpans by kind:")
    print(spans["span_kind"].value_counts().to_string())
    if "name" in spans.columns:
        tool_spans = spans[spans["span_kind"] == "TOOL"]
        if not tool_spans.empty:
            print("\nTool usage:")
            print(tool_spans["name"].value_counts().to_string())
    llm_spans = spans[spans["span_kind"] == "LLM"]
    if not llm_spans.empty and "attributes.llm.token_count.total" in llm_spans.columns:
        print(f"\nLLM calls: {len(llm_spans)}, total tokens: {int(llm_spans['attributes.llm.token_count.total'].fillna(0).sum())}")
    print(f"\nSaved {csv_path} and {jsonl_path}")

if __name__ == "__main__":
    main()
