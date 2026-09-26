# Transform annot filter into a dataset
import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv
from phoenix.client import Client
from phoenix.client.types.spans import SpanQuery

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tracing import collector_endpoint, project_name

PRESETS = {
    "frustrated-interactions": ("user_friction", "friction", "Turns where the user_friction judge labeled the user message as friction"),
    "hallucinated-answers": ("hallucination", "hallucination", "Turns where the hallucination judge found claims not supported by the tool results"),
    "tool-errors": ("tool_output_structured", "tool_error", "Tool spans whose output is an error payload"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="frustrated-interactions", help="dataset name, one of the presets or any name with --annotation and --label")
    parser.add_argument("--annotation", help="annotation name, e.g. hallucination")
    parser.add_argument("--label", help="label to keep, e.g. hallucination")
    parser.add_argument("--all-spans", action="store_true", help="do not restrict to root spans")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    annotation, label, description = PRESETS.get(args.name, (args.annotation, args.label, f"Spans where {args.annotation} == {args.label}"))
    if not annotation or not label:
        sys.exit("Give --annotation and --label, or use a preset name")
    root_only = "" if args.all_spans or annotation == "tool_output_structured" else "parent_id is None and "
    condition = f"{root_only}annotations['{annotation}'].label == '{label}'"
    client = Client(base_url=collector_endpoint())
    print(f"Querying Phoenix project {project_name()!r} with filter: {condition}")
    spans = client.spans.get_spans_dataframe(project_identifier=project_name(), query=SpanQuery().where(condition), limit=1000)
    if spans.empty:
        sys.exit("No matching spans. Run `make evals` first.")
    spans["context.span_id"] = spans.index
    spans["input"] = spans["attributes.input.value"].fillna("")
    spans["output"] = spans["attributes.output.value"].fillna("")
    spans["session_id"] = spans.get("attributes.session.id", "").fillna("")
    spans["trace_id"] = spans["context.trace_id"]
    print(f"Found {len(spans)} spans:")
    for _, row in spans.iterrows():
        print(f"  - [{row['session_id']}] {row['input'][:110]}")
    dataset = client.datasets.create_dataset(
        name=args.name,
        dataframe=spans[["context.span_id", "input", "output", "session_id", "trace_id"]].reset_index(drop=True),
        input_keys=["input"],
        output_keys=["output"],
        metadata_keys=["session_id", "trace_id"],
        span_id_key="context.span_id",
        dataset_description=description,
    )
    print(f"Dataset {args.name!r} ready with {len(spans)} examples: {collector_endpoint()}/datasets/{dataset.id}")

if __name__ == "__main__":
    main()
