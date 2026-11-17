from __future__ import annotations

import copy
import json
from typing import Any

from resume_bench.providers.base import (
    ExtractionRequest,
    Provider,
    ProviderConfigError,
    ProviderUsage,
)
from resume_bench.providers.registry import register_provider


def _strip_object_fields(schema_node: dict) -> None:
    """Recursively remove properties with type ["object", "null"] from schema."""
    if not isinstance(schema_node, dict):
        return
    props = schema_node.get("properties", {})
    to_remove = []
    for key, val in props.items():
        if isinstance(val, dict):
            t = val.get("type")
            if isinstance(t, list) and "object" in t:
                to_remove.append(key)
            else:
                _strip_object_fields(val)
    for key in to_remove:
        del props[key]
    if "items" in schema_node:
        _strip_object_fields(schema_node["items"])


@register_provider("extend")
class ExtendProvider(Provider):
    def healthcheck(self) -> None:
        from resume_bench.settings import settings

        if not settings.extend_api_key:
            raise ProviderConfigError("RESUME_BENCH_EXTEND_API_KEY not set")

    def extract(self, req: ExtractionRequest) -> dict[str, Any]:
        from extend_ai import Extend

        from resume_bench.settings import settings

        client = Extend(token=settings.extend_api_key)

        with open(req.pdf_path, "rb") as f:
            file_obj = client.files.upload(file=f)

        extend_schema = copy.deepcopy(req.extraction_schema)
        extend_schema.get("properties", {}).pop("layout", None)
        _strip_object_fields(extend_schema)

        processor = self.spec.config.get("processor", "extraction_light")

        run = client.extract_runs.create_and_poll(
            file={"id": file_obj.id},
            config={
                "schema": extend_schema,
                "base_processor": processor,
                "extractionRules": req.system_prompt,
            },
        )

        run_id = getattr(run, "id", None)
        run_status = getattr(run, "status", None)
        credits_used = None
        pages_extracted = None
        if hasattr(run, "usage") and run.usage:
            pages_extracted = getattr(run.usage, "pages", None)
            credits_used = getattr(run.usage, "credits", None)

        data: Any = {}
        if hasattr(run, "output") and run.output:
            data = run.output
            if isinstance(data, list) and data:
                data = data[0]
            if hasattr(data, "model_dump"):
                data = data.model_dump()
            elif not isinstance(data, dict):
                data = json.loads(str(data))
            if isinstance(data, dict) and "value" in data and isinstance(data["value"], dict):
                data = data["value"]

        return {
            "parsed": data,
            "run_id": str(run_id) if run_id else None,
            "run_status": str(run_status) if run_status else None,
            "credits_used": credits_used,
            "pages_extracted": pages_extracted,
            "processor": processor,
        }

    def to_canonical(self, raw: dict[str, Any]) -> dict[str, Any]:
        return raw.get("parsed", raw)

    def get_usage(self, raw: dict[str, Any]) -> ProviderUsage | None:
        return ProviderUsage(
            credits_used=raw.get("credits_used"),
            pages_extracted=raw.get("pages_extracted"),
            request_id=raw.get("run_id"),
        )
