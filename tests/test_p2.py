from __future__ import annotations

from pathlib import Path

import httpx
import pytest
import torch
import yaml

from llm_bench.llm import LocalLLM
from llm_bench.p2.backends import create_backend
from llm_bench.p2.comparison import BackendRun, compare_same_workload
from llm_bench.p2.metrics import (
    parse_prometheus_text,
    summarize_serving_metrics,
    summarize_sglang_serving_metrics,
)
from llm_bench.p2.sglang import SGLangServer, SGLangServerConfig
from llm_bench.p2.triton_ref import rmsnorm_reference, vector_add_reference
from llm_bench.p2.vllm import VLLMServer, VLLMServerConfig


def test_vllm_command_exposes_required_flags() -> None:
    config = VLLMServerConfig(
        model="Qwen/Qwen3-1.7B",
        max_model_len=2048,
        max_num_seqs=4,
        max_num_batched_tokens=4096,
        gpu_memory_utilization=0.8,
        enable_prefix_caching=True,
        enable_cuda_graph=True,
    )
    command = config.command()
    for flag, value in (
        ("--max-model-len", "2048"),
        ("--max-num-seqs", "4"),
        ("--max-num-batched-tokens", "4096"),
        ("--gpu-memory-utilization", "0.8"),
        ("--kv-cache-dtype", "auto"),
    ):
        assert command[command.index(flag) + 1] == value
    assert "--enable-prefix-caching" in command
    assert "--enforce-eager" not in command


def test_vllm_rejects_unsupported_continuous_batching_disable() -> None:
    with pytest.raises(ValueError, match="continuous batching"):
        VLLMServerConfig(model="m", continuous_batching=False).validate()


def test_vllm_prefix_cache_disable_uses_explicit_flag() -> None:
    command = VLLMServerConfig(
        model="m",
        enable_prefix_caching=False,
    ).command()
    assert "--no-enable-prefix-caching" in command


def test_prometheus_parser_and_prefix_cache_rate() -> None:
    text = """
# HELP vllm
vllm:num_requests_running 3
vllm:num_requests_waiting{model="q"} 2
vllm:gpu_cache_usage_perc 0.42
vllm:prefix_cache_queries_total 100
vllm:prefix_cache_hits_total 75
process_resident_memory_bytes 1.0
"""
    samples = parse_prometheus_text(text)
    assert len(samples) == 6
    summary = summarize_serving_metrics(text)
    assert summary["num_requests_running"] == 3
    assert summary["num_requests_waiting"] == 2
    assert summary["kv_cache_usage_perc"] == 0.42
    assert summary["prefix_cache_hit_rate"] == 0.75
    assert "vllm:gpu_cache_usage_perc" in summary["raw_metric_names"]


def test_missing_metric_fields_are_explicitly_none() -> None:
    summary = summarize_serving_metrics("process_resident_memory_bytes 1\n")
    assert summary["num_requests_running"] is None
    assert summary["kv_cache_usage_perc"] is None
    assert summary["prefix_cache_hit_rate"] is None
    assert summary["ttft_p95_s"] is None


def test_histogram_metrics_are_computed_from_prometheus_buckets() -> None:
    text = """
vllm:time_to_first_token_seconds_bucket{le="0.1"} 50
vllm:time_to_first_token_seconds_bucket{le="0.2"} 95
vllm:time_to_first_token_seconds_bucket{le="+Inf"} 100
vllm:time_to_first_token_seconds_count 100
vllm:request_time_per_output_token_seconds_sum 2.0
vllm:request_time_per_output_token_seconds_count 100
vllm:e2e_request_latency_seconds_bucket{le="1"} 90
vllm:e2e_request_latency_seconds_bucket{le="2"} 100
vllm:e2e_request_latency_seconds_bucket{le="+Inf"} 100
"""
    summary = summarize_serving_metrics(text)
    assert summary["ttft_p50_s"] == pytest.approx(0.1)
    assert summary["ttft_p95_s"] == pytest.approx(0.2)
    assert summary["tpot_mean_s"] == pytest.approx(0.02)
    assert summary["e2e_latency_p95_s"] == pytest.approx(1.5)


def test_same_hardware_comparison_produces_ratios() -> None:
    result = compare_same_workload(
        [
            BackendRun("llama.cpp", "gpu-4090d", "batch", {"ttft_p95_s": 2.0}),
            BackendRun("vllm", "gpu-4090d", "batch", {"ttft_p95_s": 1.0}),
        ]
    )
    assert result["same_hardware"] is True
    assert result["ratios"]["vllm_vs_llama.cpp"]["ttft_p95_s"] == 0.5


def test_cross_hardware_comparison_keeps_results_separate() -> None:
    result = compare_same_workload(
        [
            BackendRun("llama.cpp", "m4", "batch", {"ttft_p95_s": 2.0}),
            BackendRun("vllm", "gpu-4090d", "batch", {"ttft_p95_s": 1.0}),
        ]
    )
    assert result["same_hardware"] is False
    assert result["cross_hardware_ratio_available"] is False
    assert result["ratios"] == {}


def test_triton_reference_matches_cpu_addition() -> None:
    left = torch.tensor([1.0, 2.0, 3.0])
    right = torch.tensor([4.0, 5.0, 6.0])
    torch.testing.assert_close(vector_add_reference(left, right), torch.tensor([5.0, 7.0, 9.0]))


def test_rmsnorm_reference_has_expected_normalization() -> None:
    values = torch.tensor([[3.0, 4.0], [1.0, 1.0]])
    weight = torch.ones(2)
    result = rmsnorm_reference(values, weight)
    expected = torch.tensor(
        [
            [3.0 / ((12.5 + 1e-6) ** 0.5), 4.0 / ((12.5 + 1e-6) ** 0.5)],
            [1.0 / ((1.0 + 1e-6) ** 0.5), 1.0 / ((1.0 + 1e-6) ** 0.5)],
        ]
    )
    torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize(
    ("dtype", "tolerance"),
    [
        (torch.float32, 1e-4),
        (torch.float16, 2e-2),
        (torch.bfloat16, 2e-2),
    ],
)
def test_rmsnorm_reference_supports_expected_dtypes(
    dtype: torch.dtype,
    tolerance: float,
) -> None:
    values = torch.ones((2, 8), dtype=dtype)
    result = rmsnorm_reference(values, torch.ones(8, dtype=dtype))
    torch.testing.assert_close(
        result,
        torch.ones_like(result),
        atol=tolerance,
        rtol=tolerance,
    )


def test_sglang_command_exposes_radix_and_runtime_flags() -> None:
    config = SGLangServerConfig(
        model_path="Qwen/Qwen3-1.7B",
        enable_radix_cache=True,
        enable_cache_report=True,
    )
    command = config.command()
    assert "--enable-cache-report" in command
    assert "--disable-radix-cache" not in command
    assert command[command.index("--mem-fraction-static") + 1] == "0.85"
    disabled = SGLangServerConfig(
        model_path="Qwen/Qwen3-1.7B",
        enable_radix_cache=False,
    ).command()
    assert "--disable-radix-cache" in disabled


def test_sglang_metrics_parse_cache_and_latency() -> None:
    text = """
sglang:num_running_reqs 3
sglang:num_queue_reqs 2
sglang:token_usage 0.4
sglang:cache_hit_rate 0.75
sglang:prompt_tokens_total 100
sglang:generation_tokens_total 40
sglang:time_to_first_token_seconds_bucket{le="0.1"} 50
sglang:time_to_first_token_seconds_bucket{le="+Inf"} 100
"""
    summary = summarize_sglang_serving_metrics(text)
    assert summary["num_running_reqs"] == 3
    assert summary["num_queue_reqs"] == 2
    assert summary["cache_hit_rate"] == 0.75
    assert summary["prompt_tokens_total"] == 100


def test_backend_factory_supports_all_declared_kinds(tmp_path: Path) -> None:
    llama = create_backend(
        "llamacpp",
        {"model_path": str(tmp_path / "model.gguf")},
        tmp_path / "llama.log",
    )
    vllm = create_backend(
        "vllm",
        {"model": "Qwen/Qwen3-1.7B"},
        tmp_path / "vllm.log",
    )
    sglang = create_backend(
        "sglang",
        {"model_path": "Qwen/Qwen3-1.7B"},
        tmp_path / "sglang.log",
    )
    assert type(llama).__name__ == "LlamaServer"
    assert isinstance(vllm, VLLMServer)
    assert isinstance(sglang, SGLangServer)


def test_vllm_health_success_and_metrics_timeout(monkeypatch, tmp_path: Path) -> None:
    class FakeProcess:
        def poll(self):
            return None

    class HealthyResponse:
        status_code = 200
        text = "vllm:num_requests_running 1\n"

        def raise_for_status(self):
            return None

    monkeypatch.setattr("llm_bench.p2.vllm.subprocess.Popen", lambda *_, **__: FakeProcess())
    monkeypatch.setattr("llm_bench.p2.vllm.httpx.get", lambda *_, **__: HealthyResponse())
    server = VLLMServer(VLLMServerConfig(model="fake"), tmp_path / "vllm.log")
    server.start(timeout_s=0.01)
    assert server.cold_start_s is not None
    assert server.metrics_text().startswith("vllm:")

    def timeout(*_, **__):
        raise httpx.TimeoutException("timed out")

    monkeypatch.setattr("llm_bench.p2.vllm.httpx.get", timeout)
    with pytest.raises(httpx.TimeoutException):
        server.metrics_text()


def test_vllm_health_timeout_is_explicit(monkeypatch, tmp_path: Path) -> None:
    class FakeProcess:
        def poll(self):
            return None

    monkeypatch.setattr("llm_bench.p2.vllm.subprocess.Popen", lambda *_, **__: FakeProcess())
    monkeypatch.setattr(
        "llm_bench.p2.vllm.httpx.get",
        lambda *_, **__: (_ for _ in ()).throw(httpx.ConnectError("down")),
    )
    server = VLLMServer(VLLMServerConfig(model="fake"), tmp_path / "vllm.log")
    monkeypatch.setattr(server, "stop", lambda: None)
    with pytest.raises(TimeoutError, match="did not become ready"):
        server.start(timeout_s=0.01)


def test_openai_request_contract_uses_local_endpoint() -> None:
    client = LocalLLM("http://127.0.0.1:8000/v1", "Qwen/Qwen3-1.7B")
    payload = client.build_payload(
        system_prompt="system",
        user_prompt="question",
        max_tokens=8,
        seed=42,
        disable_cache=False,
        cache_prompt=True,
    )
    assert client.base_url == "http://127.0.0.1:8000/v1"
    assert payload["model"] == "Qwen/Qwen3-1.7B"
    assert payload["cache_prompt"] is True


def test_gpu_configs_are_parseable_and_declare_expected_backend() -> None:
    backend = yaml.safe_load(
        Path("configs/vllm_backends.yaml").read_text(encoding="utf-8")
    )
    assert "qwen3_1_7b" in backend["vllm"]
    assert backend["hardware_profile"]["expected_gpu"] == "NVIDIA GeForce RTX 4090 D"
    compose = yaml.safe_load(
        Path("docker/vllm/compose.yaml").read_text(encoding="utf-8")
    )
    assert compose["services"]["vllm"]["ipc"] == "host"
    assert compose["services"]["vllm"]["deploy"]["resources"]["reservations"]["devices"][0][
        "capabilities"
    ] == ["gpu"]
    sglang = yaml.safe_load(
        Path("configs/sglang_backends.yaml").read_text(encoding="utf-8")
    )
    assert sglang["sglang"]["qwen3_1_7b_radix"]["enable_radix_cache"] is True
    assert sglang["sglang"]["qwen3_1_7b_no_radix"]["enable_radix_cache"] is False
