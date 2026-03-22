from __future__ import annotations

from src.utils.phase8_verify import eval_config_label, run_name_to_config_basenames


def test_run_name_to_config_basenames_prefers_seeded_configs() -> None:
    basenames = run_name_to_config_basenames(
        "hard_st_b_v2_controlsticky_keepalive_teacher_medium_waitrelease_only_longrelease_selectroute_seed1826_p1"
    )

    assert basenames[0] == (
        "hard_st_benchmark_b_v2_controlsticky_keepalive_teacher_medium_waitrelease_only_longrelease_selectroute_seed1826.yaml"
    )
    assert basenames[1] == (
        "hard_st_benchmark_b_v2_controlsticky_keepalive_teacher_medium_waitrelease_only_longrelease_selectroute.yaml"
    )


def test_run_name_to_config_basenames_keeps_generic_fallback() -> None:
    basenames = run_name_to_config_basenames("hard_st_b_v2_controlsticky_keepalive_seed989_rerun1")

    assert basenames == [
        "hard_st_benchmark_b_v2_controlsticky_keepalive_seed989.yaml",
        "hard_st_benchmark_b_v2_controlsticky_keepalive.yaml",
    ]


def test_eval_config_label_uses_readable_confirm_names() -> None:
    assert (
        eval_config_label("configs/phase8/confirm/hard_st_benchmark_b_v2_confirm_full_locked.yaml", 0)
        == "full_locked"
    )
    assert (
        eval_config_label("configs/phase8/confirm/hard_st_benchmark_b_v2_confirm_finalqueryheavy.yaml", 1)
        == "finalquery_heavy"
    )
    assert (
        eval_config_label("configs/phase8/confirm/hard_st_benchmark_b_v2_confirm_longdistance.yaml", 2)
        == "longdistance"
    )
