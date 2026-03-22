from src.models.packet_routing import PacketRoutingModel
from src.train.run import configure_es_parameter_names


def build_model() -> PacketRoutingModel:
    return PacketRoutingModel(
        {
            "num_nodes": 2,
            "obs_dim": 8,
            "hidden_dim": 16,
            "num_classes": 4,
            "max_internal_steps": 2,
            "max_total_steps": 16,
            "adapter_rank": 2,
            "packet_memory_slots": 2,
            "packet_memory_dim": 8,
            "control_state_dim": 2,
            "readout_mode": "query_conditioned",
        }
    )


def test_configure_es_parameter_names_defaults_to_router_family() -> None:
    model = build_model()
    info = configure_es_parameter_names(model, {"evolve_adapters": False})
    names = info["es_parameter_names"]
    assert names
    assert all(
        name.startswith(
            (
                "core.router_mlp",
                "core.router_out",
                "core.router_act_out",
                "core.router_wait_out",
                "control_update_mlp",
                "control_set_gate",
                "control_clear_gate",
                "control_router_out",
                "control_wait_out",
                "control_head",
                "wait_update_mlp",
                "wait_state_cell",
                "wait_head",
                "wait_input_proj",
                "release_head",
            )
        )
        for name in names
    )
    assert not any(name.startswith("readout") for name in names)


def test_configure_es_parameter_names_accepts_custom_head_only_filters() -> None:
    model = build_model()
    info = configure_es_parameter_names(
        model,
        {
            "trainable_prefixes": ["readout", "query_readout_proj"],
            "freeze_prefixes": ["core.", "control_", "wait_", "release_", "memory_", "sink_proj"],
        },
    )
    names = info["es_parameter_names"]
    assert names
    assert all(name.startswith(("readout", "query_readout_proj")) for name in names)
    assert not any(name.startswith("core.") for name in names)
