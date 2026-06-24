def _as_bool(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return list(value)


def _get_lora_target_model(model, scope: str):
    if scope == "language_model":
        language_model = getattr(getattr(model, "model", None), "language_model", None)
        if language_model is None:
            raise ValueError(
                "LoRA scope `language_model` requires the VLM to expose "
                "`model.language_model`; this backbone does not."
            )
        return language_model, ("model", "language_model")
    if scope == "all":
        return model, ()
    raise ValueError(f"Unsupported qwenvl.lora_scope={scope!r}; expected `language_model` or `all`.")


def _set_nested_module(root, path, module):
    if not path:
        return module
    parent = root
    for attr in path[:-1]:
        parent = getattr(parent, attr)
    setattr(parent, path[-1], module)
    return root


def _maybe_apply_lora(vlm_interface, config):
    qwenvl_config = config.framework.get("qwenvl", {})
    if not _as_bool(qwenvl_config.get("use_lora", False)):
        return vlm_interface

    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as exc:
        raise ImportError("`framework.qwenvl.use_lora=true` requires the `peft` package.") from exc

    scope = qwenvl_config.get("lora_scope", "language_model")
    target_modules = _as_list(
        qwenvl_config.get("lora_target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])
    )
    if not target_modules:
        raise ValueError("`framework.qwenvl.lora_target_modules` must not be empty when LoRA is enabled.")

    target_model, target_path = _get_lora_target_model(vlm_interface.model, scope)
    freeze_base_model = _as_bool(qwenvl_config.get("lora_freeze_base_model", True))
    if freeze_base_model:
        for param in target_model.parameters():
            param.requires_grad = False

    lora_config = LoraConfig(
        r=int(qwenvl_config.get("lora_r", 16)),
        lora_alpha=int(qwenvl_config.get("lora_alpha", 32)),
        lora_dropout=float(qwenvl_config.get("lora_dropout", 0.05)),
        bias=qwenvl_config.get("lora_bias", "none"),
        target_modules=target_modules,
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    target_model = get_peft_model(target_model, lora_config)
    vlm_interface.model = _set_nested_module(vlm_interface.model, target_path, target_model)

    print(
        "[VLM LoRA] enabled "
        f"scope={scope}, r={lora_config.r}, alpha={lora_config.lora_alpha}, "
        f"targets={target_modules}, freeze_base_model={freeze_base_model}"
    )
    if hasattr(target_model, "print_trainable_parameters"):
        target_model.print_trainable_parameters()
    return vlm_interface


def get_vlm_model(config):

    vlm_name = config.framework.qwenvl.base_vlm

    if "Qwen2.5-VL" in vlm_name or "nora" in vlm_name.lower():  # temp for some ckpt
        from .QWen2_5 import _QWen_VL_Interface

        return _maybe_apply_lora(_QWen_VL_Interface(config), config)
    elif "Qwen3-VL" in vlm_name:
        from .QWen3 import _QWen3_VL_Interface

        return _maybe_apply_lora(_QWen3_VL_Interface(config), config)
    elif "Qwen3.5" in vlm_name:
        from .QWen3_5 import _QWen3_5_VL_Interface

        return _maybe_apply_lora(_QWen3_5_VL_Interface(config), config)
    elif "gemma-4" in vlm_name.lower() or "gemma4" in vlm_name.lower():
        from .Gemma4 import _Gemma4_VL_Interface

        return _maybe_apply_lora(_Gemma4_VL_Interface(config), config)
    elif "molmo2" in vlm_name.lower():
        from .Molmo2 import _Molmo2_VL_Interface

        return _maybe_apply_lora(_Molmo2_VL_Interface(config), config)
    elif "florence" in vlm_name.lower():  # temp for some ckpt
        from .Florence2 import _Florence_Interface

        return _maybe_apply_lora(_Florence_Interface(config), config)
    elif "cosmos-reason2" in vlm_name.lower():
        # Cosmos-Reason2 is architecturally Qwen3-VL (VLM), but implemented
        # in world_model/ for historical reasons. Import directly.
        from starVLA.model.modules.vlm.CosmosReason2 import _CosmosReason2_Interface

        return _maybe_apply_lora(_CosmosReason2_Interface(config), config)
    else:
        raise NotImplementedError(f"VLM model {vlm_name} not implemented")
