from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Optional
import copy


VALID_PRESET_KEYS: tuple[str, ...] = (
    "enable_conv",
    "target_module",
    "target_name",
    "module_algo_map",
    "name_algo_map",
    "lora_prefix",
    "use_fnmatch",
    "unet_target_module",
    "unet_target_name",
    "text_encoder_target_module",
    "text_encoder_target_name",
    "exclude_name",
)


@dataclass(frozen=True)
class AlgoSpec:
    """Metadata that describes which arguments an algorithm understands."""

    name: str
    description: str
    supported_args: tuple[str, ...] = ()
    required_args: tuple[str, ...] = ()
    notes: Optional[str] = None

    def supports(self, arg: str) -> bool:
        return arg in self.supported_args


ALGO_REGISTRY: Dict[str, AlgoSpec] = {
    "lora": AlgoSpec(
        name="lora",
        description="Standard LoRA / LoCon adapter.",
        supported_args=(
            "dim",
            "alpha",
            "dropout",
            "rank_dropout",
            "module_dropout",
            "use_tucker",
            "use_scalar",
            "weight_decompose",
            "wd_on_output",
            "bypass_mode",
        ),
    ),
    "locon": AlgoSpec(
        name="locon",
        description="Alias of lora that enables convolution layers by default.",
        supported_args=(
            "dim",
            "alpha",
            "dropout",
            "rank_dropout",
            "module_dropout",
            "use_tucker",
            "use_scalar",
            "weight_decompose",
            "wd_on_output",
            "bypass_mode",
        ),
    ),
    "loha": AlgoSpec(
        name="loha",
        description="LoHa adapter that factorizes with Hadamard products.",
        supported_args=(
            "dim",
            "alpha",
            "dropout",
            "rank_dropout",
            "module_dropout",
            "use_tucker",
            "use_scalar",
            "weight_decompose",
            "wd_on_output",
        ),
        notes="High dimensions may require lower learning rates to remain stable.",
    ),
    "lokr": AlgoSpec(
        name="lokr",
        description="Kronecker-product based adapter (LoKr).",
        supported_args=(
            "dim",
            "alpha",
            "factor",
            "dropout",
            "rank_dropout",
            "module_dropout",
            "use_scalar",
            "full_matrix",
            "weight_decompose",
            "wd_on_output",
            "unbalanced_factorization",
        ),
        notes="Setting dim to a very large value triggers the full-matrix path.",
    ),
    "dylora": AlgoSpec(
        name="dylora",
        description="Dynamic LoRA that incrementally updates rank blocks.",
        supported_args=(
            "dim",
            "alpha",
            "block_size",
            "dropout",
            "rank_dropout",
            "module_dropout",
        ),
    ),
    "glora": AlgoSpec(
        name="glora",
        description="Generalized LoRA adapter.",
        supported_args=(
            "dim",
            "alpha",
            "dropout",
            "rank_dropout",
            "module_dropout",
        ),
    ),
    "full": AlgoSpec(
        name="full",
        description="Native fine-tuning (full weight matrices).",
        supported_args=(
            "dim",
            "alpha",
            "dropout",
            "rank_dropout",
            "module_dropout",
        ),
        notes="Used for dreambooth-like full matrix training.",
    ),
    "diag-oft": AlgoSpec(
        name="diag-oft",
        description="Diagonal Orthogonal Finetuning.",
        supported_args=("dim", "constraint", "rescaled"),
    ),
    "boft": AlgoSpec(
        name="boft",
        description="Butterfly Orthogonal Finetuning.",
        supported_args=("dim", "constraint", "rescaled"),
    ),
    "ia3": AlgoSpec(
        name="ia3",
        description="Input-output scaling adapter (IA^3).",
        supported_args=(),
        notes="Most training setups use preset 'ia3' with dedicated module selection.",
    ),
    "tlora": AlgoSpec(
        name="tlora",
        description="Timestep-aware LoRA with SVD-orthogonal initialization.",
        supported_args=(
            "dim",
            "alpha",
            "dropout",
            "rank_dropout",
            "module_dropout",
            "bypass_mode",
            "sig_type",
            "use_data_init",
            "mask_group_id",
        ),
        notes=(
            "Uses SVD-based orthogonal initialization and supports timestep-dependent rank masking. "
            "Set sig_type to 'principal', 'last', or 'middle' to select which singular vectors to use. "
            "Training frameworks should call set_timestep_mask() before each forward pass."
        ),
    ),
}


class PresetValidationError(ValueError):
    pass


@dataclass
class AlgoOverride:
    """Per-module override describing which algorithm and kwargs to use."""

    algo: Optional[str] = None
    options: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "AlgoOverride":
        data = dict(mapping)
        algo = data.pop("algo", None)
        return cls(algo=algo, options=data)

    def to_dict(self) -> Dict[str, Any]:
        result = copy.deepcopy(self.options)
        if self.algo is not None:
            result["algo"] = self.algo
        return result

    def validate(self) -> None:
        if self.algo is None:
            return
        algo_name = self.algo.lower()
        if algo_name not in ALGO_REGISTRY:
            raise PresetValidationError(f"Unknown algorithm '{self.algo}'.")
        spec = ALGO_REGISTRY[algo_name]
        for key in self.options.keys():
            if key not in spec.supported_args:
                # Allow unknown options for forward compatibility but warn via exception details.
                raise PresetValidationError(
                    f"Unsupported option '{key}' for algo '{self.algo}'. "
                    f"Supported options: {spec.supported_args or 'None'}"
                )


def _copy_value(value: Any) -> Any:
    if value is None:
        return None
    return copy.deepcopy(value)


@dataclass
class PresetConfig:
    enable_conv: Optional[bool] = None
    target_module: Optional[list[str]] = None
    target_name: Optional[list[str]] = None
    module_algo_map: Dict[str, AlgoOverride] = field(default_factory=dict)
    name_algo_map: Dict[str, AlgoOverride] = field(default_factory=dict)
    lora_prefix: Optional[str] = None
    use_fnmatch: Optional[bool] = None
    unet_target_module: Optional[list[str]] = None
    unet_target_name: Optional[list[str]] = None
    text_encoder_target_module: Optional[list[str]] = None
    text_encoder_target_name: Optional[list[str]] = None
    exclude_name: Optional[list[str]] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(
        cls, data: Mapping[str, Any], *, strict: bool = False
    ) -> "PresetConfig":
        unknown_keys = [key for key in data.keys() if key not in VALID_PRESET_KEYS]
        if unknown_keys:
            raise PresetValidationError(
                f"Unknown preset keys: {', '.join(sorted(unknown_keys))}. "
                f"Valid keys: {', '.join(VALID_PRESET_KEYS)}"
            )

        kwargs: Dict[str, Any] = {}
        extra: Dict[str, Any] = {}
        for key in VALID_PRESET_KEYS:
            if key not in data:
                continue
            if key in ("module_algo_map", "name_algo_map"):
                overrides = {}
                raw_map = data.get(key) or {}
                for override_key, override_value in raw_map.items():
                    override = AlgoOverride.from_mapping(override_value or {})
                    if strict:
                        override.validate()
                    overrides[override_key] = override
                kwargs[key] = overrides
            else:
                kwargs[key] = _copy_value(data.get(key))
        preset = cls(**kwargs)
        preset.extra = extra
        return preset

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {}

        def maybe_set(key: str, value: Any) -> None:
            if value is None:
                return
            data[key] = _copy_value(value)

        maybe_set("enable_conv", self.enable_conv)
        maybe_set("target_module", self.target_module)
        maybe_set("target_name", self.target_name)
        if self.module_algo_map:
            data["module_algo_map"] = {
                key: override.to_dict()
                for key, override in self.module_algo_map.items()
            }
        if self.name_algo_map:
            data["name_algo_map"] = {
                key: override.to_dict() for key, override in self.name_algo_map.items()
            }
        maybe_set("lora_prefix", self.lora_prefix)
        maybe_set("use_fnmatch", self.use_fnmatch)
        maybe_set("unet_target_module", self.unet_target_module)
        maybe_set("unet_target_name", self.unet_target_name)
        maybe_set("text_encoder_target_module", self.text_encoder_target_module)
        maybe_set("text_encoder_target_name", self.text_encoder_target_name)
        maybe_set("exclude_name", self.exclude_name)

        data.update(_copy_value(self.extra))
        return data

    def list_algorithms(self) -> Iterable[str]:
        for override in self.module_algo_map.values():
            if override.algo:
                yield override.algo
        for override in self.name_algo_map.values():
            if override.algo:
                yield override.algo


def describe_algo(name: str) -> AlgoSpec:
    algo_name = name.lower()
    if algo_name not in ALGO_REGISTRY:
        raise PresetValidationError(f"Unknown algorithm '{name}'.")
    return ALGO_REGISTRY[algo_name]


def list_algorithms() -> Iterable[AlgoSpec]:
    return ALGO_REGISTRY.values()
