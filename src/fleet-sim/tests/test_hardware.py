"""Unit tests for fleet_sim.hardware — HardwareSpec and catalog."""

import pytest

from fleet_sim.hardware import (
    A100_SXM,
    B200_SXM,
    GB200,
    GB300,
    H100_SXM,
    H200_SXM,
    HardwareSpec,
    get_hardware,
    list_hardware,
)
from fleet_sim.hardware.spec import MEM_BW_SCALE, NCCL_MEM


class TestHardwareSpec:
    def test_effective_mem_bw_is_scaled(self):
        assert H100_SXM.effective_mem_bw == pytest.approx(
            H100_SXM.mem_bw * MEM_BW_SCALE
        )

    def test_effective_mem_bw_h100(self):
        # H100: 3.35 TB/s × 0.80 ≈ 2.68 TB/s
        assert H100_SXM.effective_mem_bw == pytest.approx(2.68e12, rel=0.01)

    def test_free_vram_subtracts_all_overheads(self):
        # With 10 GB model on H100 (TP=1), nccl=0, other=3.5 GB
        model_bytes = 10 * 1024**3
        free = H100_SXM.free_vram(model_bytes, tp=1)
        expected = H100_SXM.mem_capacity - model_bytes - 0 - H100_SXM.other_mem
        assert free == expected

    def test_free_vram_tp2_subtracts_nccl(self):
        model_bytes = 10 * 1024**3
        free_tp1 = H100_SXM.free_vram(model_bytes, tp=1)
        free_tp2 = H100_SXM.free_vram(model_bytes, tp=2)
        assert free_tp2 < free_tp1
        # Difference should equal NCCL_MEM[2]
        assert free_tp1 - free_tp2 == NCCL_MEM[2]

    def test_free_vram_never_negative(self):
        huge_model = H100_SXM.mem_capacity * 10
        assert H100_SXM.free_vram(huge_model, tp=1) == 0

    def test_h200_has_more_vram_than_h100(self):
        assert H200_SXM.mem_capacity > H100_SXM.mem_capacity

    def test_h200_faster_mem_bw_than_h100(self):
        assert H200_SXM.mem_bw > H100_SXM.mem_bw

    def test_blackwell_highest_mem_bw(self):
        assert B200_SXM.mem_bw == GB200.mem_bw == GB300.mem_bw
        assert B200_SXM.mem_bw > H200_SXM.mem_bw

    def test_fp8_supported_on_hopper_blackwell(self):
        assert H100_SXM.fp8_tc_flops > 0
        assert H200_SXM.fp8_tc_flops > 0
        assert B200_SXM.fp8_tc_flops > 0

    def test_fp8_not_on_a100(self):
        assert A100_SXM.fp8_tc_flops == 0

    def test_cost_per_hr_ordering(self):
        # Rough cost ordering: A100 < H100 < H200 < B200 < GB200
        assert A100_SXM.cost_per_hr < H100_SXM.cost_per_hr
        assert H100_SXM.cost_per_hr < H200_SXM.cost_per_hr
        assert H200_SXM.cost_per_hr < B200_SXM.cost_per_hr

    def test_frozen_immutable(self):
        with pytest.raises((AttributeError, TypeError)):
            H100_SXM.mem_bw = 999


class TestHardwareCatalog:
    def test_list_hardware_returns_eight_gpus(self):
        names = list_hardware()
        assert len(names) == 8

    def test_get_by_canonical_name(self):
        assert get_hardware("h100") is H100_SXM
        assert get_hardware("a100") is A100_SXM
        assert get_hardware("h200") is H200_SXM

    def test_get_case_insensitive(self):
        assert get_hardware("H100") is H100_SXM
        assert get_hardware("H100_SXM") is H100_SXM
        assert get_hardware("h100-sxm") is H100_SXM

    def test_get_unknown_raises_keyerror(self):
        with pytest.raises(KeyError):
            get_hardware("v100")

    def test_all_catalog_names_resolvable(self):
        for name in list_hardware():
            hw = get_hardware(name)
            assert isinstance(hw, HardwareSpec)
