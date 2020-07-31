""" Tensorization code generation """
import tvm

from .pattern import arm_sdot128_i8i16, x86_vnni
from . import cpu, gpu

INTRINSICS = {
  'vnni': {
    'pattern': x86_vnni,
    'operands': [cpu.x86_loader, cpu.x86_loader, cpu.x86_loader],
    'write': cpu.x86_writeback,
    'init': cpu.x86_init,
    'schedule': cpu.x86_schedule
  },
  'vdot': {
      'operands': [cpu.arm_acc, cpu.arm_operand, cpu.arm_operand],
      'write': cpu.arm_writeback,
      'init': cpu.arm_init,
      'schedule': cpu.arm_schedule
  },
  'tensorcore': {
      'operands': [gpu.operand_c_fp32_compute, gpu.operand_a_fp16x2, gpu.operand_b_fp16x2],
      'write': gpu.write_fp32,
      'init': gpu.initializer,
      'schedule': gpu.schedule,
      'cleanup': gpu.cleanup
  },
  'tensorcore.load_a': {
      'operands': [gpu.load_a],
      'write': gpu.store_ab,
  },
  'tensorcore.load_b': {
      'operands': [gpu.load_b],
      'write': gpu.store_ab,
  },
  'tensorcore.store_c': {
      'operands': [gpu.operand_c_fp32_store],
      'write': gpu.cleanup,
  }
}
