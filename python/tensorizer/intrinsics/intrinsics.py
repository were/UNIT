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
      'operands': [gpu.loader, gpu.loader, gpu.loader],
      'write': gpu.writeback,
      'init': gpu.initializer,
      'schedule': gpu.schedule,
      'cleanup': gpu.cleanup
  }
}
