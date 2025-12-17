#! python3
# -*- encoding: utf-8 -*-
'''
@File    :   config.py
@Time    :   2022/7/12 18:17
@Author  :   Songnan Lin, Ye Ma
@Note    :   Third-party code from DVS-Voltmeter
@inproceedings{lin2022dvsvoltmeter,
  title={DVS-Voltmeter: Stochastic Process-based Event Simulator for Dynamic Vision Sensors},
  author={Lin, Songnan and Ma, Ye and Guo, Zhenhua and Wen, Bihan},
  booktitle={ECCV},
  year={2022}
}
'''

from easydict import EasyDict as edict

__C     = edict()
cfg     = __C

# SENSOR
__C.SENSOR                              = edict()
__C.SENSOR.CAMERA_TYPE = 'DVS346'

__C.SENSOR.K = None
if cfg.SENSOR.CAMERA_TYPE == 'DVS346':
    __C.SENSOR.K = [6.517776598640289, 20, 0.0001, 1e-7, 5e-9, 1e-05]
elif cfg.SENSOR.CAMERA_TYPE == 'DVS240':
    __C.SENSOR.K = [6.517776598640289, 20, 0.0001, 1e-7, 5e-9, 1e-05]


# Directories
__C.DIR                                 = edict()
__C.DIR.IN_PATH = '/tmp/light_source_events_cnb_5ygj/'
__C.DIR.OUT_PATH = '/tmp/light_source_events_cnb_5ygj/'


# Visualize
__C.Visual                              = edict()
__C.Visual.FRAME_STEP = 5
