import runpy,sys
sys.path.insert(0,'/home/choon/isaaclab/IsaacLab/scripts/reinforcement_learning')
from isaaclab.controllers import DifferentialIKController
original=DifferentialIKController.compute
def compute(self,ee_pos,ee_quat,jacobian,joint_pos):
    solved=original(self,ee_pos,ee_quat,jacobian,joint_pos)
    return joint_pos+0.25*(solved-joint_pos)
DifferentialIKController.compute=compute
sys.argv=['play_g1_reference.py','--checkpoint','logs/rsl_rl/g1_lipm_online_hands_finetune/run_cTgJrLdg/training/model_499.pt','--duration','20','--device','cuda:0','--output','/home/choon/isaaclab/LIP/previews/g1_heading_anchored_499_cTgJrLdg_quarter_ik_test']
runpy.run_module('play_g1_reference',run_name='__main__')
