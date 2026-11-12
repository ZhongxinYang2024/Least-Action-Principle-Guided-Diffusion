#%% standard 3rd lib
import tensorflow as tf
import logging
import os
from absl import app, flags
from ml_collections.config_flags import config_flags

## user def function
import run_lib
from models import ncsnpp_1d
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
#

#%%
FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    "config", None, "Training configuration.", lock_config=True
)
config_flags.DEFINE_config_file(
    "Hconfig", None, "eval configuration.", lock_config=True
)
flags.DEFINE_string("workdir", None, "Work directory.")
flags.DEFINE_enum("mode", None, ["train", "eval"], "Running mode: train or eval")
flags.DEFINE_string("eval_folder", "eval",
                    "The folder name for storing evaluation results")
flags.mark_flags_as_required(["workdir", "config", "mode"])
 
def main(argv):
    if FLAGS.mode == "train":
            
        # Create the working directory
        tf.io.gfile.makedirs(FLAGS.workdir)
        # Set logger so that it outputs to both console and file
        # Make logging work for both disk and Google Cloud Storage
        gfile_stream = open(os.path.join(FLAGS.workdir, 'stdout.txt'), 'w')
        handler = logging.StreamHandler(gfile_stream)
        formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
        handler.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler)
        logger.setLevel('INFO')
        # Run the training pipeline
        run_lib.train(FLAGS.config, FLAGS.workdir)
    elif FLAGS.mode == "eval":
        # Run the evaluation pipeline
        run_lib.evaluate(FLAGS.config, FLAGS.workdir, FLAGS.eval_folder, FLAGS.Hconfig)
    else:
        raise ValueError(f"Mode {FLAGS.mode} not recognized.")
    
#%%
if __name__   == '__main__':
    import sys
    Id_case = 'bubblev'  # freefall, bubblenv, bubblev, vortexsys, foil
    
    match Id_case.lower():
        case 'freefall':
            sys.argv = [
                "main.py",  # 程序名称
                f"--mode=eval",  # 运行模式：train 或 eval
                f"--workdir=./log/FreeFall_VE_Ncsnpp1d",  # 工作目录
                f"--config=./configs/FreeFall_ncsnpp1d_continous_ve.py",  # config file path
                f"--Hconfig=./HamiltonSys/Hconfigs/Q1_C1_config.py",
                f"--eval_folder=eval"      
            ]
        case 'bubblenv':
            sys.argv = [
                "main.py",  # 程序名称
                f"--mode=eval",  # 运行模式：train 或 eval
                f"--workdir=./log/BubbleNV_VE_Ncsnpp1d",  # 工作目录
                f"--config=./configs/BubbleNV_ncsnpp1d_continous_ve.py",  # config file path
                f"--Hconfig=./HamiltonSys/Hconfigs/Q2_C1_config.py",
                f"--eval_folder=eval"      
            ]
        case 'bubblev':
            sys.argv = [
                "main.py",  # 程序名称
                f"--mode=eval",  # 运行模式：train 或 eval
                f"--workdir=./log/BubbleV_VE_Ncsnpp1d",  # 工作目录
                f"--config=./configs/BubbleV_ncsnpp1d_continous_ve.py",  # config file path
                f"--Hconfig=./HamiltonSys/Hconfigs/Q3_C1_config.py",
                f"--eval_folder=eval"      
            ]
        case 'vortexsys':
            sys.argv = [
                "main.py",  # 程序名称
                f"--mode=eval",  # 运行模式：train 或 eval
                f"--workdir=./log/VortexSys_VE_Ncsnpp1d",  # 工作目录
                f"--config=./configs/VortexSys_ncsnpp1d_continous_ve.py",  # config file path
                f"--Hconfig=./HamiltonSys/Hconfigs/Q4_C3_config.py",
                f"--eval_folder=eval"      
            ]
        case 'foil':
            sys.argv = [
                "main.py",  # 程序名称
                f"--mode=eval",  # 运行模式：train 或 eval
                f"--workdir=./log/Foil_VE_Ncsnpp2d",  # 工作目录
                f"--config=./configs/Foil_ncsnpp2d_continous_ve.py",  # config file path
                f"--Hconfig=./HamiltonSys/Hconfigs/Q5_C3_config.py",
                f"--eval_folder=eval"      
            ]
        case _:
            raise ValueError(f'Id_case {Id_case} cannot be recognized')
    app.run(main)