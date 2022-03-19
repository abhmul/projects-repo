import os
from pytz import timezone
from datetime import datetime

import tensorflow as tf

from projectslib.file_utils import safe_open_dir

from experiments import EXPERIMENTS

class Logger:
    
    def __init__(self) -> None:
        pass
    
    def log(self, msg: str) -> str:
        return msg

class PrintLogger(Logger):
    
    def log(self, msg: str) -> str:
        print(msg)
        return msg

class Experimenter:
    
    def __init__(
        self,
        tzname='America/Chicago', 
        write_to_tensorboard=False, 
        tensorboard_dir='./tensorboard/experiments/',
    ) -> None:
        self.tz = timezone(tzname)
        self.timestamp = datetime.now().astimezone(self.tz).strftime('%Y-%m-%d.%H-%M-%S')
        # Create tensorboard writer
        self.write_to_tensorboard = write_to_tensorboard
        self.tensorboard_dir = tensorboard_dir
    
    def make_experiment_id(self, experiment_num):
        return '.'.join([f'exp{experiment_num}', self.timestamp])
    
    def run_experiment(self, experiment_num, rng):
        experiment_id = self.make_experiment_id(experiment_num)
        summary_writer = tf.summary.create_file_writer(os.path.join(safe_open_dir(self.tensorboard_dir), experiment_id)) if self.write_to_tensorboard else None
        logger = PrintLogger()
        experiment_function = EXPERIMENTS[experiment_num]
        experiment_function(rng, logger, summary_writer)