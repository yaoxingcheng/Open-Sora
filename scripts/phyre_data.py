# This scripts should be run with a python<=3.7 environments that support phyre
# Part of the code borrowed from the examples of phyre git repo

import os
import random
import math
import multiprocessing
import imageio
import copy

import numpy as np
from tqdm import tqdm

import phyre

SEED = 42
NUM_WORKER = 64
SAMPLES_PER_TASK = 100
MAX_SAME_FRAME = 5
FPS = 20
OUTPUT_DIR = "/local2/xingcheng/data/phyre"

def sample_phyre_video_worker(task_ids, simulator, actions, eval_setup, splits):
    
    random.seed(SEED + int(os.getpid()))

    for task_index in tqdm(task_ids, desc=f"Proc ID: {os.getpid()}"):
        for sampled in range(SAMPLES_PER_TASK):
            task_name = simulator.task_ids[task_index]
            video_dir = os.path.join(OUTPUT_DIR, eval_setup, splits, task_name)
            os.makedirs(video_dir, exist_ok=True)

            sample_succ = False
            while not sample_succ:
                action = random.choice(actions)
                simulation = simulator.simulate_action(task_index, action, need_images=True, need_featurized_objects=True, stride=3)
                if simulation.status.is_invalid() or simulation.status == phyre.SimulationStatus.INVALID_INPUT:
                    sample_succ = False
                else:
                    sample_succ = True
            
            if simulation.status.is_solved():
                video_name = str(sampled) + "-s.mp4"
            else:
                video_name = str(sampled) + "-f.mp4"
            
            video_file = os.path.join(video_dir, video_name)
            
            prev_image:np.array = copy.deepcopy(simulation.images[0])
            frames:list = [phyre.observations_to_uint8_rgb(simulation.images[0])]
            same_frame = 0

            for i, image in enumerate(simulation.images[1:]):
                if np.absolute(image - prev_image).sum() < 0.1:
                    same_frame += 1
                    if same_frame > MAX_SAME_FRAME:
                        break
                else:
                    same_frame = 0
                prev_image = copy.deepcopy(image)
                frames.append(phyre.observations_to_uint8_rgb(image))
            frames = np.stack(frames, axis=0)
            imageio.mimwrite(video_file, frames, fps=FPS)


def sample_phyre_data():

    random.seed(SEED)

    for eval_setup in phyre.MAIN_EVAL_SETUPS:

        print(f"Sampling videos with eval setup {eval_setup}")
        train_tasks, dev_tasks, test_tasks = phyre.get_fold(eval_setup, SEED)

        print(
            'Size of resulting splits:\n train:', len(train_tasks), '\n dev:',
      len(dev_tasks), '\n test:', len(test_tasks)
        )

        print("Initializing simulator...")
        
        action_tier = phyre.eval_setup_to_action_tier(eval_setup)

        # Create the simulator from the tasks and tier.
        simulator = phyre.initialize_simulator(train_tasks, action_tier)
        actions = simulator.build_discrete_action_space(max_actions=SAMPLES_PER_TASK * 10)

        print("Initializing simulator finished.")

        task_ids = list(range(len(train_tasks)))
        num_tasks_per_proc = math.ceil(len(train_tasks) / NUM_WORKER)
        jobs = [
            (task_ids[ids:ids+num_tasks_per_proc], simulator, actions, eval_setup, "train") for ids in range(0, len(train_tasks), num_tasks_per_proc)
        ]
        with multiprocessing.Pool(processes=NUM_WORKER) as pool:
            # Use pool.starmap to pass multiple arguments to worker_process
            results = pool.starmap(sample_phyre_video_worker, jobs)
        
        print(f"Results: {results}")

if __name__=="__main__":
    sample_phyre_data()