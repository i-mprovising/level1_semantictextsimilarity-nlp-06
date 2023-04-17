import os
import random
import numpy as np
import pandas as pd
import torch.backends.cudnn as cudnn

from datetime import datetime
from datetime import timezone
from datetime import timedelta
from glob import glob


def get_folder_name(CFG):
    now = datetime.now(tz=timezone(timedelta(hours=9)))
    folder_name = now.strftime('%Y-%m-%d-%H:%M:%S') + f"_{CFG['admin']}"
    save_path = f"./results/{folder_name}"
    CFG['save_path'] = save_path
    os.makedirs(save_path)

    return folder_name, save_path


def get_data():
    train_df = pd.read_csv('./data/train.csv')
    val_df = pd.read_csv('./data/dev.csv')
    test_df = pd.read_csv('./data/test.csv')

    return train_df, val_df, test_df


def get_best_check_point(save_path):
    """
    가장 최근 체크포인트로 학습된 모델을 가져오는 메소드
    """
    check_point_list = glob(f'results/{save_path}/*/*/*/epoch*')
    check_point_list.sort(reverse=True)
    
    last_check_point = check_point_list[0]
    
    return last_check_point


if __name__ == "__main__":
    get_best_check_point('2023-04-17-14:13:02_KGB')