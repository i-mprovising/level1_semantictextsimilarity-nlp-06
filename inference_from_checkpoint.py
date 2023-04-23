import yaml
import torch
import transformers
import pandas as pd
import pytorch_lightning as pl

from models.model import Model
from utils import utils, train

import warnings
warnings.filterwarnings('ignore')

if __name__ == "__main__":
    # --- Setting ---
    # config file
    with open('baselines/baseline_config.yaml') as f:
        CFG = yaml.load(f, Loader=yaml.FullLoader)
    # 실험 결과 저장할 폴더 생성
    folder_name, save_path = utils.get_folder_name(CFG)
    # seed 설정
    pl.seed_everything(CFG['seed'])

    # # --- Fit ---
    # # load data and model
    tokenizer = transformers.AutoTokenizer.from_pretrained(CFG['train']['model_name'], max_length=CFG['train']['max_len'])
    dataloader = train.Dataloader(tokenizer, CFG)

    # train and test
    torch.cuda.is_available()
    trainer = pl.Trainer(accelerator='gpu',
                         max_epochs=CFG['train']['epoch'],
                         default_root_dir=save_path)

    # inference
    # checkpoint의 경로를 붙여넣어주세요.
    model = Model.load_from_checkpoint('results/2023-04-15-23:09:58_KGB/lightning_logs/o2x4gru7/checkpoints/epoch=20-val_loss=0.07.ckpt')
    predictions = trainer.predict(model=model, datamodule=dataloader)
    pred_y = list(float(i) for i in torch.cat(predictions))

    # --- save ---
    # write yaml
    with open(f'{save_path}/{folder_name}_config.yaml', 'w') as f:
        yaml.dump(CFG, f)
    # save mode
    torch.save(model, f'{save_path}/{folder_name}_model.pt')
    # save submit
    submit = pd.read_csv('./data/sample_submission.csv')
    submit['target'] = pred_y
    submit.to_csv(f'{save_path}/{folder_name}_submit.csv', index=False)