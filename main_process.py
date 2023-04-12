import yaml
import torch
import transformers
import pandas as pd
import pytorch_lightning as pl

from models.model import Model
from utils import data_preprocessing as dp
from utils import utils, train

if __name__ == "__main__":
    # --- Setting ---
    # config file
    with open('./baselines/baselines_config.yaml') as f:
        CFG = yaml.load(f, Loader=yaml.FullLoader)
    # 실험 결과 저장할 폴더 생성
    folder_name, save_path = utils.get_folder_name()
    # seed 설정
    utils.set_seed(CFG.seed)

    # --- Fit ---
    # load data and model
    tokenizer = transformers.AutoTokenizer.from_pretrained(CFG.train.model_name, CFG.train.max_len)
    plm = transformers.AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=CFG.train.model_name, num_labels=1)
    dataloader = train.Dataloader(tokenizer, CFG.train.batch_size, CFG.train.shuffle)
    model = Model(CFG, plm)

    # train and test
    trainer = pl.Trainer(accelerator='gpu', max_epochs=CFG.train.epoch, log_every_n_steps=1)
    trainer.fit(model, dataloader)
    trainer.test(model, dataloader)

    # inference
    predictions = trainer.predict(model, dataloader)
    pred_y = list(round(float(i), 1) for i in torch.cat(predictions))

    # --- save ---
    # write yaml
    with open(f'{save_path}/{folder_name}_config.yaml', 'w') as f:
        yaml.dump(CFG, f)
    torch.save(model, f'{save_path}/{folder_name}_model.pt') # save model
    # save submit
    submit = pd.read_csv('./data/sample_submission.csv')
    submit['target'] = pred_y
    submit.to_csv(f'{save_path}/{folder_name}_submit.csv', index=False)