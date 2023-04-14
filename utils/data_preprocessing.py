import pandas as pd

from transformers import pipeline
from googletrans import Translator
from tqdm.auto import tqdm

def remove_special_word(df):
    df['sentence_1'] = df['sentence_1'].str.replace(pat=r'[^\w]', repl=r' ', regex=True)
    df['sentence_2'] = df['sentence_2'].str.replace(pat=r'[^\w]', repl=r' ', regex=True)

    return df

def text_style_transfer(df):
    """
    text style을 두 가지로 바꾸는 augmentation. 
    
    필요 라이브러리 :
    pip install transformers
    
    Args:
        df (pd.DataFrame): 원본 train data
    Returns:
        pd.DataFrame    
    """
    model = pipeline(
    'text2text-generation',
    model='heegyu/kobart-text-style-transfer'
    )
    styles = ['문어체','구어체']
    
    #스타일이 바뀐 문장을 반환하는 함수.
    def get_transferred_text(text, target_style, **kwargs):
      input = f"{target_style} 말투로 변환:{text}"
      out = model(input, max_length=64, **kwargs)
      return out[0]['generated_text']
    
    augmented_df = pd.DataFrame(columns=df.columns)
    
    for style in styles:
        tmp = df.copy()
        sentence_1 = []
        sentence_2 = []
        for i in tqdm(range(len(df)), desc=style):
            item = df.iloc[i]
            sentence_1.append(get_transferred_text(item['sentence_1'], style))
            sentence_2.append(get_transferred_text(item['sentence_2'], style))
        tmp['sentence_1'] = sentence_1
        tmp['sentence_2'] = sentence_2
        augmented_df = pd.concat([augmented_df, tmp])
        
    return augmented_df

def back_translation(df):
    """
    각 sentence를 영어로 번역 후 다시 한글로 번역한 데이터를 사용하는 augmentation.
    
    필요 라이브러리 :
    pip install googletrans==4.0.0-rc1
    
    Args:
        df (pd.DataFrame): 원본 train data
    Returns:
        pd.DataFrame    
    """
    #한글 문장을 받아 영어로 번역하고 다시 한글로 번역한 문장을 반환하는 함수.
    def back_translate(sentence): 
        translator = Translator()
        en_translate = translator.translate(sentence, src="ko", dest="en")
        ko_translate = translator.translate(en_translate.text, src="en", dest="ko")
        return ko_translate.text
    
    bt_df = df.copy()
    bt_sentence_1 = []
    bt_sentence_2 = []
    for i in tqdm(range(len(bt_df)), desc='BT'):
        bt_sentence_1.append(back_translate(bt_df.loc[i,"sentence_1"]))
        bt_sentence_2.append(back_translate(bt_df.loc[i,"sentence_2"]))
    
    bt_df['sentence_1'] = bt_sentence_1
    bt_df['sentence_2'] = bt_sentence_2

    return bt_df

# 전처리 코드 테스트
if __name__ == "__main__":
    pass