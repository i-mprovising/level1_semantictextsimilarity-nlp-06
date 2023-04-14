import pandas as pd

from transformers import pipeline
from googletrans import Translator
from tqdm.auto import tqdm

# spellceck
from pykospacing import Spacing
from symspellpy_ko import KoSymSpell, Verbosity

def remove_special_word(df):
    df['sentence_1'] = df['sentence_1'].str.replace(pat=r'[^\w]', repl=r' ', regex=True)
    df['sentence_2'] = df['sentence_2'].str.replace(pat=r'[^\w]', repl=r' ', regex=True)

    return df

def swap_sentence(df:pd.DataFrame) -> pd.DataFrame:
    """
    sentence1과 sentence2를 바꾸는 data augmentation 기법입니다.

    Args:
        df (pd.DataFrame): 원본 train data

    Returns:
        pd.DataFrame
    """
    swapped_df = df.copy(deep=True)
    swapped_df['sentence_1'], swapped_df['sentence_2'] = swapped_df['sentence_2'], swapped_df['sentence_1']
    return swapped_df
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

def spellcheck(df):
    """
    sentence_1과 sentence_2의 띄어쓰기 및 맟춤법을 교정해주는 함수입니다.

    필요 라이브러리 :
    pip install git+https://github.com/haven-jeon/PyKoSpacing.git
    pip install symspellpy-ko

    Args:
        df (pd.DataFrame): 원본 train data

    Returns:
        pd.DataFrame    
    """
    spacing = Spacing()
    symspell = KoSymSpell()
    symspell.load_korean_dictionary(decompose_korean = True, load_bigrams=True)
    checked_df = df.copy()
    sen1 = []
    sen2 = []

    # 동일한 spacing 적용을 위해 공백을 제거합니다.
    checked_df['sentence_1'] = checked_df['sentence_1'].str.replace(pat=r'[^ㄱ-ㅎㅏ-ㅣ가-힣a-zA-Z0-9//?.,;:|*~`!^/-_+<>@/#$%&/=]+', repl=r'', regex=True)
    checked_df['sentence_2'] = checked_df['sentence_2'].str.replace(pat=r'[^ㄱ-ㅎㅏ-ㅣ가-힣a-zA-Z0-9//?.,;:|*~`!^/-_+<>@/#$%&/=]+', repl=r'', regex=True)

    # spacing을 적용합니다.
    for i in tqdm(range(len(checked_df)), desc='spacing'):
        sen1.append(spacing(checked_df['sentence_1'][i]))
        sen2.append(spacing(checked_df['sentence_2'][i]))

    # 맞춤법을 체크합니다.
    for i in tqdm(range(len(checked_df)), desc='spell'):
        term = sen1[i]
        for suggestion in symspell.lookup_compound(term, max_edit_distance=0):
            sen1[i] = suggestion.term
        term = sen2[i]
        for suggestion in symspell.lookup_compound(term, max_edit_distance=0):
            sen2[i] = suggestion.term

    checked_df['sentence_1'] = sen1
    checked_df['sentence_2'] = sen2

    return checked_df




# 전처리 코드 테스트
if __name__ == "__main__":
    train_df, _, _ = utils.get_data()
    preprocessed_df = swap_sentence(train_df)

    print('-'*30)
    print("전처리 전", train_df.head(5), sep='\n')
    print('-'*30)
    print("전처리 후", preprocessed_df.head(5), sep='\n')