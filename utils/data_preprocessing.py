import pandas as pd
import utils
import random
import re

from tqdm.auto import tqdm
from hangulize import hangulize

# spellceck
from pykospacing import Spacing
from symspellpy_ko import KoSymSpell, Verbosity
from koeda import SR, RI, RS, RD

def remove_special_word(df):
    """
    특수문자를 제거하는 메소드
    """
    df['sentence_1'] = df['sentence_1'].str.replace(pat=r'[^\w]', repl=r' ', regex=True)
    df['sentence_2'] = df['sentence_2'].str.replace(pat=r'[^\w]', repl=r' ', regex=True)

    return df


def random_deletion(df):
    """
    apply로 실행시켜야 함
    k개의 워드를 랜덤으로 삭제 기법 (일단 k=1) -> 나중에 보유 문장 길이에 따라 동적으로 변경시킬 예정
    조합 3개 (ww표시가 전처리 된 거)
    1. sen_1, sen_2ww
    2. sen_1ww, sen_2
    3. sen_1ww, sen_3ww
    """
    def func(x):
        x = x.split()
        random_item = random.choice(x)
        x.remove(random_item)

        return ' '.join(x)
    
    sen_1ww = df['sentence_1'] = df['sentence_1'].apply(lambda x: func(x)).values.tolist()
    sen_2ww = df['sentence_2'] = df['sentence_2'].apply(lambda x: func(x)).values.tolist()

    sen_1, sen_2 = [], []
    labels = [label for _ in range(3) for label in df['label']] if 'label' in df.columns else []

    # 1
    sen_1.extend(df['sentence_1'].values.tolist())
    sen_2.extend(sen_2ww)
    # 2
    sen_1.extend(sen_1ww)
    sen_2.extend(df['sentence_2'].values.tolist())
    # 3
    sen_1.extend(sen_1ww)
    sen_2.extend(sen_2ww)

    new_df = pd.DataFrame()
    new_df['sentence_1'] = sen_1
    new_df['sentence_2'] = sen_2
    if labels: new_df['label'] = labels

    return new_df

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

def sr(df):
    """
    문장 내 단어를 유의어로 바꿔주는 함수입니다.

    필요 라이브러리 :
    pip install koeda

    Args:
        df (pd.DataFrame): 원본 data

    Returns:
        pd.DataFrame    
    """
    sr_df = df.copy()
    sen1 = df['sentence_1'].to_list()
    sen2 = df['sentence_2'].to_list()

    func = SR("Okt")
    sen1 = func(sen1, 0.3)
    sen2 = func(sen2, 0.3)

    sr_df['sentence_1'] = sen1
    sr_df['sentence_2'] = sen2    

    return sr_df

def ri(df):
    """
    문장 내에 유의어를 무작위로 삽입하는 함수입니다.

    필요 라이브러리 :
    pip install koeda

    Args:
        df (pd.DataFrame): 원본 data

    Returns:
        pd.DataFrame    
    """
    ri_df = df.copy()
    sen1 = df['sentence_1'].to_list()
    sen2 = df['sentence_2'].to_list()

    func = RI("Okt")
    sen1 = func(sen1, 0.3)
    sen2 = func(sen2, 0.3)

    ri_df['sentence_1'] = sen1
    ri_df['sentence_2'] = sen2    

    return ri_df

def rs(df):
    """
    문장 내 단어들의 위치를 바꾸는 함수입니다.
    필요 라이브러리 :
    pip install koeda

    Args:
        df (pd.DataFrame): 원본 data

    Returns:
        pd.DataFrame    
    """
    rs_df = df.copy()
    sen1 = df['sentence_1'].to_list()
    sen2 = df['sentence_2'].to_list()

    func = RS("Okt")
    sen1 = func(sen1, 0.3)
    sen2 = func(sen2, 0.3)

    rs_df['sentence_1'] = sen1
    rs_df['sentence_2'] = sen2    

    return rs_df

def rd(df):
    """
    문장 내 단어를 무작위로 삭제하는 함수입니다.

    필요 라이브러리 :
    pip install koeda

    Args:
        df (pd.DataFrame): 원본 data

    Returns:
        pd.DataFrame    
    """
    rd_df = df.copy()
    sen1 = df['sentence_1'].to_list()
    sen2 = df['sentence_2'].to_list()

    func = RD("Okt")
    sen1 = func(sen1, 0.3)
    sen2 = func(sen2, 0.3)

    rd_df['sentence_1'] = sen1
    rd_df['sentence_2'] = sen2    

    return rd_df

def text_style_transfer(df):
    """
    remove_special_word, spellcheck를 진행한 train.csv에 text_style_transfer를 진행한 csv를 불러옵니다.
    스타일 변환은 문어체, 구어체 두 가지로 진행되었습니다.
    """
    """
    text style을 세 가지로 바꾸는 augmentation. 

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
    
    def get_transferred_text(text, target_style, **kwargs):
      input = f"{target_style} 말투로 변환:{text}"
      out = model(input, max_length=64, **kwargs)
      return out[0]['generated_text']
    
    sen1 = []
    sen2 = []
    spoken = df.copy()
    for i in tqdm(range(len(df))):
        item = df.iloc[i]
        sen2.append(get_transferred_text(item['sentence_2'], style[0])) #sentence2를 구어체로 변환
    spoken['sentence_2'] = sen2
    written = df.copy()
    for i in tqdm(range(len(df))):
        item = df.iloc[i]
        sen1.append(get_transferred_text(item['sentence_1'], style[1])) #sentence1을 문어체로 변환
    written['sentence_1'] = sen1

    new_df = pd.concat([spoken, written])
    return new_df #ts+wt

    # ts = pd.read_csv("./data/train_spoken.csv")
    # wt = pd.read_csv("./data/written_train.csv")
    # tw = pd.read_csv("./def text_style_transfer(df)
    # st = pd.read_csv("./data/spoken_train.csv")
    # new_df = pd.concat([ts, wt, tw, st])
    # return new_df

def create_5(df):
    """
    label 0에서 랜덤으로 1200개를 추출해 label 5로 변경해주기
    sentence_2가 문장 길이 10 정도를 제외하고 sentence_1보다 길기 때문에 정보를 더 많이 담고 있을 것 같아 sentence_2로 label 5 만들기

    train_df 기준 label 분포는 >
    0: 3711
    1: 1368
    2: 1137
    3: 1715
    4: 1302
    5: 91
    """
    label_0_index = label_0_index = df[df['label'] == 0.0].index.tolist()
    change_index = random.sample(label_0_index, 400) # 원래 1200

    new_df = df.loc[change_index, :]
    new_df['sentence_1'] = new_df['sentence_2']
    new_df['label'] = 5.0
    df.drop(change_index, axis=0, inplace=True)
    df.reset_index(drop=True, inplace=True)

    return new_df


def create_5_1(df):
    """
    label 0에서 랜덤으로 1200개를 추출해 sentence_1을 기준으로 label 5로 변경해주기

    train_df 기준 label 분포는 >
    0: 3711
    1: 1368
    2: 1137
    3: 1715
    4: 1302
    5: 91
    """
    label_0_index = label_0_index = df[df['label'] == 0.0].index.tolist()
    change_index = random.sample(label_0_index, 400) # 원래 1200

    new_df = df.loc[change_index, :]
    new_df['sentence_2'] = new_df['sentence_1']
    new_df['label'] = 5.0
    df.drop(change_index, axis=0, inplace=True)
    df.reset_index(drop=True, inplace=True)

    return new_df

def process_eng(df):
    new_df = df.copy()
    reg = re.compile(r'[a-zA-Z]')
    sen1 = []
    sen2 = []
    for idx, row in df.iterrows():
        if(reg.match(row['sentence_1']) or reg.match(row['sentence_2'])):
            han_1 = hangulize(row['sentence_1'], 'cym')
            han_2 = hangulize(row['sentence_2'], 'cym')
            sen1.append(han_1)
            sen2.append(han_2)
        else:
            sen1.append(row['sentence_1'])
            sen2.append(row['sentence_2'])
    new_df['sentence_1'] = sen1
    new_df['sentence_2'] = sen2
    
    return new_df


def create_5_mix(df):
    """
    label 0에서 랜덤으로 1200개를 추출해 sentence_1, sentence_2를 반반으로 label 5로 변경해주기

    train_df 기준 label 분포는 >
    0: 3711
    1: 1368
    2: 1137
    3: 1715
    4: 1302
    5: 91
    """
    new_df = []
    for _ in range(2):
        label_0_index = label_0_index = df[df['label'] == 0.0].index.tolist()
        change_index = random.sample(label_0_index, 400) # 원래 1200
        focus_df = df.loc[change_index, :]
        if _ == 0:
            focus_df['sentence_2'] = focus_df['sentence_1']
        else:
            focus_df['sentence_1'] = focus_df['sentence_2']
        focus_df['label'] = 5.0
        df.drop(change_index, axis=0, inplace=True)
        df.reset_index(drop=True, inplace=True)

        new_df.append(focus_df)

    return pd.concat([new_df[0], new_df[1]], axis=0)


def remove_consonant(df):
    df['sentence_1'] = df['sentence_1'].str.replace(pat=r'[ㄱ-ㅎㅏ-ㅣ]+', repl=r'', regex=True)
    df['sentence_2'] = df['sentence_2'].str.replace(pat=r'[ㄱ-ㅎㅏ-ㅣ]+', repl=r'', regex=True)

    return df


# 전처리 코드 테스트
if __name__ == "__main__":
    train_df, _, _ = utils.get_data()
    preprocessed_df = text_style_transfer(train_df)

    print('-'*30)
    print("전처리 전", train_df.head(5), sep='\n')
    print('-'*30)
    print("전처리 후", preprocessed_df.head(5), sep='\n')
    print(len(train_df), len(preprocessed_df))
