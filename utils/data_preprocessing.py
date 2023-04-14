import pandas as pd
import utils
import random

from sklearn.model_selection import train_test_split

def remove_special_word(df):
    """
    특수문자를 제거하는 메소드
    """
    df['sentence_1'] = df['sentence_1'].str.replace(pat=r'[^\w]', repl=r' ', regex=True)
    df['sentence_2'] = df['sentence_2'].str.replace(pat=r'[^\w]', repl=r' ', regex=True)

    return df

def df_split(train_df, val_df, seed):
    """
    대회에서 제공한 train과 val 데이터를 합치고 다시 나누기
    """
    X = pd.concat([train_df, val_df], axis=0)
    y = X['label']
    X.drop(['label'], axis=1, inplace=True)

    train_x, val_x, train_y, val_y = train_test_split(X, y, test_size=0.055, stratify=y, random_state=seed)

    train_x['label'] = train_y
    val_x['label'] = val_y

    return train_x, val_x

def random_deletion(x):
    """
    apply로 실행시켜야 하는 랜덤 삭제 기법
    """
    x = x.split()
    random_item = random.choice(x)
    
    try:
        x.remove(random_item)
    except:
        print(x)

    return ' '.join(x)

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

# 전처리 코드 테스트
if __name__ == "__main__":
    train_df, _, _ = utils.get_data()
    preprocessed_df = swap_sentence(train_df)

    print('-'*30)
    print("전처리 전", train_df.head(5), sep='\n')
    print('-'*30)
    print("전처리 후", preprocessed_df.head(5), sep='\n')