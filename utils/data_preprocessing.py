import pandas as pd
import utils
from matplotlib import pyplot as plt
from collections import Counter
from transformers import AutoTokenizer 
import random


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


# reverse text
def reverse(sentence):
  sen_split = sentence.split(" ")
  length = len(sen_split)
  new_sen = ""
  for i in range(length):
    if(i==length-1):
      new_sen = new_sen + sen_split[length-1-i]
      break
    new_sen = new_sen + sen_split[length-1-i]+" "
  sentence = new_sen
  return sentence

# reverse text function
def rev_text(df:pd.DataFrame) -> pd.DataFrame:
  
  # reverse sentence_1
  df['sentence_1']=df['sentence_1'].apply(lambda x : reverse(x))
  # reverse sentence_2
  df['sentence_2']=df['sentence_2'].apply(lambda x : reverse(x))

  return df




# random swap text
def random_swap(sentence):
  sen_split = sentence.split(" ")
  length = len(sen_split)
  new_sen = ""
  random.shuffle(sen_split)
  sentence = " ".join(sen_split)
  return sentence

# reverse text function
def rand_swap_text(df:pd.DataFrame) -> pd.DataFrame:
  # reverse sentence_1
  df['sentence_1']=df['sentence_1'].apply(lambda x : random_swap(x))
  # reverse sentence_2
  df['sentence_2']=df['sentence_2'].apply(lambda x : random_swap(x))

  return df




# 전처리 코드 테스트
if __name__ == "__main__":
    train_df, _, _ = utils.get_data()
    preprocessed_df = rand_swap_text(train_df)

    print('-'*30)
    print("전처리 전", train_df.head(5), sep='\n')
    print('-'*30)
    print("전처리 후", preprocessed_df.head(5), sep='\n')