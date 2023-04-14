import pandas as pd
from typing import List
import utils
import data_preprocessing


class SequentialCleaning:
    def __init__(self, cleaning_list: List[str]=[]):
        """
        여러 개의 cleaning method를 데이터 프레임에 순차적으로 적용하는 클래스입니다.

        Args:
            cleaning_list (List[str], optional): 적용할 cleaning method의 이름 리스트입니다.
        
        Example:
            sc = SequentialCleaning(["remove_special_word", "remove_stop_word"])
            train_df = pm.process(train_df)
        """
        self.cleaning_list = cleaning_list
    
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        입력으로 받은 데이터프레임에 cleaning 메소드를 순차적으로 적용해서 반환합니다.

        Args:
            df (pd.DataFrame)

        Returns:
            pd.DataFrame
        """
        method_package_name = "data_preprocessing."
        for method in self.cleaning_list:
            cleaning_method = eval(method_package_name + method)
            df = cleaning_method(df)

        return df


class SequentialAugmentation:
    def __init__(self, augmentation_list: List[str]=[]):
        """
        여러 개의 augmentation 기법을 데이터 프레임에 순차적으로 적용하는 클래스입니다.

        Args:
            augmentation_list (List[str], optional): 적용할 augmentation method의 이름 리스트입니다.
        
        Example:
            sa = SequentialAugmentation(["swap_sentence", "back_translation"])
            train_df = sa.process(train_df)
        """
        self.augmentation_list = augmentation_list
    
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        입력으로 받은 데이터프레임에 augmentation 메소드를 순차적으로 적용해서 반환합니다.

        Args:
            df (pd.DataFrame)

        Returns:
            pd.DataFrame
        """
        method_package_name = "data_preprocessing."
        # augmentation으로 만든 데이터를 또 augmentation하지 않기 위해 augmented_df에 따로 저장합니다.
        augmented_df = pd.DataFrame(columns=df.columns)
        for method in self.augmentation_list:
            augmentation_method = eval(method_package_name + method)
            augmented_df = pd.concat([augmented_df, augmentation_method(df)])
        df = pd.concat([df, augmented_df])

        return df


# 테스트 코드
if __name__ == "__main__":
    train_df, _, _ = utils.get_data()
    sa = SequentialAugmentation(['swap_sentence'])
    preprocessed_df = sa.process(train_df)

    print('-'*30)
    print("전처리 전", train_df.tail(5), train_df.shape, sep='\n')
    print('-'*30)
    print("전처리 후", preprocessed_df.tail(5), preprocessed_df.shape, sep='\n')
