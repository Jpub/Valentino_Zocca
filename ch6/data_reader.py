from __future__ import print_function, division

from six.moves import range
import codecs
import numpy as np

# 저자의 레포지토리에서 가지고옴 

class DataReader(object):
    """언어모델을 훈련시키기위한 데이터 읽는 파일"""
    def __init__(self, filepath, batch_length, batch_size):
        self.batch_length = batch_length
        self.batch_size = batch_size
        # 데이터를 읽어 스트링으로 바꾼다
        with codecs.open(filepath, encoding='utf-8', mode='r') as f:
            self.data_str = f.read()
        self.data_length = len(self.data_str)
        print('데이터 길이: ', self.data_length)
        # 철자들로 구성된 리스트를 생성한다. 소프트맥스 실행을 위해 글자들을 자른다. 
        char_set = set()
        for ch in self.data_str:
            char_set.add(ch)
        self.char_list = sorted(list(char_set))
        print('char_list: ', len(self.char_list), self.char_list)
        # 철자 기반으로 인덱스를 역순으로 해서 다시 매핑한다. 
        self.char_dict = {val: idx for idx, val in enumerate(self.char_list)}
        print('char_dict: ', self.char_dict)
        # 무작위로 적당한 시작시점(인덱스)를 찾는다. 
        self.reset_indices()

    def reset_indices(self):
        self.start_idxs = np.random.random_integers(
            0, self.data_length, self.batch_size)

    def get_sample(self, start_idx, length):
        # Get a sample and wrap around the data string
        return [self.char_dict[self.data_str[i % self.data_length]]
                for i in range(start_idx, start_idx+length)]

    def get_input_target_sample(self, start_idx):
        sample = self.get_sample(start_idx, self.batch_length+1)
        inpt = sample[0:self.batch_length]
        trgt = sample[1:self.batch_length+1]
        return inpt, trgt

    def get_batch(self, start_idxs):
        input_batch = np.zeros((self.batch_size, self.batch_length),
                               dtype=np.int32)
        target_batch = np.zeros((self.batch_size, self.batch_length),
                                dtype=np.int32)
        for i, start_idx in enumerate(start_idxs):
            inpt, trgt = self.get_input_target_sample(start_idx)
            input_batch[i, :] = inpt
            target_batch[i, :] = trgt
        return input_batch, target_batch

    def __iter__(self):
        while True:
            input_batch, target_batch = self.get_batch(self.start_idxs)
            self.start_idxs = (
                self.start_idxs + self.batch_length) % self.data_length
            yield input_batch, target_batch


def main():
    filepath = './wap.txt'
    batch_length = 10
    batch_size = 2
    reader = DataReader(filepath, batch_length, batch_size)
    s = 'As in the question of astronomy then, so in the question of history now,'
    print([reader.char_dict[c] for c in s])

if __name__ == "__main__":
    main()
