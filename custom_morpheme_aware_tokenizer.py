from tokenizers import NormalizedString, PreTokenizedString
from typing import List
from konlpy.tag import Mecab
import re

class morphemeTokenizer:
    def morpheme_split(self, i: int, normalized_string: NormalizedString) -> List[str]:
        mecab = Mecab()
        splits = []
        start = 0
        end = 0
        tmp_string_to_append = normalized_string
        tmp_string = normalized_string.normalized
        for s in mecab.morphs(normalized_string.normalized):
            end = len(s)
            splits.append(tmp_string_to_append[start:end])
            tmp_string = tmp_string[end:]
            tmp_string_to_append = tmp_string_to_append[end:]
            ## 제일 앞에오는 공백 제거해주기
            if re.match(r" +", tmp_string) is not None:
                new_start_idx = re.match(r" +", tmp_string).span()[1]
                tmp_string = tmp_string[new_start_idx:]
                tmp_string_to_append = tmp_string_to_append[new_start_idx:]
        return splits

    def pre_tokenize(self, pretokenized_string: PreTokenizedString):
        pretokenized_string.split(self.morpheme_split)