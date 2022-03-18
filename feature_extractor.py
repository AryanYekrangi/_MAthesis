import os
import math
import numpy as np
import pandas as pd
import nltk
import textstat
from string import punctuation

class Text:
    def __init__(self, filename:str, directory:str='CEFR_texts'): # file name must include .txt ending
        self.filename = filename
        self.dir = directory
        # CEFR levels
        self.cefr = filename[:2]
        self.cefr_micro_numerical = self.cefr_conv('6 numerical')
        self.cefr_macro_categorical = self.cefr_conv('3 categorical')
        self.cefr_macro_numerical = self.cefr_conv('3 numerical')
        # EXTRACTED
        self.text = self.extract_text()
        self.tokens = nltk.word_tokenize(self.text)
        self.sents = nltk.sent_tokenize(self.text)
        self.sents_tokens = self.extract_sents_tokens()
        self.pos = self.pos_tagger()
        # CLEANED AND PREPROCESSED
        self.sents_tokens_cleaned = self.sents_tokens_clean()
        self.tokens_cleaned = self.tokens_clean()
        self.sents_tokens_lemmatized = self.lemmatize()
        self.tokens_lemmatized = self.lemmatize_tokens()
        self.tokens_stemmed = self.stem_tokens()
        # FEATURES: LEXICAL
        self.awl = self.extract_awl()
        self.ttr = self.extract_ttr()
        self.attro = len(self.tokens)*self.ttr # ORIGINAL ATTR
        self.attr = math.log(len(self.tokens)) * self.ttr
        self.abvmean = self.extract_abv('mean')
        self.abvmin = self.extract_abv('min')
        self.abvmax = self.extract_abv('max')
        self.ajcv = self.extract_ajcv('ajcv')
        self.jcpp = self.extract_ajcv('na_perc')
        self.bpera = self.extract_bpera()
        # FEATURES: SYNTACTIC
        self.asl = self.extract_asl()
        self.avps = self.extract_avps()
        self.apps = self.extract_apps()
        # FEATURES: READABILITY FORMULAS
        self.ari = self.extract_ari()
        self.cli = textstat.coleman_liau_index(self.text)
        self.dcrs = textstat.dale_chall_readability_score_v2(self.text)
        self.fre = textstat.flesch_reading_ease(self.text)
        self.fkg = textstat.flesch_kincaid_grade(self.text)
        self.len = len(self.tokens)
        
                
    def cefr_conv(self, conversion):
        """Converts A1-C2 CEFR levels to sclae or categorical variables based on the conversion variable:
conversion = '6 categorical'
conversion = '6 numerical'
conversion = '3 categorical'
conversion = '3 numerical'"""
        cefr_micro = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']
        if conversion == '6 categorical':
            return self.cefr
        elif conversion == '6 numerical':
            cefr_converted = cefr_micro.index(self.cefr)
            return cefr_converted
        elif conversion == '3 categorical':
            cefr_macro = ['A', 'A', 'B', 'B', 'C', 'C']
            cefr_converted = cefr_macro[cefr_micro.index(self.cefr)]
            return cefr_converted
        elif conversion == '3 numerical':
            cefr_macro = [0, 0, 1, 1, 2, 2]
            cefr_converted = cefr_macro[cefr_micro.index(self.cefr)]
            return cefr_converted

        
    def extract_text(self) -> str:
        """Converts txt file to a string."""
        fhand = open(f'{self.dir}/{self.filename}', encoding='utf-8')
        text = [line.strip() for line in fhand]
        text = ' '.join(text)
        fhand.close()
        return text


    def extract_sents_tokens(self) -> list:
        """Extracts a list of sentences consisting of a list of tokens."""
        sents = self.sents.copy()
        sents_tokens = []
        for sent in sents:
            tokens = nltk.word_tokenize(sent)
            sents_tokens.append(tokens)
        return sents_tokens


    def sents_tokens_clean(self) -> list: # may want to change to lower case as well
        """PREPOCESSING: Removes the folowing:
1. number related
2. punctuation signs""" 
        sents_tokens = self.sents_tokens.copy()
        # replace all tokens that start with number of punc with $removed$
        for i in range(len(self.sents_tokens)):
            for j in range(len(self.sents_tokens[i])):
                if sents_tokens[i][j][0] in '0123456789!"%\'+,-.:;?_`':
                    sents_tokens[i][j] = '$removed$'
        # remove all the $removed$
        for i in range(len(sents_tokens)):
            for j in range(sents_tokens[i].count('$removed$')):
                sents_tokens[i].remove('$removed$')
        return sents_tokens


    def tokens_clean(self) -> list: # may not need
        """Removes numbers and punctuation marks from self.tokens"""
        tokens = self.tokens
        for i in range(len(self.tokens)):
            if self.tokens[i][0] in '0123456789!"%\'+,-.:;?_`':
                tokens[i] = '$removed$'
        for i in range(tokens.count('$removed$')):
            tokens.remove('$removed$')
        return tokens
    

    def pos_tagger(self) -> list:
        """Returns tuples of words and pos in sents_tokens format
The input is self.sents_tokens"""
        all_pos = nltk.pos_tag_sents(self.sents_tokens)
        return all_pos
        

    # EXTRACT FEATURES     
    def extract_asl(self) -> float:
        """Extracts the average sentence length (ASL) feature in tokens
CAUTION: includes punctuations as tokens"""
        sents_tokens = self.sents_tokens.copy()
        sents_lens = [len(sent) for sent in sents_tokens]
        return np.mean(sents_lens)
        

    def extract_awl(self) -> float:
        """Extracts the average word length (AWL) feature
CAUTION: need to create a custom punctuation mark list that includes [’] for instance. not the same as [']"""
        tokens = self.tokens
        for token in tokens:
            if token in punctuation:
                tokens.remove(token)
        tokens_lens = [len(token) for token in tokens]
        return np.mean(tokens_lens)


    def extract_avps(self) -> float:
        """Extracts the average verb per sentence (AVPS) feature"""
        verb_tags = ['VB', 'VBG', 'BVD', 'VBN', 'VBP', 'VBZ']
        all_pos = nltk.pos_tag_sents(self.sents_tokens)
        verbs = [[]for sent in all_pos]
        for i in range(len(all_pos)):
            for j in range(len(all_pos[i])):
                if all_pos[i][j][1] in verb_tags:
                    verbs[i].append(all_pos[i][j])
        vps = [len(sent) for sent in verbs]
        return np.mean(vps)


    def extract_apps(self) -> float:
        """Extracts the average pronouns per sentence (APPS) feature"""
        pronoun_tags = ['PRP', 'PRP$', 'WP']
        all_pos = nltk.pos_tag_sents(self.sents_tokens)
        pronouns = [[]for sent in all_pos]
        for i in range(len(all_pos)):
            for j in range(len(all_pos[i])):
                if all_pos[i][j][1] in pronoun_tags:
                    pronouns[i].append(all_pos[i][j])
        pps = [len(sent) for sent in pronouns]
        return np.mean(pps)


    def extract_ttr(self) -> float:
        """Extracts the type token ratio (TTR) feature
CAUTION: types and tokens include punctuation marks and arabic numbers"""
        tokens = self.tokens
        types =  set(tokens)
        ttr = len(types)/len(tokens)
        return(ttr)


    def extract_abv(self, kind) -> float: # incomplete NA part (may not need NA part)
        """Extracts the average bandwidth value (ABV) from 20k Academic English
kind = 'mean', 'min', 'max'"""
        df = pd.read_csv('allWords - list.csv', index_col = 'word')
        dic = []
        na = [] # NA
        for sent in self.sents_tokens_lemmatized:
            for word in sent:
                if (df.index == word.lower()).any() == True:
                    dic.append(df.loc[word.lower()].band)
                else: # NA
                    na.append(word) # NA
            dic_band = []
            if kind == 'mean':
                for i in dic:
                    dic_band.append(np.mean(i))
            elif kind == 'min':
                for i in dic:
                    if type(i) == pd.Series:
                        dic_band.append(min(i))
                    else:
                        dic_band.append(i)
            elif kind == 'max':
                for i in dic:
                    if type(i) == pd.Series:
                        dic_band.append(max(i))
                    else:
                        dic_band.append(i)
        dic_band_average = sum(dic_band) / len(dic_band)
        dic_band_average = sum(dic_band) / len(dic_band)
        return dic_band_average


    def extract_ajcv(self, output):
        """Extracts the average CEFR-J value (AJCV) feature
output = 'ajcv'
output = 'na'
output = 'na_perc'
method: mean"""
        df = pd.read_csv('jcefr_numerical.csv', index_col = 'headword')
        dic = []
        na = [] # NA
        for word in self.tokens_stemmed:
            if (df.index == word.lower()).any() == True:
                dic.append(df.loc[word.lower()].CEFR)
            else: # NA
                na.append(word) # NA
        cefrl = []
        for i in dic:
            cefrl.append(np.mean(i))
        acv = np.mean(cefrl)
        if output == 'ajcv':
            return acv
        elif output == 'na':
            return na
        elif output == 'na_perc':
            perc = len(self.extract_ajcv('na')) / len(self.tokens_stemmed) * 100
            return perc
        

    def extract_bpera(self,):
        """Extracts the ratio of B words to A words (BPERA) feature
method: mean"""
        df = pd.read_csv('jcefr_numerical.csv', index_col = 'headword')
        dic = []
        na = [] # NA
        for word in self.tokens_stemmed:
            if (df.index == word.lower()).any() == True:
                dic.append(df.loc[word.lower()].CEFR)
            else: # NA
                na.append(word) # NA
        cefrl = []
        for i in dic:
            cefrl.append(np.mean(i))        
        cefrab = []
        for i in cefrl:
            if i < 1.5:
                cefrab.append(0)
            elif i >= 1.5:
                cefrab.append(1)
        bpera = cefrab.count(1)/cefrab.count(0)
        return bpera

        
    def extract_ari(self):
        """Extracts the automated readability index (ARI) feature"""
        grade_level = (0.5 * self.asl) + (4.71 * self.awl) - 21.43
        return grade_level
    

    # NOT FEATURES            
    def lemmatize(self) -> list:
        """Input is self.sents_tokens
Input must be list of list, i.e. list = [['list', 'must', 'be', 'like', 'this'], [...]]
Returns a lemmatized version of sents_tokens"""
        lemmatizer = nltk.stem.WordNetLemmatizer()
        lemmas = self.sents_tokens_cleaned.copy()
        for i in range(len(self.sents_tokens)):
            for j in range(len(lemmas[i])):
                lemmas[i][j] = lemmatizer.lemmatize((self.sents_tokens[i][j]).lower())
        return lemmas


    def lemmatize_tokens(self) -> list:
        """Input is self.tokens
Returns a lemmatized version of tokens"""
        lemmatizer = nltk.stem.WordNetLemmatizer()
        lemmas = self.tokens.copy()
        for i in range(len(self.tokens)):
            lemma = lemmatizer.lemmatize(self.tokens[i])
            lemmas[i] = lemma
        return lemmas


    def stem_tokens(self) -> list:
        """Input is self.tokens_lemmatized
Returns a stemmed version of the lemmatized tokens"""
        df = pd.read_csv('verbs.csv')
        stemmed = []
        for token in self.tokens_lemmatized:
            word = token.lower()
            result = df.isin([word])
            series = result.any()
            cols = list(series[series == True].index)
            all_rows = []
            for col in cols:
                rows = list(result[col][result[col] == True].index)
                all_rows.append(rows)
            if len(all_rows) == 0:
                stem = word
            else:
                ind = all_rows[0][0]
                stem = df[ind:ind+1].stem.values[0]
            stemmed.append(stem)
        return stemmed
         
            
    def write_features(filename='training_data', directory='train_data', cefr_format='6 numerical'): # write what features
        """Text.write_features('training_data', 'train_data', '6 numerical')

General information about each text:
filename
cefr: cefr level of the text (formats include '6 categorical', '6 numerical', '3 categorical', '3 numerical')
Extracts features from texts in the 'CEFR_texts' folder:

The extracted features are:
ABVMAX, ABVMEAN, ABVMIN, AJCV, APPS, ARI, ASL, ASL.AVPS, ATTR, AVPS, AWL, BPERA, CLI, DCRS, FKG, FRE, JCPP, LEN, TTR
"""
        counter = 1
        all_texts = os.listdir(directory)
        # empty lists for y and discriptions
        file_names = []
        cefr_levels = []
        # empty lists for features     
        abvmax = []
        abvmean = []
        abvmin = []
        ajcv = []
        apps = []
        ari = []
        asl = []
        aslavps = []
        attr = []
        avps = []
        awl = []
        bpera = []
        cli = []
        dcrs = []
        fkg = []
        fre = []
        jcpp = []
        length = []
        ttr = []
        for text in all_texts:
            print(f'{counter} / {len(all_texts)}')
            current_text = Text(text, directory=directory)
            # Choosing cefr format
            if cefr_format == '6 categorical':
                cefr = current_text.cefr
            elif cefr_format == '6 numerical':
                cefr = current_text.cefr_micro_numerical
            elif cefr_format == '3 categorical':
                cefr = current_text.cefr_macro_categorical
            elif cefr_format == '3 numerical':
                cefr = current_text.cefr_macro_numerical
            # Appending features for each text into the empty list:
            file_names.append(current_text.filename)
            cefr_levels.append(cefr)
            abvmax.append(current_text.abvmax)
            abvmean.append(current_text.abvmean)
            abvmin.append(current_text.abvmin)
            ajcv.append(current_text.ajcv)
            apps.append(current_text.apps)
            ari.append(current_text.ari)
            asl.append(current_text.asl)
            aslavps.append(current_text.asl*current_text.avps)
            attr.append(current_text.attr)
            avps.append(current_text.avps)
            awl.append(current_text.awl)
            bpera.append(current_text.bpera)
            cli.append(current_text.cli)    
            dcrs.append(current_text.dcrs)
            fkg.append(current_text.fkg)
            fre.append(current_text.fre)
            jcpp.append(current_text.jcpp)
            length.append(current_text.len)
            ttr.append(current_text.ttr)
            counter += 1
        # Converting lists into columnns of a dataframe, which is then converted into a .csv file
        data_tuples = list(zip(file_names, cefr_levels, abvmax, abvmean, abvmin, ajcv, apps, ari, asl, aslavps, attr, avps, awl, bpera, cli, dcrs, fkg, fre, jcpp, length, ttr))
        df = pd.DataFrame(data_tuples, columns=['filename', 'cefr', 'abvmax', 'abvmean', 'abvmin', 'ajcv', 'apps', 'ari', 'asl', 'aslavps', 'attr', 'avps', 'awl', 'bpera', 'cli', 'dcrs', 'fkg', 'fre', 'jcpp', 'len', 'ttr'])
        df.to_csv(f'{filename}.csv', index=False)

if __name__ == '__main__':
    Text.write_features('EXAMPLE_training_data', 'train_data', '6 numerical')
