# Appendix (D): Python files

This repository contains Python files used for Aryan Yekrangi's MA thesis with the title "Leveraging simple features and machine learning approaches for text level assessment".
University of Eastern Finland, MA Linguistic Data Sciences

- The sub-folder Experiment 1 contains the Python files written for Section 4.1 of the MA thesis. It consists of five Python files, each corresponding to one classification approach. The feature set used to train and test each classifier can be defined on line 26 of each Python file.

- The sub-folder Experiment 2 contains the Python files written for Section 4.2 of the MA thesis. It consists of three Python files, each corresponding to onne classification approach. The feature set used to train and test each classifier can be defined on line 22 of each Python file.

- The sub-folder Feature Extractor contains one Python program which was used to convert .txt files into .csv data frame. This sub-folder also contains a few   additional data fiels, mostly in the form of word lists, which were used to calculate some of the features or preprocess the data. These are:
  - <i>allWords - list.csv</i>    Modified COCA Academic texts word list.
  - <i>jcefr_numerical.csv</i>    Modified CEFR-J word list.
  - <i>verbs.csv</i>              A .csv file containing the 1000 most common English verbs in different forms. This data frame was used for converting verb tokens to their dictionary form. The data frame can be accessed through https://www.worldclasslearning.com/english/five-verb-forms.html.

- The sub-folder Feature Selection contains two Python files

References: </br>
The CEFR-J Wordlist Version 1.6. Compiled by Yukio Tono, Tokyo University of Foreign Studies. Retrieved from http://cefr-j.org/download.html on 07/06/2021.</br>
Gardner, D., & Davies, M. (2014). A new academic vocabulary list. Applied linguistics, 35(3), 305-327. Retrieved from https://www.academicvocabulary.info/download.asp on 11/06/2021.</br>
