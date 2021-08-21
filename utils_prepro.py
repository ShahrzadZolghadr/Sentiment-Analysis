# -*- coding: utf-8 -*-
"""utils_prepro.ipynb

Automatically generated by Colaboratory.

    https://colab.research.google.com/drive/1U-wzJ1rsIQHolWTpIlqLpBwcie1d9TND
"""

from cleantext import clean
import re
import hazm

alphabet_dict={'ﺎ':'ا','ﺍ':'ا','ﺑ':'ب','ﺒ':'ﺐ','ﺐ':'ب','ﺏ':'ب','ﺗ':'ت','ﺘ':'ت','ﺖ':'ت','ﺕ':'ت','ﺛ':'ث','ﺜ':'ث','ﺚ':'ث','ﺙ':'ث','ﺟ':'ج','ﺠ':'ج',
               'ﺞ':'ج','ﺝ':'ج','ﺣ':'ح','ﺤ':'ح','ﺢ':'ح','ﺡ':'ح','ﺧ':'خ','ﺨ':'خ','ﺦ':'خ','ﺥ':'خ','ﺪ':'د','ﺩ':'د','ﺬ':'ذ','ﺫ':'ذ','ﺮ':'ر','ﺭ':'ر','ﺰ':'ز',
               'ﺯ':'ز','ﺳ':'س','ﺴ':'س','ﺲ':'س','ﺱ':'س','ﺷ':'ش','ﺸ':'ش','ﺶ':'ش','ﺵ':'ش','ﺻ':'ص','ﺼ':'ص','ﺺ':'ص','ﺹ':'ص','ﺿ':'ض',
               'ﻀ':'ض','ﺾ':'ض','ﺽ':'ض','ﻃ':'ط','ﻄ':'ط','ﻂ':'ط','ﻁ':'ط','ﻇ':'ظ','ﻈ':'ظ','ﻆ':'ظ','ﻅ':'ظ','ﻋ':'ع','ﻌ':'ع','ﻊ':'ع','ﻉ':'ع',
               'ﻏ':'غ','ﻐ':'غ','ﻎ':'غ','ﻍ':'غ','ﻓ':'ف','ﻔ':'ف','ﻒ':'ف','ﻑ':'ف','ﻗ':'ق','ﻘ':'ق','ﻖ':'ق','ﻕ':'ق','ﻛ':'ک','ﻜ':'ک','ﻚ':'ک','ﻙ':'ک',
               'ﻟ':'ل','ﻠ':'ل','ﻞ':'ل','ﻝ':'ل','ﻣ':'م','ﻤ':'م','ﻢ':'م','ﻡ':'م','ﻧ':'ن','ﻨ':'ن','ﻦ':'ن','ﻥ':'ن','ﻫ':'ه','ﻬ':'ه','ﻪ':'ه','ﻩ':'ه','ﻮ':'و',
               'ﻭ':'و','ﻳ':'ی','ﻴ':'ی','ﻲ':'ی','ﻱ':'ی','ﺂ':'آ','ﺁ':'آ','ﺔ':'ه','ﺓ':'ه','ﻰ':'ی','ﻯ':'ی','ئ':'ی','ﭖ':'پ','ﭻ':'چ','ڗ':'ژ','ٶ':'و','ۯ':'ژ',
               'ٱ':'ا','ﺅ':'و','ﮔ':'گ','ﯿ':'ی','ى':'ی','ۃ':'ه','ە':'ه','ة':'ه','ہ':'ه','أ':'ا','ك':'ک','إ':'ا','ۀ':'ه','ھ':'ه','ۆ':'و','ﮐ':'ک','ﭘ':'پ','ﺐ':'ب',
               'ﮏ':'ک','ﭼ':'چ','ﯾ':'ی','ﮕ':'گ','ﮋ':'ژ','ﮑ':'ک','ﭽ':'چ','ۍ':'ی','ﯼ':'ی','ﺆ':'و','ﺌ':'ی','ﺄ':'ا','ﭗ':'پ','ﮎ':'ک','ﮒ':'گ','ګ':'گ',
               'ﭙ':'پ','ﺋ':'ی','ﮓ':'گ','ي':'ی','ﯽ':'ی'}


def edit_alphabet(text):
  output=''
  for t in text:
    if t in alphabet_dict.keys():
      output = output + alphabet_dict.get(t)
    else:
      output = output + t
  return output

def clean_text(t):

  t = edit_alphabet(t)
  t = re.sub(r'[^آابپتثئجچحخدذرزژسشصضطظعغفقکگلمنوهی\d\s]+','',t)
  t = re.sub(r'\d+' , ' N ' , t)
  t = re.sub(r'  ', ' ', t)
  t = re.sub(r'_' , '' , t)
  t = re.sub(r'\.+' , '.' , t)
  t = re.sub(r'\n' , ' ' , t)
  t = re.sub(r'\r|\u2003|\u200a|\u2009|\u3000|\x1f|\u2028|\u2029|\x0c|\u2005|\x85' ,' ' ,t)

  return t

#def remove_stopwords(text):
  #with open('stopwords.dat') as f:
    #stopwords = f.readlines()
  #clean_sample = ' '.join([t for t in text.split() if not t in list(clean_text(stopwords).split())])
  #return clean_sample

#def stem(text):
  #stemmer = hazm.Stemmer()
  #stem_text = [stemmer.stem(w) for w in text.split()]
  #return ' '.join(stem_text)
    

def cleanhtml(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext

def apply_cleaning(text):
    text = text.strip()
    
    # regular cleaning
    text = clean(text,
        fix_unicode=True,
        to_ascii=False,
        lower=True,
        no_line_breaks=True,
        no_urls=True,
        no_emails=True,
        no_phone_numbers=True,
        no_numbers=False,
        no_digits=False,
        no_currency_symbols=True,
        no_punct=False,
        replace_with_url="",
        replace_with_email="",
        replace_with_phone_number="",
        replace_with_number="",
        replace_with_digit="0",
        replace_with_currency_symbol="",
    )

    # cleaning htmls
    text = cleanhtml(text)
    
    # normalizing
    normalizer = hazm.Normalizer()
    text = normalizer.normalize(text)
    
    # removing weird patterns
    weird_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u'\U00010000-\U0010ffff'
        u"\u200d"
        u"\u2640-\u2642"
        u"\u2600-\u2B55"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\u3030"
        u"\ufe0f"
        u"\u2069"
        u"\u2066"
        # u"\u200c"
        u"\u2068"
        u"\u2067"
        "]+", flags=re.UNICODE)
    
    text = weird_pattern.sub(r'', text)
    
    # removing extra spaces, hashtags
    text = re.sub("#", "", text)
    text = re.sub("\s+", " ", text)
    
    # edit alphabets and clean any other unusefull character
    text = clean_text(text)

    # remove stopwords
    #text = remove_stopwords(text)
    #stemming
    #text = stem(text)
    
    return text