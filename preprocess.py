import os
import re
from networks import Graphnx
from nltk import pos_tag
from nltk import WordNetLemmatizer
from nltk.corpus import wordnet as wn





def lemmatizer(word):
    #word = YOUR_WORD_IN_LOWERCASE
    # Determining the POS tag.
    tag = pos_tag([word])[0][1]
    # Converting it to WordNet format.
    mapping = {'N': wn.NOUN, 'V': wn.VERB, 'R': wn.ADV,'J': wn.ADJ}
    tag_wn = mapping.get(tag[0], wn.NOUN)
    # Lemmatizing.
    lemmatizer = WordNetLemmatizer()
    lemma = lemmatizer.lemmatize(word, tag_wn)
    #print('wwwww   ',lemma.encode('ascii', 'ignore'))
    return lemma.encode('ascii', 'ignore').decode('ascii')


def Dutchpreproc():
    count = 1
    f = open("associationData.csv")
    
    d =  open("preprocdutch.csv",'w')
    
    l = f.readlines()
    l = l[1:]
    for line in l:
        line = line.strip('\n')
        line = re.sub(r'"', '',line)
        #line = line.strip('"')
        #print(count)
        count = count+1
        if True:#count%10 == 0:
            #print(line)
            w = line.split(";")[2:]
            print(w)
            #print(line)
            for i in range(3):
                if len(w[i].split(' ')) > 1:
                    w[i] = 'x'
            d.write(w[0] + ',' + w[1] + ',' + w[2] + ',' + '\n' )
    d.close()
            
            
            
            
def Engpreprocess():
    f = open("./EAT/EATnew.net")
    
    d = open("./EATEATnewLemmas.net", 'w')
    
    l = f.readlines()
    chunk = l[232:23251]
    new_chunk = []
    
    for i in range(len(chunk)):
        line = chunk[i].split(' ',1)
        
        l1 = line[1].strip('\n')
        l1 = l1.strip('\r')
        l1 = re.sub(r'"', '',l1)
        print(line)
        if len(l1.split(' ')) > 1:
            l1 = 'x'
        new_chunk.append(line[0] + ' ' + lemmatizer(l1.lower()) + '\n')
    
    d.writelines(l[:231])
    d.writelines(new_chunk)
    d.writelines(l[23251:])
    d.close()
        
        
    


if __name__ == "__main__":
    #Dutchpreproc()
    Engpreprocess()
    
    
    
    
