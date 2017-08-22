import preprocess
import re
import csv


'''
CONVENTIONS ::::
    '>' indicates ENGLISH to DUTCH direction
    '<' indictaes DUTCH to ENGLISH direction
    
    dictionaries:::
        cue: dict(associate1:value1, associate2:value2 .........)
        
    
    FINAL produced file format::
    
    ENGLISH_WORD     DUTCH_WORD         E->D alignment probability              D->E alignment probability 


'''
def normalizer(d):
    for cue in d:
        
        summ = 0.0
        for norm in d[cue]:
            summ = summ + d[cue][norm]
        
        for norm in d[cue]:
            d[cue][norm] = d[cue][norm]/summ
            
    return d

def eng_to_dutch_dict(fil,Dutchlemma=0 ):
    
    freq = dict()
    
    if Dutchlemma:
        import frog
        frog = frog.Frog(frog.FrogOptions(parser=False))
    
    
    f = open(fil,'r')
    lines = f.readlines()
    
    lines = lines[3:1178111]
    cue = None 
    
    for l in lines:
        #print(l)
        if l[0] != ' ':
            cue = l.split('\t')[0]
            cue =  re.sub(r"[^\w-]", '', cue)
            cue = cue.strip('-')
            eng = cue
            if eng != '':
                eng = preprocess.lemmatizer(eng)
                if eng not in freq:
                    freq[eng] = dict()
                
                
            
        if l[0] == ' ':
            if eng != '':
                dutch = l.strip()
                dutch = l.rsplit(':',1)[0]
                dutch = re.sub(r"[^\w-]", '', dutch)
                dutch = dutch.strip('-')
                if Dutchlemma:
                    dutch = frog.process(dutch)
                    
                value = l.rsplit(':',1)[1]
                value = value.strip()
                value = float(value)
                
                
                if dutch != '':
                    if dutch not in freq[eng]:
                        freq[eng][dutch] = value
                        
                    else:
                        freq[eng][dutch] = freq[eng][dutch] + value
                    

    #print(freq)
    print('DONE SO FAR!!!!!!!!!!!')
    #input()
    
    freq = normalizer(freq)
    return freq


###############################################################################################################################################################################


def dutch_to_eng_dict(fil,Dutchlemma=0 ):
    
    freq = dict()
    
    if Dutchlemma:
        import frog
        frog = frog.Frog(frog.FrogOptions(parser=False))
    
    
    f = open(fil,'r')
    lines = f.readlines()
    
    lines = lines[3:1206287]
    cue = None 
    
    for l in lines:
        #print(l)
        if l[0] != ' ':
            cue = l.split('\t')[0]
            cue =  re.sub(r"[^\w-]", '', cue)
            cue = cue.strip('-')
            dutch = cue
            if dutch != '':
                if Dutchlemma:
                    dutch = frog.process(dutch)

                if dutch not in freq:
                    freq[dutch] = dict()
                
                
            
        if l[0] == ' ':
            if dutch != '':
                eng = l.strip()
                eng = l.rsplit(':',1)[0]
                eng = re.sub(r"[^\w-]", '', eng)
                eng = eng.strip('-')
                if eng != '':
                    eng = preprocess.lemmatizer(eng)
                        
                    value = l.rsplit(':',1)[1]
                    value = value.strip()
                    value = float(value)
                    
                    if eng not in freq[dutch]:
                        freq[dutch][eng] = value
                        
                    else:
                        freq[dutch][eng] = freq[dutch][eng] + value
                    

    #print(freq)
    print('DONE SO FAR!!!!!!!!!!!')
    #input()
    
    freq = normalizer(freq)
    
    return freq





def csv_writer(Dlemma=0):
    e_to_d = eng_to_dutch_dict('hmm/1.params.txt',Dlemma)
    d_to_e = dutch_to_eng_dict('hmm/2.params.txt',Dlemma)
    
    with open('aligns.csv', 'w') as csvfile:
        fieldnames = ['ENGLISH', 'DUTCH','English->Dutch' , 'Dutch->English']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        #first take care of e_to_d
        for eng in e_to_d:
            for dutch in e_to_d[eng]:
                value = e_to_d[eng][dutch]
                
                if dutch in d_to_e and eng in d_to_e[dutch]:
                    reverse = d_to_e[dutch][eng]
                    
                else:
                    reverse = '-----'
                    
                
                writer.writerow({'ENGLISH': eng, 'DUTCH': dutch, 'English->Dutch':value ,'Dutch->English':reverse  })
                
        
        # NOW we deal with d-to-e
        
        for dutch in d_to_e:
            for eng in d_to_e[dutch]:
                
                if eng in e_to_d and dutch in e_to_d[eng]:
                    pass
                
                else:
                    writer.writerow({'ENGLISH': eng, 'DUTCH': dutch, 'English->Dutch':'-----','Dutch->English':d_to_e[dutch][eng]  })
        
        
        
    
 
 
 
if __name__ == "__main__":
     #eng_to_dutch_dict('hmm/1.params.txt')
     #dutch_to_eng_dict('hmm/2.params.txt')
     csv_writer(0)
