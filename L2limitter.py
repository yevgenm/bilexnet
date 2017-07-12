import csv
eng_set = set()

f = open('EAT/shrunkEAT.net','r')

l = f.readlines()
l = l[2:23219]
for i in l:
    line = i.split(' ')[1].strip('\n').strip('\r').strip('"')
    eng_set.add(line)
    
f.close()

f = open('/Users/amirardalankalantaridehgahi/Desktop/school/stevensonRA/clone/bilexnet/sothflorida_complete.csv' ,'r')
l = f.readlines()
for i in l:
    line = i.split(';')
    eng_set.add(line[0].strip('\n').strip('\r').strip('"'))
    eng_set.add(line[1].strip('\n').strip('\r').strip('"'))
    
f.close()

written = open('zipf.csv','w')

print(sorted(eng_set))
raw_input()
goog = ['a_goog', 'b_goog', 'c_goog']
l = None
for zipf in goog:
    print(zipf)
    f = open(zipf, 'r')
    
    
    prev_word = None
    prev_count = 0
    while True:
        #print(prev_word)
        l = f.readline()
        #print(l)
        #raw_input()
        if not l: break
        l = l.split('\t')
        word = l[0].split('_')[0].lower()
       
        if word == prev_word:
            prev_count = prev_count + int(l[2])
            
        else:
            #print(word)
            if prev_word in eng_set:
                #print(word + '   ' + prev_word + '  ' + str(prev_count) + '\n')
                #raw_input()
                written.write(prev_word + '  ' + str(prev_count) + '\n')
            prev_word = word
            prev_count = 0
            #print(word + '   ' + prev_word + '  ' + str(prev_count) + '\n')
    f.close()
    
    
