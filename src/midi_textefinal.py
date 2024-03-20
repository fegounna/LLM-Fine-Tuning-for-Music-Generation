import py_midicsv as pm
# Load the MIDI file and parse it into CSV format
def texte(lieu):
    csv_string = pm.midi_to_csv(r"{}".format(lieu))

    l = []

    for i in csv_string[:100]:
        int = i.split(', ')

        if int[0] == '2'and 'Note' in int[2]:
            l.append([int[1],int[4],int[5][:-1]]) # on ajoute time pitch velocité

    # rectification on ajoute des notes off là ou il n'y en a pas
            

    i=0
    while i<len(l):
        if l[i][2]!='0':
            note = l[i][1]

            compteur = 0
            for j in range(i+1,len(l)):
                if l[j][1] == note:
                    if l[j][2]== '0' and compteur == 0:
                        a =l[j][0]
                        b= l[i][0]
                        l[i].append (str(float(a)-float(b)))
                        break
                    elif l[j][2]== '0' and compteur != 0:
                        compteur -=1
                    else:
                        compteur +=1
        i+=1
    i=0
    while i <len(l):
        if len(l[i]) ==3:
            l.remove(l[i])
            i-=1
        i+=1
    for i in l:
        i[3] = i[3][:-2]

    for i in range(len(l) -1) :
        l[i].append(str(float(l[i+1][0])- float(l[i][0]))[:-2])

    for i in range(len(l)):
        l[i].remove(l[i][0])   
    l[-1].append('0')
    s = ''
    for i in l:
        s+= 'p'+i[0]+':v'+i[1]+':d'+i[2]+':t'+i[3]+' '
    return s


lieu = r"C:\Users\cypri\Desktop\python PSC\Chopin, Frédéric, Études, Op.10, g0hoN6_HDVU.mid"
print(texte(lieu))