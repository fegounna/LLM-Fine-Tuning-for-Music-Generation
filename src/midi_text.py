import py_midicsv as pm

# Load the MIDI file and parse it into CSV format
csv_string = pm.midi_to_csv(r"C:\Users\Moakher\Desktop\psc\Ravel, Maurice, Jeux d'eau, v-QmwrhO3ec.mid")
l = []
for ligne in csv_string:
    if 'Note_on' in ligne and ligne[0] == '2':
        l.append(ligne[:-1])
for i in range(len(l)):
    l[i] = l[i].split(', ')
    l[i].remove(l[i][2])
    l[i].remove(l[i][0])
    for j in range(4):
        l[i][j] = int(l[i][j])

# Implémenter les durées des notes dans une nouvelle colonne
for i in range(len(l)):
    # on ne considère que les notes à vélocité non nulle
    if (l[i][3]!=0):
        j=i
        note1=l[i][2]
        note2=-1
        while (note1 != note2):
            note2=l[j+1][2]
            j=j+1
        # on a trouvé le premier j>i pour lequel on a la même note : c'est le note off !
        durée=l[j][0]-l[i][0]
        #on ajoute la durée à la liste l[i]
        l[i].append(durée)        

# on n'a maintenant plus besoin des lignes correspondant à des "note off", on les supprime
k=0
while (k<len(l)):
    if (l[k][3]==0):
        l.remove(l[k])
        k -= 1
    k=k+1
        
#maintenant pour une ligne correspondant à une note donnée, la ligne d'après est nécessairement la note suivante à être jouée
#on peut donc facilement calculer pour chaque note la durée avant la note suivante
for i in range(len(l)-1):
    #time before next note : tbnn
    tbnn = l[i+1][0]-l[i][0]
    #on ajoute à la liste
    l[i].append(tbnn)  
#il manque la dernière note qui n'a pas de note ensuite, on donne par convention un tbnn de -1
l[len(l)-1].append(-1)
#on se débarasse maintenant de la colonne du temps absolu dont on n'a pas besoin pour le simplify midi
s = ""
for i in range(len(l)):  
    s+="p" + str(l[i][2])+":" + "v" + str(l[i][3]) +":" + "d" + str(l[i][4]) + ":t" + str(l[i][5])+" "
    

print(s)

#with open(r"C:\Users\cypri\Desktop\python PSC\example_converted.csv", "w") as f:
#    f.writelines(csv_string)

# Parse the CSV output of the previous command back into a MIDI file
#midi_object = pm.csv_to_midi(csv_string)

# Save the parsed MIDI file to disk
#with open("example_converted.mid", "wb") as output_file:
#    midi_writer = pm.FileWriter(output_file)
#    midi_writer.write(midi_object)
