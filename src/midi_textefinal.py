import py_midicsv as pm
import sys

# Load the MIDI file and parse it into CSV format
def texte(lieu):
    csv_string = pm.midi_to_csv(r"{}".format(lieu))
    lmidi = []
    lfinal = [["0","0","0","0","0"]]

    for i in csv_string:
        intermediaire = i.split(', ')

        if intermediaire[0] == '2'and 'Note' in intermediaire[2]:
            lmidi.append([intermediaire[1],intermediaire[4],intermediaire[5][:-1]]) # on ajoute time pitch velocité

    # rectification on ajoute des notes off là ou il n'y en a pas
            
    unfinished_notes = {}
    for i in range(len(lmidi)) :
        if lmidi[i][2] != "0":
            lfinal.append([lmidi[i][0],lmidi[i][1], lmidi[i][2], "", ""])
            lfinal[-2][4] = (str(int(lfinal[-1][0]) - int(lfinal[-2][0])))
            if ((lmidi[i][1] in unfinished_notes)) :
                unfinished_notes[lmidi[i][1]].append(len(lfinal) - 1)
            else :
                unfinished_notes[lmidi[i][1]] = [len(lfinal) - 1]
        
        if lmidi[i][2] == "0" :
            if ((lmidi[i][1] in unfinished_notes) and (unfinished_notes[lmidi[i][1]] != [])):
                idx = unfinished_notes[lmidi[i][1]][-1]
                lfinal[idx][3] = str(int(lmidi[i][0]) - int(lfinal[idx][0]))
                unfinished_notes[lmidi[i][1]].pop()

    lfinal.pop(0)
    for i in range(len(lfinal)):
        lfinal[i].pop(0)   
    lfinal[-1][3] = "0"
    s = ''
    for i in lfinal:
        s+= 'p'+i[0]+':v'+i[1]+':d'+i[2]+':t'+i[3]+' '
    return s

def main(input_file, output_file):

    with open(output_file, "w") as file:
        file.write(texte(input_file))


