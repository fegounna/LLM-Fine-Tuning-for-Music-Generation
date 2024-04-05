import py_midicsv as pm
import sys


####################################################################
#  We get s_MIDI quadruplets if Pitchbend is not activated
#  We get s_MIDI quintuplets if Pitchbend is activated
####################################################################


def texte(lieu, pitchbend = False):
    csv_string = pm.midi_to_csv(r"{}".format(lieu))
    lmidi = []
    temp_pitchbend_value = '8192' # initial value of pitchbend that has no effect on sound


    if pitchbend:
        lfinal = [["0","0","0","0","0","0"]]
    if not pitchbend:
        lfinal = [["0","0","0","0","0"]]


    for i in csv_string:

        intermediaire = i.split(', ')
        if pitchbend:
            if 'Pitch_bend' in intermediaire[2]:
                temp_pitchbend_value = intermediaire[4][:-1]
        if  'Note' in intermediaire[2]:
        ##############là on écrase toutes les tracks, intermediaire[0] et intermediaire[3] ne sont pas considérés
            if not pitchbend:
                lmidi.append([intermediaire[1],intermediaire[4],intermediaire[5][:-1]]) #  time pitch velocité
            else:
                lmidi.append([intermediaire[1],intermediaire[4],intermediaire[5][:-1],temp_pitchbend_value]) #  time pitch velocité pitchbend_value


    ### because we have potentially  multiple tracks and we don't care, we must reorder them temporally to get the right sequence of notes
    lmidi = sorted(lmidi, key=lambda x: int(x[0]))
  
    # rectification on ajoute des notes off là ou il n'y en a pas
    unfinished_notes = {}
    for i in range(len(lmidi)) :
        if lmidi[i][2] != "0":

            if pitchbend:
                lfinal.append([lmidi[i][0],lmidi[i][1], lmidi[i][2], "","", lmidi[i][3]])

            else:
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

        if pitchbend:
            s+= 'p'+i[0]+':v'+i[1]+':d'+i[2]+':t'+i[3]+':b'+i[4]+' ' #b pour pitchbend
        else:   
            s+= 'p'+i[0]+':v'+i[1]+':d'+i[2]+':t'+i[3]+' '
    print('\n')
    print(s)
    
    return s



def main(input_file, output_file, pitchbend = False):

    with open(output_file, "w") as file:
        file.write(texte(input_file, pitchbend = pitchbend))




# for testing
main("../generated files/midi_data_from_jams/00_BN1-129-Eb_solo.mid","../generated files/00_BN1-129-Eb_solo_converted_NoPB.txt", pitchbend = False)
main("../generated files/midi_data_from_jams/00_BN1-129-Eb_solo.mid","../generated files/00_BN1-129-Eb_solo_converted_PB.txt", pitchbend = True )

'''

# for testing 

header = ["0, 0, Header,1, 2, 384\n", "1, 0, Start_track\n", "1, 0, Tempo, 500000\n", "1, 0, Time_signature, 4, 2, 24, 8\n", "1, 1, End_track\n", "2, 0, Start_track\n"]
with open("../generated files/00_BN1-129-Eb_solo_converted(1).txt","r") as file:
            str = file.read()[:-1] 
text_to_midi.create_midi_file("../generated files/00_BN1-129-Eb_solo_converted(1).mid",str, header )


# for testing 

if  os.path.exists('../generated files/GMP_midis_converted'):
    shutil.rmtree('../generated files/GMP_midis_converted')
os.mkdir('../generated files/GMP_midis_converted')

output_dir_path = "../generated files/GMP_midis_converted"
input_dir_path = "../generated files/GMP_s_midis"
# All files in GMP use the same header
header = ["0, 0, Header,1, 2, 384\n", "1, 0, Start_track\n", "1, 0, Tempo, 500000\n", "1, 0, Time_signature, 4, 2, 24, 8\n", "1, 1, End_track\n", "2, 0, Start_track\n"]

print("beginning untranscription...")
dataset_text_to_midi(output_dir_path, input_dir_path, header)
print("Done")

'''

