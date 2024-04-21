import py_midicsv as pm
import sys





# get the list of possible names for lead tracks : in file DadaGP_bended_track_names.txt
with open('DadaGP_bended_track_names.txt', 'r') as file:
    track_names_line = file.readline()
accounted_track_names  = track_names_line.strip()[1:-1].split(', ')

# Remove any extra quotes and newline characters from each track name
accounted_track_names = [str(track.strip("'\"\\n")) for track in accounted_track_names]
accounted_track_names = str(accounted_track_names).replace("'", '"')





####################################################################
#  We get s_MIDI quadruplets if Pitchbend is not activated
#  We get s_MIDI quintuplets if Pitchbend is activated
####################################################################

def to_text(input_file, output_file, pitchbend = False):
    csv_string = pm.midi_to_csv(r"{}".format(input_file))
    lmidi = []

    temp_pitchbend_value = '23' # initial value of pitchbend that has no effect on sound (8192 correspond to 23)


    ############### on ne garde que les tracks de guitare
    guitar_tracks = []

    
 


    for i in csv_string:

        intermediaire = i.split(', ')
        if 'Title_t' in intermediaire[2]:

            # check if the track is a guitar track : beware of the quotes : e.g.   , "Guitar\n"
            if intermediaire[3].replace("\n","") in accounted_track_names:
                guitar_tracks.append(int(intermediaire[0]))

    print(guitar_tracks)
                
        



    if guitar_tracks == []:
        print("No guitar tracks found in the MIDI file")
        return

    if pitchbend:
        lfinal = [["0","0","0","0","0","0"]]
    if not pitchbend:
        lfinal = [["0","0","0","0","0"]]

    for i in csv_string:
        intermediaire = i.split(', ')
        # skipping midi events not related to guitar tracks
        if int(intermediaire[0]) in guitar_tracks:
            
            if pitchbend:
                if 'Pitch_bend' in intermediaire[2]:
                    temp_pitchbend_value = str(pitchbend_quantization_encoder(int(intermediaire[4][:-1]))) # quantization of pitchbend values
            if  'Note' in intermediaire[2]:
            ##############là on écrase toutes les tracks, intermediaire[0] et intermediaire[3] ne sont pas considérés
                if not pitchbend:
                    lmidi.append([intermediaire[1],intermediaire[4],intermediaire[5][:-1]]) #  time pitch velocité
                else:
                    lmidi.append([intermediaire[1],intermediaire[4],intermediaire[5][:-1],temp_pitchbend_value]) #  time pitch velocité pitchbend_value


   # précaution : réordonnancement temporel si plusieurs pistes passent 
    lmidi = sorted(lmidi, key=lambda x: int(x[0]))
    
    # rectification vis à vis des notes off
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
    
    with open(output_file, "w") as file:
        file.write(s)

    print("done")




########## MIDI to text conversion :  pitchbend Quantization central IDEA : keeping a lot of values near 8192 (no pitchbend)

def pitchbend_quantization_encoder(pitchbend_value): 
    # si entre 0 et 7680 (8192-512) , on arrondit à 500 près
    # si entre 7680 et 8704 , on arrondit à 64 près
    # si entre 8704 (8192+512) et 16784 , on arrondit à 512 près
    # 15 + 16 + 15 valeurs possibles
    # puis bijection avec [O,46] (avec 0 et 46 données une seule fois pour 0 et 16384 respectivement)
    if pitchbend_value < 7680:
        return pitchbend_value//512

    if pitchbend_value < 8704:
        return 15 + (pitchbend_value-7680)//64
    
    if pitchbend_value < 16784:
        return 31 + (pitchbend_value-8704)//512

    return 46

def pitchbend_quantization_decoder(pitchbend_value):

    if pitchbend_value < 15:
        return pitchbend_value*512
    
    if pitchbend_value < 31:
        return 7680 + (pitchbend_value-15)*64
    
    if pitchbend_value < 47:
        return 8704 + (pitchbend_value-31)*512

    return 16384



