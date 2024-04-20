import py_midicsv as pm
import sys


####################################################################
#  We get s_MIDI quadruplets if Pitchbend is not activated
#  We get s_MIDI quintuplets if Pitchbend is activated
####################################################################


def to_text(input_file, output_file, pitchbend = False):
    csv_string = pm.midi_to_csv(r"{}".format(input_file))
    lmidi = []

    temp_pitchbend_value = '8192' # initial value of pitchbend that has no effect on sound (8192/128 = 64)


    ############### on ne garde que les tracks de guitare
    guitar_tracks = []
    # dictionnary of possible names for lead tracks : 
    possible_lead_names = ['guitar', 'Guitar', 'Lead', 'lead', 'Solo', 'solo', 'Angus Young']
    count=0
    for i in csv_string:
        intermediaire = i.split(', ')

        
    
        if 'Title_t' in intermediaire[2]:
            # check if the track is a guitar track
            for name in possible_lead_names:
                if name in intermediaire[3]:
                    count+=1
            if count>0:
                # get the track number 
                guitar_tracks.append(int(intermediaire[0]))
                                    
            count=0

  

    ###############
    # TO DO : Quantization of pitchbend values ; 16384 values => 128 values


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
                    temp_pitchbend_value = intermediaire[4][:-1] # quantization of pitchbend values
            if  'Note' in intermediaire[2]:
            ##############là on écrase toutes les tracks, intermediaire[0] et intermediaire[3] ne sont pas considérés
                if not pitchbend:
                    lmidi.append([intermediaire[1],intermediaire[4],intermediaire[5][:-1]]) #  time pitch velocité
                else:
                    lmidi.append([intermediaire[1],intermediaire[4],intermediaire[5][:-1],temp_pitchbend_value]) #  time pitch velocité pitchbend_value


   # précaution : réordonnancement temporel
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






