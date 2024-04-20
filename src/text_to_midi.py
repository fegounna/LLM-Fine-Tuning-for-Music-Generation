from copy import deepcopy
import py_midicsv as pm
import os
import shutil

import midi_textefinal # to test


####################################################################
# Requires that the simplified_midi file finishes with a blank " " 
####################################################################

def create_midi_file(input_file_path,output_file_path, header, pitchbend = False):

    with open(input_file_path,"r") as file:
            simplified_midi = file.read()[:-1] 

    l = simplified_midi.split(' ')
    
    
    for i in range(len(l)):
        l[i] = l[i].split(':')
    
    for i in range(len(l)):

        if pitchbend:
            for j in range(5):
                l[i][j] = int(l[i][j][1:])
        if not pitchbend:
            for j in range(4):
                l[i][j] = int(l[i][j][1:])

    t_abs = 0
    for i in range (len(l)):
        l[i].append(t_abs)
        t_abs+= l[i][3]
    j=deepcopy(l)
    if not pitchbend:
        for a in j:
            l.append([a[0],0,0,0,a[4]+a[2]])
        l = sorted(l, key=lambda x: x[4])
        l.append([0,0,0,0,l[-1][4]])

    if pitchbend:
        for a in j:
            l.append([a[0],0,0,0,0,a[5]+a[2]])
        l = sorted(l, key=lambda x: x[5])
        l.append([0,0,0,0,0,l[-1][5]])
         

    csv_string = deepcopy(header)

    for i in range(len(l)):

        if pitchbend:
            #verifying it is not a Note Off
            if l[i][1]!=0:
                # because of the quantization of pitchbend values we multiply by 128
                csv_string.append("2, " + str(l[i][5]) + ", Pitch_bend_c, 0, "  + str(l[i][4]) + "\n") 
                
            csv_string.append("2, " + str(l[i][5]) + ", Note_on_c, 0, " + str(l[i][0]) + ", " + str(l[i][1]) + "\n")

        if not pitchbend:
            csv_string.append("2, " + str(l[i][4]) + ", Note_on_c, 0, " + str(l[i][0]) + ", " + str(l[i][1]) + "\n")


    if pitchbend:
        csv_string.append("2, " + str(l[len(l)-1][5]) + ", End_track\n")
    if not pitchbend:
        csv_string.append("2, " + str(l[len(l)-1][4]) + ", End_track\n")

    csv_string.append("0, 0, End_of_file")
    # with open("./csv_string.txt", "w") as file :
    #     file.write(str(csv_string))

    midi_object = pm.csv_to_midi(csv_string)

    with open(output_file_path, "wb") as output_file :
        midi_writer = pm.FileWriter(output_file)
        midi_writer.write(midi_object)

    



#  Can we do otherwise than giving the same header to each file ? Training to predict the header ? 
        
############################################################
# output_dir  will receive the untranscripted midis ; 
# input_dir  is a directory of Simplified Midis
############################################################ 
        
def dataset_text_to_midi(output_dir_path, input_dir_path,header):
    files = os.listdir(input_dir_path)
    for input_file_name in files:
        
        id, _ = os.path.splitext(input_file_name)
        output_file_name =  output_file_name = f"{id}_converted.mid"
        input_file_path = os.path.join(input_dir_path, input_file_name)
        output_file_path = os.path.join(output_dir_path, output_file_name)
        create_midi_file(input_file_path, output_file_path , str , header)
        




'''
# for testing dataset_text_to_midi
if  os.path.exists('../generated files/GMP_midis_converted'):
    shutil.rmtree('../generated files/GMP_midis_converted')
os.mkdir('../generated files/GMP_midis_converted')

output_dir_path = "../generated files/GMP_midis_converted"
input_dir_path = "../generated files/GMP_s_midis"
# All files in GMP use the same header


print("beginning untranscription...")
dataset_text_to_midi(output_dir_path, input_dir_path, header)
print("Done")

'''



'''
# for testing create_midi_file

midi_textefinal.main("../generated files/midi_data_from_jams/00_BN2-166-Ab_solo.mid","../generated files/00_BN2-166-Ab_solo_PB.txt", pitchbend= True)

input_file_path = "../generated files/00_BN2-166-Ab_solo_PB.txt"
with open(input_file_path,"r") as file:
            s = file.read()[:-1] 
create_midi_file("../generated files/00_BN2-166-Ab_solo_PB.mid",
                s,
                header = ["0, 0, Header,1, 2, 384\n", "1, 0, Start_track\n", "1, 0, Tempo, 500000\n", "1, 0, Time_signature, 4, 2, 24, 8\n", "1, 1, End_track\n", "2, 0, Start_track\n"],
                pitchbend = True)

#midi_textefinal.texte("../generated files/00_BN1-129-Eb_solo_converted_PB.mid",pitchbend=True)
print('\n\n')
#midi_textefinal.texte("../generated files/midi_data_from_jams/00_BN1-129-Eb_solo.mid", pitchbend=True)


'''

