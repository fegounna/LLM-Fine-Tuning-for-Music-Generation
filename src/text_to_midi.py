from copy import deepcopy
import py_midicsv as pm
import os
import shutil


####################################################################
# Requires that the simplified_midi file finishes with a blank " " 
####################################################################
def create_midi_file(path, simplified_midi, header):
    l = simplified_midi.split(' ')
    
    for i in range(len(l)):
        l[i] = l[i].split(':')
    
    for i in range(len(l)):
        for j in range(4):
            l[i][j] = int(l[i][j][1:])
    t_abs = 0
    for i in range (len(l)):
        l[i].append(t_abs)
        t_abs+= l[i][3]
    j=deepcopy(l)
    for a in j:
        l.append([a[0],0,0,0,a[4]+a[2]])
    l = sorted(l, key=lambda x: x[4])
    l.append([0,0,0,0,l[-1][4]])
    # print(l) #ici l contient pitch, velocity, duration, time, t_abs dans cet ordre

    csv_string = deepcopy(header)
    for i in range(len(l)):
        csv_string.append("2, " + str(l[i][4]) + ", Note_on_c, 0, " + str(l[i][0]) + ", " + str(l[i][1]) + "\n")
    csv_string.append("2, " + str(l[len(l)-1][4]) + ", End_track\n")
    csv_string.append("0, 0, End_of_file")

    # with open("./csv_string.txt", "w") as file :
    #     file.write(str(csv_string))

    midi_object = pm.csv_to_midi(csv_string)

    with open(path, "wb") as output_file :
        midi_writer = pm.FileWriter(output_file)
        midi_writer.write(midi_object)
        
    print(header)


#  Can we do otherwise than giving the same header to each file ? Training to predict the header ? 
        
############################################################
# output_dir  will receive the untranscripted midis ; 
# input_dir  is a directory of Simplified Midis
############################################################ 
        
def dataset_text_to_midi(output_dir_path, input_dir_path,header):
    files = os.listdir(input_dir_path)
    for input_file_name in files:
        
        id, extension = os.path.splitext(input_file_name)
        output_file_name =  output_file_name = f"{id}_converted.mid"
        input_file_path = os.path.join(input_dir_path, input_file_name)
        output_file_path = os.path.join(output_dir_path, output_file_name)
        
        with open(input_file_path,"r") as file:
            str = file.read()[:-1] 
            
        create_midi_file(output_file_path , str , header)
        

    

#For testing 
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
