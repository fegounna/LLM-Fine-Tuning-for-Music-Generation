
import os 
import re
import shutil
import subprocess
import midi_textefinal
import py_midicsv as pm
import midi_textfinal_DadaGP
import text_to_midi



# Download and Unzip DadaGP-v1.1 in /LLM-Fine-Tuning-for-Music-Generation ; 
# Download  GuitarProToMidi https://github.com/rageagainsthepc/GuitarPro-to-Midi/releases in /LLM-Fine-Tuning-for-Music-Generation ; 

#update :
path_GuitarProToMidi = "/Users/charlesbenichouchaffanjon/Documents/LLM-Fine-Tuning-for-Music-Generation/GuitarProToMidi"



#######################################
# 1st step : keeping only the GuitarPro files with some bend and transcribing those to midi files with GuitarProToMidi
#######################################



"""
#output directory
if  os.path.exists('../generated files/DadaGP_metal_midis'):
    shutil.rmtree('../generated files/DadaGP_metal_midis')
os.mkdir("../generated files/DadaGP_metal_midis")

output_directory = "../generated files/DadaGP_metal_midis"
count_errors = 0


#iterate recursively over the GuitarPro files : 
for root, _dirs, files in os.walk("../DadaGP_metal_gp"):
    for file in files:

        #getting the .txt files and know if "bend" appears in it
        if file.endswith(".txt"):
            # get file name :
            
            try:
                with open(os.path.join(root, file), "r") as file:
                    str = file.read()
                    if "bend" in str:
                    
                        # getting the name of the ".gp4 associated file"  from the example.gp4.tokens.txt to the example.gp4
                        gp_filename = re.sub(r'\.tokens\.txt$', '', file.name)
                        ####### do the conversion
                        bash_command = f"{path_GuitarProToMidi} \"{gp_filename}\""
                        subprocess.run(bash_command, shell=True)

                        #move the file to output_directory, collapsing the directory structure, after verifying the file exists
                        # counts errors of GuitarProToMidi too
                        id, extension = os.path.splitext(gp_filename)
                        output_file_name =  f"{id}.mid"

                        # beware of multiple files with the same name and different gp versions
                        if(os.path.exists(output_file_name) and not os.path.exists(os.path.join(output_directory, os.path.basename(output_file_name).rsplit('/', 1)[-1]))):
                            shutil.move(output_file_name, output_directory)
                        # removing the midi_files that are converted twice in different formats
                        if(not os.path.exists(output_file_name)) :
                            count_errors += 1

                        if( os.path.exists(output_file_name) and  os.path.exists(os.path.join(output_directory, os.path.basename(output_file_name).rsplit('/', 1)[-1]))):
                            os.remove(output_file_name)

            except Exception as e:
                print(e)
                count_errors += 1
                
print(count_errors)

"""




                    
                    

#######################################
# 2nd step : keeping only the lead tracks of each piece and transcribing those into simplified MIDI : code in midi_textfinal_DadaGP.py ; 
#######################################

#getting rid of any previous generation


if  os.path.exists('../generated files/DadaGP_metal_s_midis'):
    shutil.rmtree('../generated files/DadaGP_metal_s_midis')
os.mkdir("../generated files/DadaGP_metal_s_midis")

files = os.listdir("../generated files/DadaGP_metal_midis")

for input_file_name in files:
    id, extension = os.path.splitext(input_file_name)
    output_file_name = f"{id}.txt"
    input_file_path = os.path.join("../generated files/DadaGP_metal_midis", input_file_name)
    output_file_path = os.path.join("../generated files/DadaGP_metal_s_midis", output_file_name)
    try:
        midi_textfinal_DadaGP.to_text(input_file_path,output_file_path, pitchbend = True)
    except Exception as e:
        continue
    
    
    
##################################### It gives 2695 unique MIDI files with bends  and then ... 1258 Simplified_MIDI files with bends #####################################
# Les limites de notre code : quand il y a plusieurs guitares, on combine les tracks ( il faudrait changer beaucoup de choses pour les séparer)
    


'''
input_file_name = "AC-DC - Cover You In Oil.txt"
id, extension = os.path.splitext(input_file_name)
output_file_name = f"{id}.mid"
input_file_path = os.path.join("../generated files/DadaGP_metal_s_midis", input_file_name)
output_file_path = os.path.join("../generated files", output_file_name)



'''

# testing  on a single file  : 



'''
text_to_midi.create_midi_file(input_file_path,
                              output_file_path,
                              ["0, 0, Header,1, 2, 480\n", "1, 0, Start_track\n", "1, 0, Tempo, 500000\n", "1, 0, Time_signature, 4, 2, 24, 8\n", "1, 1, End_track\n", "2, 0, Start_track\n"],
                              pitchbend = True)
                               
'''











