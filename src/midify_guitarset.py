import requests
import zipfile
from io import BytesIO
import os
from interpreter import jams_to_midi



###############################################################################################################

# This code generates midi files from the GuitarSet dataset in the folder 'generated files/midi_data_from_jams' 

###############################################################################################################


# get the jams_file from GuitarSet
jams_data_url = 'https://zenodo.org/records/3371780/files/annotation.zip?download=1'
r = requests.get(jams_data_url)

#remove the folder of dataset if it already exists to be sure it's up to date and that we don't have multiple copies of each file
if not os.path.exists('../generated files/jams_data'):
    os.mkdir('../generated files/jams_data')


with zipfile.ZipFile(BytesIO(r.content), 'r') as zip_ref:
    zip_ref.extractall('../generated files/jams_data')


#keeping only the solo files
print('removing files that are not solos')
# Get list of files in the directory
files = os.listdir('../generated files/jams_data')
for file in files:
    # Check if the file name contains the sequence "solo"
    if "solo" not in file:
        # Construct the full file path
        file_path = os.path.join('../generated files/jams_data', file)
        # Remove the file
        os.remove(file_path)
print('done')



# the transcription to midi : this uses jams_to_midi from the GuitarSet git ; 
#remove the folder of dataset if it already exists to be sure it's up to date and that we don't have multiple copies of each file
   
if not os.path.exists('../generated files/midi_data_from_jams'):
    os.mkdir('../generated files/midi_data_from_jams')

files = os.listdir('../generated files/jams_data')
jams_files = [f for f in files]

# Convert each JAMS file to MIDI
for jams_file in jams_files:
    # Construct paths for input JAMS file and output MIDI file
    jams_path = os.path.join('../generated files/jams_data', jams_file)
    midi_file_name = os.path.splitext(jams_file)[0] + '.mid'
    midi_file_path = os.path.join('../generated files/midi_data_from_jams', midi_file_name)

    jams_to_midi(jams_path,title=os.path.splitext(jams_file)[0] + '.mid')  # Convert JAMS to MIDI
    #reorganize in right folder
    os.replace(midi_file_name, midi_file_path)  

print("Conversion completed successfully.")


# removing the solo jams data : 
print('removing  jams solos.... ')
files = os.listdir('../generated files/jams_data')
for file in files:
    file_path = os.path.join('../generated files/jams_data', file)
    # Remove the file
    os.remove(file_path)

os.rmdir('../generated files/jams_data')
print('done')   
