
import gdown  # for importing from GoogleDrive
import os
import zipfile
import data_augmentation
import midi_textefinal
import random
import shutil
###############################################################################################################

# Running this code generates simplified midi files from the Giant Midi Piano dataset in the folder 'generated files/GMP_s_midis' (10854 files)
# It takes ~ 2 min to run on my computer with nb_output = 1000 and augmentation_level =2

#Choose the level of augmentation           -------> augmentation_level (int)
#Choose the number of output files you want -------> nb_output (int)

###############################################################################################################




nb_output = 1000
augmentation_level = 2





###########################################################################

##### get the midi files

url = 'https://drive.google.com/uc?id=1t7r3-SbDWKrbBYah5lOSSb6GPhygMlIL'
output_dir = '../generated files'


#getting rid of any previous generation
if  os.path.exists('../generated files/GMP_midis'):
    shutil.rmtree('../generated files/GMP_midis')

#
output_zip_path = os.path.join(output_dir, 'zip_file')
gdown.download(url, output_zip_path, quiet=False)
# Unzip the downloaded file
with zipfile.ZipFile(output_zip_path, 'r') as zip_ref:
    zip_ref.extractall(output_dir)
os.rename('../generated files/midis', '../generated files/GMP_midis')
# Remove the zip file after extraction 
os.remove(output_zip_path)


###########################################################################


#  Choosing subset of GMP (here simply in alphabetical order)
files = sorted(os.listdir('../generated files/GMP_midis'))
files_to_keep = files[:int(nb_output/augmentation_level)]

for file_name in files:
    if file_name not in files_to_keep:
        file_path = os.path.join('../generated files/GMP_midis', file_name)
        os.remove(file_path)



###### Augmentation

# choosing the augmentation : Here simple transposition (of each piece  at random)

files = os.listdir('../generated files/GMP_midis')
for entree in files:
    # Split the filename and extension
    
    filename, extension = os.path.splitext(entree)
    # Append 'augmented_1' before the extension
    sortie = f"{filename}_augmented_1{extension}" # to be modified if multiple augmentations
    entree_file_path = os.path.join('../generated files/GMP_midis',entree)
    sortie_file_path = os.path.join('../generated files/GMP_midis',sortie)

    # random transpo between -12 and +12 and not 0 ; 
    transpo = random.choice([x for x in range(-12, 0)] + [x for x in range(1, 13)])
    data_augmentation.DA_transposition(entree_file_path,sortie_file_path, transpo)
    




###### Conversion to simplified midi 

#getting rid of any previous generation
if  os.path.exists('../generated files/GMP_s_midis'):
    shutil.rmtree('../generated files/GMP_s_midis')
os.mkdir('../generated files/GMP_s_midis')


files = os.listdir('../generated files/GMP_midis')
for input_file_name in files:
    
    id, extension = os.path.splitext(input_file_name)
    output_file_name = f"{id}.txt"
    input_file_path = os.path.join('../generated files/GMP_midis', input_file_name)
    output_file_path = os.path.join('../generated files/GMP_s_midis', output_file_name)
    midi_textefinal.main(input_file_path,output_file_path)


    


###### Truncature :
    

    # beware to let a space after the last sequence(TO BE DONE)
    
# on garde les 3000 premiers string, ce qui correspond en moyenne à un peu moins de 3000 tokens
max_chars = 3000
    
files = os.listdir('../generated files/GMP_s_midis')
for file_name in files :
    
    file_path = os.path.join('../generated files/GMP_s_midis', file_name)
   
    with open(file_path, 'r') as file:
        content = file.read()

    truncated_content = content[:max_chars]
    # beware of not cutting a motif and leting a space at the end
    last_space_index = truncated_content.rfind(' ')
    if last_space_index != -1:
        truncated_content = truncated_content[:last_space_index]

    with open(file_path, 'w') as file:
        file.write(truncated_content)

