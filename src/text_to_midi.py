import mido
from mido import MidiFile, MidiTrack, Message, MetaMessage

with open("../generated files/GMP_s_midis/A., Jag, Je t'aime Juliette, OXC7Fd0ZN8o_augmented_1.txt", "r") as file:
    s = file.read()[:-1]

print(s)
def create_midi_file(lieu,simplified_midi):
    fichier = MidiFile()

    track0 = MidiTrack()
    trackinit = MidiTrack()
    fichier.tracks.append(trackinit)
    fichier.tracks.append(track0)
    trackinit.append(MetaMessage('set_tempo', tempo=100000, time=0))
    trackinit.append(MetaMessage('time_signature', numerator=4, denominator=4, clocks_per_click=24, notated_32nd_notes_per_beat=8, time=0))
#    track0.append(Message('program_change', channel=0 ,program=0, time=0))
#    track0.append(Message('program_change', channel=0 ,program=0, time=0))
#    track0.append(MetaMessage('instrument_name', name='acoustic grand', time=0))
#    track0.append(Message('control_change' ,channel=0 ,control=7 ,value=100 ,time=0))
#    track0.append(MetaMessage('key_signature', key='C', time=0))
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
    j=list(l)

    for a in j:
        l.append([a[0],0,0,0,a[4]+a[2]])
    l = sorted(l, key=lambda x: x[4])
    l.append([0,0,0,0,l[-1][4]])
    #print(l) 
    for i in range(len(l)-1):
        track0.append(Message('note_on', note=l[i][0], velocity=l[i][1], time=l[i+1][4]-l[i][4]))
    fichier.save(lieu)


lieu = r"../generated files/TEST(1).mid"
create_midi_file(lieu,s)