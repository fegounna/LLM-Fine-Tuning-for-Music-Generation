import mido
from mido import MidiFile, MidiTrack, Message, MetaMessage
s= 'p36:v95:d1711:t1 p48:v96:d1695:t88 p55:v63:d1606:t48 p60:v67:d1389:t47 p64:v66:d1510:t66 p60:v79:d2:t53 p67:v82:d1044:t53 p72:v86:d832:t52 p76:v70:d1303:t49 p72:v90:d2:t59 p79:v84:d611:t48 p84:v83:d1129:t106 p84:v82:d1:t56 p91:v90:d163:t48 p96:v93:d919:t36 p100:v105:d887:t83 p91:v94:d810:t73 p84:v74:d1:t60 p88:v98:d668:t53 p84:v77:d2:t51 p79:v78:d573:t60 p72:v84:d504:t67 p76:v88:d0:t48 p72:v64:d2:t44 p67:v81:d345:t67 p60:v75:d3:t69 p64:v90:d0:t43 p60:v57:d1052:t45 p55:v83:d2:t57 p48:v80:d4:t69 p60:v94:d4:t156 p60:v75:d1:t49 p53:v72:d676:t50 p60:v75:d4:t83 p69:v79:d543:t51 p72:v90:d491:t28 p60:v55:d2:t31 p77:v83:d429:t52 p72:v90:d6:t73 p81:v82:d307:t41 p84:v90:d266:t66 p89:v81:d197:t39 p84:v85:d3:t0'
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
    tabs = 0
    for i in range (len(l)):
        l[i].append(tabs)
        tabs+= l[i][3]
    j=list(l)

    for a in j:
        l.append([a[0],0,0,0,a[4]+a[2]])

    for i in range (len(l)-1,0,-1):
        for j in range(i):
            if l[j][4] > l[j+1][4]:
                l[j],l[j+1]= l[j+1],l[j]
    l.append([0,0,0,0])
    print(l)
    for i in range(len(l)-2):
        track0.append(Message('note_on', note=l[i][0], velocity=l[i][1], time=l[i+1][4]-l[i][4]))
    fichier.save(lieu)

lieu = r"C:\Users\cypri\Desktop\python PSC\TEST.mid"
create_midi_file(lieu,s)