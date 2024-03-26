import mido
import os
import pretty_midi

def DA_modulation(entree, sortie, modulation, debut, fin):
 res = pretty_midi.PrettyMIDI(entree)
 #On traduit début et fin pour les utiliser dans le fichier MIDI
 start = int(res.time_to_tick(debut*res.get_end_time()))
 end = int(res.time_to_tick(fin*res.get_end_time()))
 #Modulation
 for instrument in res.instruments:
    for note in instrument.notes:
        if start <= note.start < end:
            note.pitch += modulation
 #On remplit le fichier de sortie avec notre modulation
 res.write(sortie)

def DA_transposition(entree,sortie,transpo):
 res = pretty_midi.PrettyMIDI(entree)
 #transposition
 for instrument in res.instruments:
    for note in instrument.notes:
        note.pitch += transpo
 res.write(sortie)

def DA_Tempo(entree,sortie,facteur):
 res = pretty_midi.PrettyMIDI(entree)
 #on modifie le tempo de l'ensemble
 for tempo in res.get_tempo_changes():
    tempo[0] *= facteur
 res.write(sortie)
 
 