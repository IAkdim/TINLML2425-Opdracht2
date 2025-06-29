"""
Melody Maker - Phase 1 Implementation
Genetic Algorithm for Musical Composition

Phase 1 Features:
- Building blocks (predefined musical clichés)
- Manual fitness scoring (0-100)
- Tournament selection
- Single-point crossover
- Pitch and duration mutation
- Population management with elitism
"""

import random
import copy
from music.muser import Muser

class MelodyMaker:
    def __init__(self, population_size=20, mutation_rate=0.1, crossover_rate=0.7):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.population = []
        self.generation = 0
        self.muser = Muser()
        
        # Define building blocks - common musical clichés
        self.building_blocks = {
            'intro_major': [('c', 4), ('e', 4), ('g', 4), ('c*', 4)],
            'intro_minor': [('a3', 4), ('c', 4), ('e', 4), ('a', 4)],
            'verse_simple': [('c', 8), ('d', 8), ('e', 4), ('f', 4), ('g', 4)],
            'verse_complex': [('c', 8), ('e', 8), ('g', 8), ('f', 8), ('e', 4), ('d', 4)],
            'chorus_uplifting': [('g', 4), ('a', 4), ('b', 4), ('c*', 2)],
            'chorus_powerful': [('c*', 4), ('b', 4), ('a', 4), ('g', 4), ('f', 4), ('e', 4)],
            'bridge_calm': [('f', 8), ('g', 8), ('a', 4), ('g', 4), ('f', 4)],
            'bridge_dramatic': [('d', 4), ('f', 4), ('a', 4), ('d*', 2)],
            'outro_resolve': [('g', 4), ('f', 4), ('e', 4), ('c', 2)],
            'outro_fade': [('c', 8), ('e', 8), ('g', 8), ('c*', 8), ('g', 4), ('c', -2)],
            'bass_walk': [('c2', 4), ('d2', 4), ('e2', 4), ('f2', 4)],
            'bass_steady': [('c2', 2), ('g2', 2), ('c2', 2), ('g2', 2)],
            'rhythm_simple': [('c', 8), ('c', 8), ('c', 4), ('c', 4)],
            'rhythm_syncopated': [('c', 8), ('c', 16), ('c', 16), ('c', 8), ('c', 4)]
        }
        
        self.notes = ['c', 'd', 'e', 'f', 'g', 'a', 'b']
        self.octaves = ['2', '3', '', '*']  # 2=low, 3=mid-low, ''=mid, *=high
        
    def create_individual(self):
        """Create a random individual (composition) from building blocks"""
        num_blocks = random.randint(4, 8)
        melody_track = []
        bass_track = []
        
        for _ in range(num_blocks):
            block_name = random.choice(list(self.building_blocks.keys()))
            block = copy.deepcopy(self.building_blocks[block_name])
            
            if 'bass' in block_name:
                bass_track.extend(block)
            else:
                melody_track.extend(block)
        
        if not melody_track:
            melody_track = copy.deepcopy(self.building_blocks['verse_simple'])
        if not bass_track:
            bass_track = copy.deepcopy(self.building_blocks['bass_steady'])
            
        return (melody_track, bass_track)
    
    def initialize_population(self):
        """Initialize the population with random individuals"""
        self.population = []
        for _ in range(self.population_size):
            individual = self.create_individual()
            self.population.append(individual)
    
    def fitness_function(self, individual):
        """Manual fitness scoring - you rate each composition"""
        print(f"\n--- Evaluating Composition ---")
        print(f"Melody track length: {len(individual[0])} notes")
        print(f"Bass track length: {len(individual[1])} notes")
        
        print("Melody preview:", individual[0][:5], "..." if len(individual[0]) > 5 else "")
        print("Bass preview:", individual[1][:5], "..." if len(individual[1]) > 5 else "")
        
        temp_filename = None
        try:
            temp_filename = f"temp_composition_{random.randint(1000,9999)}.wav"
            self.generate_music(individual, temp_filename)
            print(f"Generated audio: {temp_filename}")
            
            self.play_with_repeat_option(temp_filename)
            
        except Exception as e:
            print(f"Could not generate audio: {e}")
        
        # Manual rating
        while True:
            try:
                fitness_score = float(input("Rate this composition (0-100): "))
                if 0 <= fitness_score <= 100:
                    break
                print("Please enter a score between 0 and 100")
            except ValueError:
                print("Please enter a valid number")
        
        if temp_filename:
            self.cleanup_temp_file(temp_filename)
        
        print(f"Fitness: {fitness_score}/100")
        return fitness_score
    
    def tournament_selection(self, fitness_cache, tournament_size=3):
        """Tournament selection to choose parents using cached fitness scores"""
        tournament = random.sample(list(range(len(self.population))), min(tournament_size, len(self.population)))
        best_idx = max(tournament, key=lambda i: fitness_cache[i])
        return self.population[best_idx]
    
    def crossover(self, parent1, parent2):
        """Single-point crossover between two compositions"""
        if random.random() > self.crossover_rate:
            return parent1, parent2
            
        # Crossover melody tracks
        if len(parent1[0]) > 1 and len(parent2[0]) > 1:
            point1 = random.randint(1, len(parent1[0]) - 1)
            point2 = random.randint(1, len(parent2[0]) - 1)
            
            child1_melody = parent1[0][:point1] + parent2[0][point2:]
            child2_melody = parent2[0][:point2] + parent1[0][point1:]
        else:
            child1_melody, child2_melody = parent1[0], parent2[0]
        
        # Crossover bass tracks
        if len(parent1[1]) > 1 and len(parent2[1]) > 1:
            point1 = random.randint(1, len(parent1[1]) - 1)
            point2 = random.randint(1, len(parent2[1]) - 1)
            
            child1_bass = parent1[1][:point1] + parent2[1][point2:]
            child2_bass = parent2[1][:point2] + parent1[1][point1:]
        else:
            child1_bass, child2_bass = parent1[1], parent2[1]
            
        return (child1_melody, child1_bass), (child2_melody, child2_bass)
    
    def mutate(self, individual):
        """Mutate an individual by changing notes or durations"""
        individual = copy.deepcopy(individual)
        
        # Mutate melody track
        for i in range(len(individual[0])):
            if random.random() < self.mutation_rate:
                if random.random() < 0.5:  # Pitch mutation
                    base_note = random.choice(self.notes)
                    octave = random.choice(self.octaves)
                    new_note = base_note + octave
                    individual[0][i] = (new_note, individual[0][i][1])
                else:  # Duration mutation
                    new_duration = random.choice([2, 4, 8, 16])
                    individual[0][i] = (individual[0][i][0], new_duration)
        
        # Mutate bass track
        for i in range(len(individual[1])):
            if random.random() < self.mutation_rate:
                if random.random() < 0.5:  # Pitch mutation
                    base_note = random.choice(self.notes)
                    octave = random.choice(['2', '3'])  # Keep bass in lower octaves
                    new_note = base_note + octave
                    individual[1][i] = (new_note, individual[1][i][1])
                else:  # Duration mutation
                    new_duration = random.choice([2, 4, 8])
                    individual[1][i] = (individual[1][i][0], new_duration)
                    
        return individual
    
    def run_generation(self):
        """Run one generation of the genetic algorithm"""
        print(f"\n=== Generation {self.generation} ===")
        
        # Evaluate fitness for all individuals if not already done
        fitness_scores = []
        for i, individual in enumerate(self.population):
            print(f"\nIndividual {i+1}/{len(self.population)}")
            fitness = self.fitness_function(individual)
            fitness_scores.append((individual, fitness))
        
        # Sort by fitness
        fitness_scores.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\nGeneration {self.generation} Results:")
        for i, (_, fitness) in enumerate(fitness_scores[:3]):
            print(f"  Top {i+1}: Fitness = {fitness:.1f}")
        
        # Create next generation
        new_population = []
        
        # Keep best individuals
        elite_count = max(1, self.population_size // 10)
        for i in range(elite_count):
            new_population.append(fitness_scores[i][0])
        
        # Create fitness cache for efficient selection
        fitness_cache = [fitness for _, fitness in fitness_scores]
        
        # Generate rest through selection, crossover, mutation
        while len(new_population) < self.population_size:
            parent1 = self.tournament_selection(fitness_cache)
            parent2 = self.tournament_selection(fitness_cache)
            
            child1, child2 = self.crossover(parent1, parent2)
            
            child1 = self.mutate(child1)
            child2 = self.mutate(child2)
            
            new_population.extend([child1, child2])
        
        # Trim to exact population size
        self.population = new_population[:self.population_size]
        self.generation += 1
        
        return fitness_scores[0]  # Return best individual
    
    def generate_music(self, individual, filename="generation_best.wav"):
        """Generate audio file from an individual"""
        try:
            # Temporarily change to music directory for file generation
            import os
            original_dir = os.getcwd()
            os.chdir('music')
            
            self.muser.generate(individual)
            
            # Move the generated file
            import shutil
            shutil.move('song.wav', f'../{filename}')
            
            os.chdir(original_dir)
            print(f"Generated audio: {filename}")
            
        except Exception as e:
            print(f"Error generating music: {e}")
    
    def play_audio(self, filename):
        """Attempt to play audio file automatically"""
        import subprocess
        import os
        
        if not os.path.exists(filename):
            print(f"Audio file {filename} not found")
            return
            
        print(f"Playing {filename}...")
        
        # Try different audio players based on the system
        players = [
            ['aplay', filename],  # ALSA player (Linux)
            ['paplay', filename],  # PulseAudio player (Linux)
            ['ffplay', '-nodisp', '-autoexit', filename],  # ffmpeg player
            ['vlc', '--play-and-exit', '--intf', 'dummy', filename],  # VLC
            ['mpv', '--no-video', filename],  # mpv
            ['play', filename],  # SoX player
        ]
        
        for player_cmd in players:
            try:
                # Try to run the player
                result = subprocess.run(player_cmd, 
                                      capture_output=True, 
                                      timeout=30,
                                      check=False)
                if result.returncode == 0:
                    print(f"✓ Played with {player_cmd[0]}")
                    return
            except (subprocess.TimeoutExpired, FileNotFoundError):
                continue
        
        # If no player worked, give manual instructions
        print("Could not autoplay. Try manually:")
        print(f"  aplay {filename}")
        print(f"  vlc {filename}")
        print(f"  Or open in file manager")
    
    def play_with_repeat_option(self, filename):
        """Play audio with option to replay or save before continuing"""
        # Initial play
        self.play_audio(filename)
        
        # Allow replay and save options
        while True:
            choice = input("Options: (p)lay again, (s)ave this composition, (c)ontinue: ").lower().strip()
            if choice in ['p', 'play']:
                self.play_audio(filename)
            elif choice in ['s', 'save']:
                self.save_composition(filename)
                break
            elif choice in ['c', 'continue', 'n', 'no', '']:
                break
            else:
                print("Please enter 'p' to play again, 's' to save, or 'c' to continue")
    
    def cleanup_temp_file(self, filename):
        """Delete temporary audio file"""
        import os
        try:
            if os.path.exists(filename):
                os.remove(filename)
                print(f"Cleaned up {filename}")
        except Exception as e:
            print(f"Could not delete {filename}: {e}")
    
    def save_composition(self, temp_filename):
        """Save a composition with a user-chosen name"""
        import shutil
        import os
        from datetime import datetime
        
        if not os.path.exists(temp_filename):
            print(f"Error: {temp_filename} not found")
            return
            
        # Get save name from user
        default_name = f"saved_composition_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        save_name = input(f"Save as (default: {default_name}): ").strip()
        
        if not save_name:
            save_name = default_name
        
        # Ensure .wav extension
        if not save_name.endswith('.wav'):
            save_name += '.wav'
            
        try:
            shutil.copy2(temp_filename, save_name)
            print(f"✓ Composition saved as: {save_name}")
        except Exception as e:
            print(f"Error saving composition: {e}")


if __name__ == "__main__":
    print("=== Melody Maker - Genetic Algorithm Phase 1 ===")
    print("Phase 1: Manual fitness scoring with building blocks")
    print("This GA creates musical compositions using building blocks and evolution.")
    print("You'll be asked to manually rate each composition (0-100).\n")
    
    # GA Parameters (tunable)
    population_size = int(input("Population size (default 10): ") or "10")
    mutation_rate = float(input("Mutation rate 0.0-1.0 (default 0.15): ") or "0.15")
    crossover_rate = float(input("Crossover rate 0.0-1.0 (default 0.7): ") or "0.7")
    generations = int(input("Number of generations (default 5): ") or "5")
    
    print(f"\nStarting GA with parameters:")
    print(f"  Population: {population_size}")
    print(f"  Mutation rate: {mutation_rate}")
    print(f"  Crossover rate: {crossover_rate}")
    print(f"  Generations: {generations}")
    
    ga = MelodyMaker(population_size, mutation_rate, crossover_rate)
    
    print("\nInitializing population...")
    ga.initialize_population()
    
    best_overall = None
    best_fitness = -1
    
    for gen in range(generations):
        try:
            best_individual = ga.run_generation()
            
            if best_individual[1] > best_fitness:
                best_fitness = best_individual[1]
                best_overall = best_individual[0]
                
                filename = f"generation_{gen}_best.wav"
                ga.generate_music(best_overall, filename)
                print(f"New best composition saved as: {filename}")
            
            print(f"Best fitness so far: {best_fitness:.1f}")
            
            if gen < generations - 1:
                continue_choice = input(f"\nContinue to generation {gen + 2}? (y/n): ").lower()
                if continue_choice != 'y' and continue_choice != 'yes':
                    break
                    
        except KeyboardInterrupt:
            print("\nGA interrupted by user.")
            break
        except Exception as e:
            print(f"Error in generation {gen}: {e}")
            continue
    
    print(f"\n=== Final Results ===")
    print(f"Completed {ga.generation} generations")
    print(f"Best fitness achieved: {best_fitness:.1f}")
    
    if best_overall:
        print("\nGenerating final best composition...")
        ga.generate_music(best_overall, "final_best.wav")
        print("Final composition saved as: final_best.wav")
        
        print(f"\nBest composition structure:")
        print(f"  Melody track: {len(best_overall[0])} notes")
        print(f"  Bass track: {len(best_overall[1])} notes")
    
    print("\nGA completed successfully!")
