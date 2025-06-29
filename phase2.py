"""
Genetic Algorithm - Phase 2 Implementation
Voor TINLML03 - Heuristieken & Geavanceerde operators

Fase 2 Features:
- Geautomatiseerde fitness met heuristieken
- Lengte, akkoordprogressies, maatvoering
- Verschillende recombinatie strategieën
- Pitch vs ritme mutatie
- Populatie diversiteit analyse
"""

import random
import copy
from muser import Muser

class AdvancedGA:
    def __init__(self, population_size=20, mutation_rate=0.12, crossover_rate=0.8, 
                 crossover_type='uniform', mutation_strategy='both'):
        """
        Geavanceerde GA met heuristieken
        
        Parameters:
        - crossover_type: 'single' of 'uniform'
        - mutation_strategy: 'pitch', 'rhythm', of 'both'
        """
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.crossover_type = crossover_type
        self.mutation_strategy = mutation_strategy
        self.population = []
        self.generation = 0
        self.muser = Muser()
        self.diversity_history = []
        
        # Building blocks
        self.building_blocks = {
            'intro': [('c', 4), ('e', 4), ('g', 4), ('c*', 4)],
            'verse': [('c', 8), ('d', 8), ('e', 4), ('f', 4), ('g', 4)],
            'chorus': [('g', 4), ('a', 4), ('b', 4), ('c*', 2)],
            'bridge': [('f', 8), ('g', 8), ('a', 4), ('g', 4), ('f', 4)],
            'outro': [('g', 4), ('f', 4), ('e', 4), ('c', 2)],
            'bass_steady': [('c2', 2), ('g2', 2), ('c2', 2), ('g2', 2)],
            'bass_walk': [('c2', 4), ('d2', 4), ('e2', 4), ('f2', 4)]
        }
        
        self.notes = ['c', 'd', 'e', 'f', 'g', 'a', 'b']
        self.note_to_num = {'c': 0, 'd': 1, 'e': 2, 'f': 3, 'g': 4, 'a': 5, 'b': 6}
        
        # Bekende akkoordprogressies (schaalgraden)
        self.chord_progressions = [
            [0, 3, 4, 0],     # I-IV-V-I
            [1, 4, 0],        # ii-V-I
            [5, 3, 1, 4, 0],  # vi-IV-ii-V-I
        ]

    def create_individual(self):
        """Maak individu met betere muzikale structuur"""
        num_blocks = random.randint(4, 6)
        melody = []
        bass = []
        
        for _ in range(num_blocks):
            block_name = random.choice(list(self.building_blocks.keys()))
            block = copy.deepcopy(self.building_blocks[block_name])
            
            if 'bass' in block_name:
                bass.extend(block)
            else:
                melody.extend(block)
        
        if not melody:
            melody = copy.deepcopy(self.building_blocks['verse'])
        if not bass:
            bass = copy.deepcopy(self.building_blocks['bass_steady'])
            
        return (melody, bass)

    def initialize_population(self):
        """Initialiseer populatie"""
        self.population = []
        for _ in range(self.population_size):
            individual = self.create_individual()
            self.population.append(individual)

    def automated_fitness(self, individual):
        """
        Geautomatiseerde fitness met muzikale heuristieken
        Totaal: 100 punten
        """
        melody, bass = individual
        fitness = 0
        
        # 1. Lengte en maatvoering (0-25 punten)
        fitness += self.evaluate_length_structure(melody)
        
        # 2. Akkoordprogressies (0-25 punten)
        fitness += self.evaluate_chord_progressions(bass)
        
        # 3. Melodische beweging (0-20 punten)
        fitness += self.evaluate_melodic_movement(melody)
        
        # 4. Ritme diversiteit (0-15 punten)
        fitness += self.evaluate_rhythm_diversity(melody)
        
        # 5. Muzikale afsluiting (0-15 punten)
        fitness += self.evaluate_ending(melody, bass)
        
        return min(fitness, 100)

    def evaluate_length_structure(self, melody):
        """Evalueer lengte en maatstructuur (4/4 maat)"""
        score = 0
        total_beats = sum(abs(note[1]) for note in melody)
        
        # Voorkeur voor 8 of 16 maten (32 of 64 beats in 4/4)
        if 30 <= total_beats <= 34:      # ~8 maten
            score += 20
        elif 62 <= total_beats <= 66:    # ~16 maten
            score += 25
        elif 14 <= total_beats <= 18:    # ~4 maten
            score += 15
        else:
            score += max(0, 15 - abs(total_beats - 32) // 4)
        
        # Bonus voor correcte maatindeling
        if total_beats % 4 == 0:
            score += 5
            
        return min(score, 25)

    def evaluate_chord_progressions(self, bass):
        """Analyseer akkoordprogressies in baslijn"""
        score = 0
        
        # Extract root noten
        bass_roots = []
        for note, duration in bass:
            if duration > 0:
                clean_note = note.replace('2', '').replace('3', '')
                if clean_note in self.note_to_num:
                    bass_roots.append(self.note_to_num[clean_note])
        
        if len(bass_roots) < 3:
            return 0
        
        # Zoek naar bekende progressies
        for progression in self.chord_progressions:
            for i in range(len(bass_roots) - len(progression) + 1):
                segment = bass_roots[i:i+len(progression)]
                if len(segment) == len(progression):
                    # Normaliseer naar schaalgraden
                    normalized = [(note - segment[0]) % 7 for note in segment]
                    if normalized == progression:
                        score += 10
        
        # Extra bonus voor ii-V-I progressie
        for i in range(len(bass_roots) - 2):
            progression = [(bass_roots[i+j] - bass_roots[i]) % 7 for j in range(3)]
            if progression == [1, 4, 0]:  # ii-V-I
                score += 15
                
        return min(score, 25)

    def evaluate_melodic_movement(self, melody):
        """Evalueer melodische beweging"""
        score = 0
        
        if len(melody) < 2:
            return 0
        
        # Converteer naar pitch nummers
        pitches = []
        for note, duration in melody:
            if duration > 0:
                clean_note = note.replace('*', '').replace('2', '').replace('3', '')
                octave_bonus = 0
                if '*' in note:
                    octave_bonus = 7
                elif '3' in note:
                    octave_bonus = -7
                elif '2' in note:
                    octave_bonus = -14
                
                if clean_note in self.note_to_num:
                    pitches.append(self.note_to_num[clean_note] + octave_bonus)
        
        if len(pitches) < 2:
            return 0
        
        # Analyseer intervallen
        intervals = [abs(pitches[i+1] - pitches[i]) for i in range(len(pitches)-1)]
        
        # Beloon stapsgewijze beweging met enkele sprongen
        stepwise_count = sum(1 for interval in intervals if interval <= 2)
        leap_count = sum(1 for interval in intervals if 3 <= interval <= 5)
        large_leap_count = sum(1 for interval in intervals if interval > 5)
        
        score += min(12, stepwise_count * 2)    # Max 12 punten voor stappen
        score += min(5, leap_count)             # Max 5 punten voor sprongen
        score -= large_leap_count * 2           # Straf voor grote sprongen
        
        return max(0, min(score, 20))

    def evaluate_rhythm_diversity(self, melody):
        """Evalueer ritmische diversiteit"""
        score = 0
        
        durations = [abs(duration) for _, duration in melody if duration != 0]
        unique_durations = set(durations)
        
        # Beloon ritmische variatie
        score += min(10, len(unique_durations) * 3)
        
        # Bonus voor veelvoorkomende nootwaarden
        common_durations = {2, 4, 8, 16}
        common_count = sum(1 for d in durations if d in common_durations)
        score += min(5, common_count)
        
        return min(score, 15)

    def evaluate_ending(self, melody, bass):
        """Evalueer muzikale afsluiting"""
        score = 0
        
        # Controleer of het eindigt op tonica (C)
        if melody:
            last_note = melody[-1][0].replace('*', '').replace('2', '').replace('3', '')
            if last_note == 'c':
                score += 8
        
        if bass:
            last_bass = bass[-1][0].replace('*', '').replace('2', '').replace('3', '')
            if last_bass == 'c':
                score += 7
        
        return score

    def tournament_selection(self, fitness_scores):
        """Tournament selectie met fitness cache"""
        tournament_size = 3
        tournament = random.sample(range(len(self.population)), min(tournament_size, len(self.population)))
        best_idx = max(tournament, key=lambda i: fitness_scores[i])
        return self.population[best_idx]

    def uniform_crossover(self, parent1, parent2):
        """Uniform crossover - elke positie 50% kans van elke ouder"""
        if random.random() > self.crossover_rate:
            return parent1, parent2
        
        # Crossover melodie
        min_len = min(len(parent1[0]), len(parent2[0]))
        child1_melody = []
        child2_melody = []
        
        for i in range(min_len):
            if random.random() < 0.5:
                child1_melody.append(parent1[0][i])
                child2_melody.append(parent2[0][i])
            else:
                child1_melody.append(parent2[0][i])
                child2_melody.append(parent1[0][i])
        
        # Voeg resterende noten toe
        if len(parent1[0]) > min_len:
            child1_melody.extend(parent1[0][min_len:])
        if len(parent2[0]) > min_len:
            child2_melody.extend(parent2[0][min_len:])
        
        # Hetzelfde voor bas
        min_bass_len = min(len(parent1[1]), len(parent2[1]))
        child1_bass = []
        child2_bass = []
        
        for i in range(min_bass_len):
            if random.random() < 0.5:
                child1_bass.append(parent1[1][i])
                child2_bass.append(parent2[1][i])
            else:
                child1_bass.append(parent2[1][i])
                child2_bass.append(parent1[1][i])
        
        if len(parent1[1]) > min_bass_len:
            child1_bass.extend(parent1[1][min_bass_len:])
        if len(parent2[1]) > min_bass_len:
            child2_bass.extend(parent2[1][min_bass_len:])
        
        return (child1_melody, child1_bass), (child2_melody, child2_bass)

    def single_point_crossover(self, parent1, parent2):
        """Single-point crossover"""
        if random.random() > self.crossover_rate:
            return parent1, parent2
        
        # Crossover melodie
        if len(parent1[0]) > 1 and len(parent2[0]) > 1:
            point = random.randint(1, min(len(parent1[0]), len(parent2[0])) - 1)
            child1_melody = parent1[0][:point] + parent2[0][point:]
            child2_melody = parent2[0][:point] + parent1[0][point:]
        else:
            child1_melody, child2_melody = parent1[0], parent2[0]
        
        # Crossover bas
        if len(parent1[1]) > 1 and len(parent2[1]) > 1:
            point = random.randint(1, min(len(parent1[1]), len(parent2[1])) - 1)
            child1_bass = parent1[1][:point] + parent2[1][point:]
            child2_bass = parent2[1][:point] + parent1[1][point:]
        else:
            child1_bass, child2_bass = parent1[1], parent2[1]
        
        return (child1_melody, child1_bass), (child2_melody, child2_bass)

    def crossover(self, parent1, parent2):
        """Kies crossover methode"""
        if self.crossover_type == 'uniform':
            return self.uniform_crossover(parent1, parent2)
        else:
            return self.single_point_crossover(parent1, parent2)

    def pitch_mutation(self, individual):
        """Gespecialiseerde pitch mutatie"""
        individual = copy.deepcopy(individual)
        octaves = ['2', '3', '', '*']
        
        # Muteer melodie pitches
        for i in range(len(individual[0])):
            if random.random() < self.mutation_rate:
                new_note = random.choice(self.notes) + random.choice(octaves)
                individual[0][i] = (new_note, individual[0][i][1])
        
        # Muteer bas pitches (lagere octaven)
        for i in range(len(individual[1])):
            if random.random() < self.mutation_rate:
                new_note = random.choice(self.notes) + random.choice(['2', '3'])
                individual[1][i] = (new_note, individual[1][i][1])
        
        return individual

    def rhythm_mutation(self, individual):
        """Gespecialiseerde ritme mutatie"""
        individual = copy.deepcopy(individual)
        
        # Muteer melodie ritmes
        for i in range(len(individual[0])):
            if random.random() < self.mutation_rate:
                new_duration = random.choice([2, 4, 8, 16])
                individual[0][i] = (individual[0][i][0], new_duration)
        
        # Muteer bas ritmes
        for i in range(len(individual[1])):
            if random.random() < self.mutation_rate:
                new_duration = random.choice([2, 4, 8])
                individual[1][i] = (individual[1][i][0], new_duration)
        
        return individual

    def mutate(self, individual):
        """Geavanceerde mutatie met strategie selectie"""
        if self.mutation_strategy == 'pitch':
            return self.pitch_mutation(individual)
        elif self.mutation_strategy == 'rhythm':
            return self.rhythm_mutation(individual)
        else:  # 'both'
            individual = self.pitch_mutation(individual)
            return self.rhythm_mutation(individual)

    def calculate_diversity(self):
        """Bereken populatie diversiteit"""
        if len(self.population) < 2:
            return 0
        
        total_distance = 0
        comparisons = 0
        
        for i in range(len(self.population)):
            for j in range(i + 1, len(self.population)):
                melody1 = [note for note, _ in self.population[i][0]]
                melody2 = [note for note, _ in self.population[j][0]]
                
                min_len = min(len(melody1), len(melody2))
                differences = sum(1 for k in range(min_len) if melody1[k] != melody2[k])
                differences += abs(len(melody1) - len(melody2))
                
                total_distance += differences
                comparisons += 1
        
        return total_distance / comparisons if comparisons > 0 else 0

    def run_generation(self):
        """Voer één generatie uit met diversiteit monitoring"""
        print(f"\n=== Generatie {self.generation} ===")
        
        # Evalueer fitness voor alle individuen
        fitness_scores = []
        for i, individual in enumerate(self.population):
            fitness = self.automated_fitness(individual)
            fitness_scores.append(fitness)
        
        # Bereken diversiteit
        diversity = self.calculate_diversity()
        self.diversity_history.append(diversity)
        
        # Toon resultaten
        best_fitness = max(fitness_scores)
        avg_fitness = sum(fitness_scores) / len(fitness_scores)
        
        print(f"Beste fitness: {best_fitness:.1f}")
        print(f"Gemiddelde fitness: {avg_fitness:.1f}")
        print(f"Populatie diversiteit: {diversity:.1f}")
        
        # Maak nieuwe generatie
        new_population = []
        
        # Elitisme - houd beste individuen
        elite_count = max(1, self.population_size // 10)
        sorted_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True)
        
        for i in range(elite_count):
            new_population.append(self.population[sorted_indices[i]])
        
        # Genereer rest via selectie, crossover, mutatie
        while len(new_population) < self.population_size:
            parent1 = self.tournament_selection(fitness_scores)
            parent2 = self.tournament_selection(fitness_scores)
            
            child1, child2 = self.crossover(parent1, parent2)
            
            child1 = self.mutate(child1)
            child2 = self.mutate(child2)
            
            new_population.extend([child1, child2])
        
        self.population = new_population[:self.population_size]
        self.generation += 1
        
        return self.population[sorted_indices[0]], best_fitness

    def generate_audio(self, individual, filename="compositie.wav"):
        """Genereer audio bestand"""
        try:
            import os
            original_dir = os.getcwd()
            os.chdir('music')
            
            self.muser.generate(individual)
            
            import shutil
            shutil.move('song.wav', f'../{filename}')
            
            os.chdir(original_dir)
            print(f"Audio gegenereerd: {filename}")
        except Exception as e:
            print(f"Fout bij audio generatie: {e}")

    def run_experiment(self, generations=10):
        """Voer experiment uit met analyse"""
        print("=== Fase 2 - Heuristieken & Geavanceerde Operators ===")
        print(f"Parameters:")
        print(f"  Populatie: {self.population_size}")
        print(f"  Mutatie kans: {self.mutation_rate}")
        print(f"  Crossover kans: {self.crossover_rate}")
        print(f"  Crossover type: {self.crossover_type}")
        print(f"  Mutatie strategie: {self.mutation_strategy}")
        print(f"  Generaties: {generations}")
        
        self.initialize_population()
        
        best_overall = None
        best_fitness = -1
        
        for gen in range(generations):
            best_individual, fitness = self.run_generation()
            
            if fitness > best_fitness:
                best_fitness = fitness
                best_overall = best_individual
        
        print(f"\n=== Experiment Resultaten ===")
        print(f"Beste fitness: {best_fitness:.1f}/100")
        print(f"Finale diversiteit: {self.diversity_history[-1]:.1f}")
        
        if len(self.diversity_history) > 1:
            diversity_trend = self.diversity_history[-1] - self.diversity_history[0]
            print(f"Diversiteit trend: {'+' if diversity_trend > 0 else ''}{diversity_trend:.1f}")
        
        # Genereer beste compositie
        if best_overall:
            filename = f"beste_compositie_{self.crossover_type}_{self.mutation_strategy}.wav"
            self.generate_audio(best_overall, filename)
            print(f"Beste compositie opgeslagen: {filename}")
        
        return {
            'crossover_type': self.crossover_type,
            'mutation_strategy': self.mutation_strategy,
            'best_fitness': best_fitness,
            'avg_fitness': sum([self.automated_fitness(ind) for ind in self.population]) / len(self.population),
            'final_diversity': self.diversity_history[-1] if self.diversity_history else 0
        }


def run_comparison_experiments():
    """Voer vergelijkingsexperimenten uit"""
    print("=== Vergelijking Verschillende Strategieën ===")
    
    experiments = [
        ('uniform', 'pitch'),
        ('uniform', 'rhythm'), 
        ('uniform', 'both'),
        ('single', 'both'),
        ('single', 'pitch')
    ]
    
    results = []
    
    for crossover_type, mutation_strategy in experiments:
        print(f"\n--- Experiment: {crossover_type} + {mutation_strategy} ---")
        ga = AdvancedGA(population_size=15, crossover_type=crossover_type, 
                       mutation_strategy=mutation_strategy)
        result = ga.run_experiment(generations=8)
        results.append(result)
    
    # Print vergelijking
    print(f"\n=== RESULTATEN VERGELIJKING ===")
    print(f"{'Configuratie':<20} {'Beste Fitness':<15} {'Gem. Fitness':<15} {'Diversiteit':<12}")
    print("-" * 65)
    
    for result in results:
        config = f"{result['crossover_type']} + {result['mutation_strategy']}"
        print(f"{config:<20} {result['best_fitness']:<15.1f} {result['avg_fitness']:<15.1f} {result['final_diversity']:<12.1f}")
    
    # Analyse
    best_fitness = max(results, key=lambda x: x['best_fitness'])
    most_diverse = max(results, key=lambda x: x['final_diversity'])
    
    print(f"\nBeste fitness: {best_fitness['crossover_type']} + {best_fitness['mutation_strategy']} ({best_fitness['best_fitness']:.1f})")
    print(f"Meest divers: {most_diverse['crossover_type']} + {most_diverse['mutation_strategy']} ({most_diverse['final_diversity']:.1f})")


if __name__ == "__main__":
    print("=== MM Phase 2 - Heuristieken & Geavanceerde Operators ===")
    print("Kies modus:")
    print("1. Enkele run")
    print("2. Vergelijking experimenten")
    
    choice = input("Keuze (1/2): ").strip() or "1"
    
    if choice == "2":
        run_comparison_experiments()
    else:
        # Enkele run
        crossover_type = input("Crossover type (single/uniform, standaard uniform): ") or "uniform"
        mutation_strategy = input("Mutatie strategie (pitch/rhythm/both, standaard both): ") or "both"
        generations = int(input("Aantal generaties (standaard 10): ") or "10")
        
        ga = AdvancedGA(crossover_type=crossover_type, mutation_strategy=mutation_strategy)
        ga.run_experiment(generations)
    
    print("\n=== Analyse van Parameters ===")
    print("Crossover strategieën:")
    print("  - Single-point: Traditioneel, behoudt lange segmenten")
    print("  - Uniform: Meer variatie, kan leiden tot meer diversiteit")
    print("Mutatie strategieën:")
    print("  - Pitch: Alleen toonhoogte veranderen")
    print("  - Rhythm: Alleen ritme veranderen")
    print("  - Both: Combinatie voor maximale variatie")
    print("Invloed op kwaliteit: Uniform + Both geeft meestal beste balans")
    print("Invloed op diversiteit: Uniform crossover houdt meer variatie")