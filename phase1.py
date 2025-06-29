"""
Simple Genetic Algorithm - Phase 1 Implementation
Voor TINLML03 - Minimum implementatie volgens opdracht

Fase 1 Features:
- Eenvoudige populatie van building blocks
- Handmatige fitness-functie (scores toekenen)
- Selectie, crossover en mutatie
- Parameter documentatie
"""

import random
import copy
from muser import Muser

class SimpleGA:
    def __init__(self, population_size=20, mutation_rate=0.1, crossover_rate=0.7):
        """
        Eenvoudige GA voor muziekcompositie
        
        Parameters:
        - population_size: Grootte van de populatie (20 werkt goed)
        - mutation_rate: Kans op mutatie per gen (0.1 = 10%)
        - crossover_rate: Kans op crossover (0.7 = 70%)
        """
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.population = []
        self.generation = 0
        self.muser = Muser()
        
        # Building blocks - voorgedefinieerde muzikale patronen
        self.building_blocks = {
            'intro': [('c', 4), ('e', 4), ('g', 4), ('c*', 4)],
            'verse': [('c', 8), ('d', 8), ('e', 4), ('f', 4), ('g', 4)],
            'chorus': [('g', 4), ('a', 4), ('b', 4), ('c*', 2)],
            'bridge': [('f', 8), ('g', 8), ('a', 4), ('g', 4), ('f', 4)],
            'outro': [('g', 4), ('f', 4), ('e', 4), ('c', 2)],
            'bass': [('c2', 2), ('g2', 2), ('c2', 2), ('g2', 2)]
        }
        
        self.notes = ['c', 'd', 'e', 'f', 'g', 'a', 'b']
        self.octaves = ['2', '3', '', '*']

    def create_individual(self):
        """Maak een individu (compositie) van willekeurige building blocks"""
        num_blocks = random.randint(3, 6)
        melody = []
        bass = []
        
        # Selecteer willekeurige building blocks
        for _ in range(num_blocks):
            block_name = random.choice(list(self.building_blocks.keys()))
            block = copy.deepcopy(self.building_blocks[block_name])
            
            if block_name == 'bass':
                bass.extend(block)
            else:
                melody.extend(block)
        
        # Zorg voor minimale compositie
        if not melody:
            melody = copy.deepcopy(self.building_blocks['verse'])
        if not bass:
            bass = copy.deepcopy(self.building_blocks['bass'])
            
        return (melody, bass)

    def initialize_population(self):
        """Initialiseer de populatie met willekeurige individuen"""
        self.population = []
        for _ in range(self.population_size):
            individual = self.create_individual()
            self.population.append(individual)

    def fitness_function(self, individual):
        """Handmatige fitness-functie - gebruiker kent scores toe"""
        melody, bass = individual
        
        print(f"\n--- Evalueer Compositie ---")
        print(f"Melodie: {len(melody)} noten")
        print(f"Bas: {len(bass)} noten")
        print(f"Melodie preview: {melody[:3]}...")
        print(f"Bas preview: {bass[:2]}...")
        
        # Probeer muziek te genereren voor evaluatie
        try:
            self.generate_audio(individual, "temp_evaluatie.wav")
            print("Audio gegenereerd: temp_evaluatie.wav")
        except Exception as e:
            print(f"Kon geen audio genereren: {e}")
        
        # Handmatige score
        while True:
            try:
                score = float(input("Geef een score (0-100): "))
                if 0 <= score <= 100:
                    return score
                print("Score moet tussen 0 en 100 zijn")
            except ValueError:
                print("Voer een geldig getal in")

    def selection(self):
        """Tournament selectie - kies beste uit willekeurige groep"""
        tournament_size = 3
        tournament = random.sample(self.population, min(tournament_size, len(self.population)))
        
        # Evalueer fitness voor tournament
        best_individual = None
        best_fitness = -1
        
        for individual in tournament:
            fitness = self.fitness_function(individual)
            if fitness > best_fitness:
                best_fitness = fitness
                best_individual = individual
        
        return best_individual

    def crossover(self, parent1, parent2):
        """Single-point crossover"""
        if random.random() > self.crossover_rate:
            return parent1, parent2
        
        # Crossover voor melodie
        if len(parent1[0]) > 1 and len(parent2[0]) > 1:
            point = random.randint(1, min(len(parent1[0]), len(parent2[0])) - 1)
            child1_melody = parent1[0][:point] + parent2[0][point:]
            child2_melody = parent2[0][:point] + parent1[0][point:]
        else:
            child1_melody, child2_melody = parent1[0], parent2[0]
        
        # Crossover voor bas
        if len(parent1[1]) > 1 and len(parent2[1]) > 1:
            point = random.randint(1, min(len(parent1[1]), len(parent2[1])) - 1)
            child1_bass = parent1[1][:point] + parent2[1][point:]
            child2_bass = parent2[1][:point] + parent1[1][point:]
        else:
            child1_bass, child2_bass = parent1[1], parent2[1]
        
        return (child1_melody, child1_bass), (child2_melody, child2_bass)

    def mutate(self, individual):
        """Mutatie - verander willekeurig noten of duur"""
        individual = copy.deepcopy(individual)
        
        # Muteer melodie
        for i in range(len(individual[0])):
            if random.random() < self.mutation_rate:
                if random.random() < 0.5:  # Pitch mutatie
                    new_note = random.choice(self.notes) + random.choice(self.octaves)
                    individual[0][i] = (new_note, individual[0][i][1])
                else:  # Duur mutatie
                    new_duration = random.choice([2, 4, 8, 16])
                    individual[0][i] = (individual[0][i][0], new_duration)
        
        # Muteer bas
        for i in range(len(individual[1])):
            if random.random() < self.mutation_rate:
                if random.random() < 0.5:  # Pitch mutatie
                    new_note = random.choice(self.notes) + random.choice(['2', '3'])
                    individual[1][i] = (new_note, individual[1][i][1])
                else:  # Duur mutatie
                    new_duration = random.choice([2, 4, 8])
                    individual[1][i] = (individual[1][i][0], new_duration)
        
        return individual

    def run_generation(self):
        """Voer één generatie uit"""
        print(f"\n=== Generatie {self.generation} ===")
        
        new_population = []
        
        # Genereer nieuwe populatie
        while len(new_population) < self.population_size:
            # Selectie
            parent1 = self.selection()
            parent2 = self.selection()
            
            # Crossover
            child1, child2 = self.crossover(parent1, parent2)
            
            # Mutatie
            child1 = self.mutate(child1)
            child2 = self.mutate(child2)
            
            new_population.extend([child1, child2])
        
        # Trim naar exacte populatie grootte
        self.population = new_population[:self.population_size]
        self.generation += 1

    def generate_audio(self, individual, filename="compositie.wav"):
        """Genereer audio bestand"""
        try:
            import os
            original_dir = os.getcwd()
            if not os.path.exists('music'):
                os.makedirs('music')
            os.chdir('music')
            
            self.muser.generate(individual)
            
            import shutil
            shutil.move('song.wav', f'../{filename}')
            
            os.chdir(original_dir)
            print(f"Audio gegenereerd: {filename}")
        except Exception as e:
            print(f"Fout bij audio generatie: {e}")

    def run(self, generations=5):
        """Voer het complete GA uit"""
        print("=== Fase 1 - Basale GA ===")
        print(f"Parameters:")
        print(f"  Populatie: {self.population_size}")
        print(f"  Mutatie kans: {self.mutation_rate}")
        print(f"  Crossover kans: {self.crossover_rate}")
        print(f"  Generaties: {generations}")
        
        print("\nInitialiseer populatie...")
        self.initialize_population()
        
        for gen in range(generations):
            print(f"\nGeneratie {gen + 1}/{generations}")
            self.run_generation()
            
            # Vraag of door wilt gaan
            if gen < generations - 1:
                continue_input = input("Doorgaan naar volgende generatie? (y/n): ")
                if continue_input.lower() != 'y':
                    break
        
        print(f"\nGA voltooid na {self.generation} generaties")
        
        # Genereer finale compositie
        if self.population:
            best = self.population[0]  # Neem eerste als voorbeeld
            self.generate_audio(best, "finale_compositie.wav")
            print("Finale compositie opgeslagen als: finale_compositie.wav")


if __name__ == "__main__":
    print("=== Melody Maker Fase 1 ===")
    print("Handmatige fitness scoring met building blocks\n")
    
    # Vraag parameters
    pop_size = int(input("Populatie grootte (standaard 10): ") or "10")
    mut_rate = float(input("Mutatie kans 0.0-1.0 (standaard 0.15): ") or "0.15")
    cross_rate = float(input("Crossover kans 0.0-1.0 (standaard 0.7): ") or "0.7")
    gens = int(input("Aantal generaties (standaard 3): ") or "3")
    
    # Start GA
    ga = SimpleGA(pop_size, mut_rate, cross_rate)
    ga.run(gens)
    
    print("\n=== Parameter Documentatie ===")
    print("Populatie grootte: Meer individuen = meer diversiteit, maar langzamer")
    print("Mutatie kans: Hoger = meer variatie, maar mogelijk slechter")
    print("Crossover kans: Hoger = meer combinaties, meestal beter")
    print("Afstemming: Start met standaard waarden, experimenteer daarna")