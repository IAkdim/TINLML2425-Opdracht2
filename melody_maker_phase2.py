"""
Melody Maker - Phase 2 Implementation
Genetic Algorithm for Musical Composition with Advanced Heuristics

Phase 2 Features:
- Automated fitness function with musical heuristics
- Chord progression analysis (II-V-I patterns)
- Measure structure validation (4/4 time)
- Key consistency checking (white keys only)
- Advanced mutation strategies (pitch vs rhythm)
- Uniform crossover option
- Population diversity monitoring
- Musical form analysis (verse/chorus structure)
"""

import random
import copy
import math
from music.muser import Muser
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

class MelodyMakerPhase2:
    def __init__(self, population_size=20, mutation_rate=0.1, crossover_rate=0.7, crossover_type='single', mutation_strategy='both'):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.crossover_type = crossover_type 
        self.mutation_strategy = mutation_strategy
        self.population = []
        self.generation = 0
        self.muser = Muser()
        self.diversity_history = []
        
        # Musical knowledge for heuristics
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
        self.octaves = ['2', '3', '', '*']
        self.note_to_num = {'c': 0, 'd': 1, 'e': 2, 'f': 3, 'g': 4, 'a': 5, 'b': 6}
        
        # Chord progressions
        self.common_progressions = [
            [0, 3, 4, 0],  # I-IV-V-I
            [1, 4, 0],     # ii-V-I
            [5, 3, 1, 4, 0],  # vi-IV-ii-V-I
            [0, 5, 3, 4],  # I-vi-IV-V
        ]
        
    def create_individual(self):
        """Create a random individual with better musical structure"""
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
        """Initialize population with diversity"""
        self.population = []
        for _ in range(self.population_size):
            individual = self.create_individual()
            self.population.append(individual)
    
    def automated_fitness_function(self, individual):
        """Automated fitness function with musical heuristics"""
        fitness = 0
        melody, bass = individual
        
        # 1. Length and structure (0-20 points)
        fitness += self.evaluate_length_structure(melody)
        
        # 2. Key consistency (0-15 points)
        fitness += self.evaluate_key_consistency(melody, bass)
        
        # 3. Chord progressions (0-20 points)
        fitness += self.evaluate_chord_progressions(melody, bass)
        
        # 4. Melodic contour (0-15 points)
        fitness += self.evaluate_melodic_contour(melody)
        
        # 5. Rhythmic diversity (0-10 points)
        fitness += self.evaluate_rhythmic_diversity(melody)
        
        # 6. Musical form (0-10 points)
        fitness += self.evaluate_musical_form(melody)
        
        # 7. Ending resolution (0-10 points)
        fitness += self.evaluate_ending_resolution(melody, bass)
        
        return min(fitness, 100)  # Cap at 100
    
    def evaluate_length_structure(self, melody):
        """Evaluate length and measure structure (4/4 time)"""
        score = 0
        total_beats = sum(abs(note[1]) for note in melody)
        
        # Prefer 8 or 16 measures (32 or 64 beats in 4/4)
        if 30 <= total_beats <= 34:  # ~8 measures
            score += 15
        elif 62 <= total_beats <= 66:  # ~16 measures
            score += 20
        elif 14 <= total_beats <= 18:  # ~4 measures
            score += 10
        else:
            # Penalty for odd lengths
            score += max(0, 10 - abs(total_beats - 32) // 4)
        
        # Check if total beats is divisible by 4 (proper measures)
        if total_beats % 4 == 0:
            score += 5
            
        return score
    
    def evaluate_key_consistency(self, melody, bass):
        """Check if composition uses only white keys"""
        score = 15  # Start with full points
        
        all_notes = melody + bass
        for note, _ in all_notes:
            clean_note = note.replace('*', '').replace('2', '').replace('3', '')
            if '#' in clean_note or 'b' in clean_note:
                score -= 2  # Penalty for black keys
                
        return max(0, score)
    
    def evaluate_chord_progressions(self, melody, bass):
        """Analyze chord progressions for common patterns"""
        score = 0
        
        # Extract root notes from bass line
        bass_roots = []
        for note, duration in bass:
            if duration > 0: 
                clean_note = note.replace('*', '').replace('2', '').replace('3', '')
                if clean_note in self.note_to_num:
                    bass_roots.append(self.note_to_num[clean_note])
        
        if len(bass_roots) < 3:
            return score
        
        # Look for common progressions
        for progression in self.common_progressions:
            for i in range(len(bass_roots) - len(progression) + 1):
                segment = bass_roots[i:i+len(progression)]
                # Normalize to scale degrees
                if len(segment) == len(progression):
                    normalized = [(note - segment[0]) % 7 for note in segment]
                    if normalized == progression:
                        score += 8  # Bonus for each recognized progression
                        
        # II-V-I progression bonus
        for i in range(len(bass_roots) - 2):
            progression = [(bass_roots[i+j] - bass_roots[i]) % 7 for j in range(3)]
            if progression == [1, 4, 0]:  # ii-V-I
                score += 12
                
        return min(score, 20)
    
    def evaluate_melodic_contour(self, melody):
        """Evaluate melodic movement and contour"""
        score = 0
        
        if len(melody) < 2:
            return score
            
        # Convert to pitch numbers for analysis
        pitches = []
        for note, duration in melody:
            if duration > 0:  # Skip rests
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
            return score
            
        # Analyze intervals
        intervals = [abs(pitches[i+1] - pitches[i]) for i in range(len(pitches)-1)]
        
        # Prefer step-wise motion with occasional leaps
        stepwise_count = sum(1 for interval in intervals if interval <= 2)
        leap_count = sum(1 for interval in intervals if 3 <= interval <= 5)
        large_leap_count = sum(1 for interval in intervals if interval > 5)
        
        # Reward mostly stepwise with some leaps
        score += min(10, stepwise_count * 2)
        score += min(5, leap_count)
        score -= large_leap_count
        
        return max(0, score)
    
    def evaluate_rhythmic_diversity(self, melody):
        """Evaluate rhythmic variety"""
        score = 0
        
        durations = [abs(duration) for _, duration in melody if duration != 0]
        unique_durations = set(durations)
        
        # Reward rhythmic variety
        score += min(6, len(unique_durations) * 2)
        
        # Prefer common note values
        common_durations = {2, 4, 8, 16}
        common_count = sum(1 for d in durations if d in common_durations)
        score += min(4, common_count)
        
        return score
    
    def evaluate_musical_form(self, melody):
        """Basic musical form analysis"""
        score = 0
        
        # Look for repetition (verse/chorus structure)
        if len(melody) >= 8:
            mid_point = len(melody) // 2
            first_half = melody[:mid_point]
            second_half = melody[mid_point:mid_point + len(first_half)]
            
            # Check for exact repetition
            if first_half == second_half:
                score += 5
            else:
                # Check for similar patterns
                similarity = sum(1 for i, (n1, d1) in enumerate(first_half)
                               if i < len(second_half) and 
                               n1.replace('*', '').replace('2', '').replace('3', '') == 
                               second_half[i][0].replace('*', '').replace('2', '').replace('3', ''))
                score += min(5, similarity)
                
        return score
    
    def evaluate_ending_resolution(self, melody, bass):
        """Check for proper musical ending"""
        score = 0
        
        if melody:
            last_note = melody[-1][0].replace('*', '').replace('2', '').replace('3', '')
            if last_note == 'c':  # Ends on tonic
                score += 5
                
        if bass:
            last_bass = bass[-1][0].replace('*', '').replace('2', '').replace('3', '')
            if last_bass == 'c':  # Bass ends on tonic
                score += 5
                
        return score
    
    def tournament_selection(self, fitness_cache, tournament_size=3):
        """Tournament selection using cached fitness scores"""
        tournament = random.sample(list(range(len(self.population))), min(tournament_size, len(self.population)))
        best_idx = max(tournament, key=lambda i: fitness_cache[i])
        return self.population[best_idx]
    
    def uniform_crossover(self, parent1, parent2):
        """Uniform crossover - each gene has 50% chance from each parent"""
        if random.random() > self.crossover_rate:
            return parent1, parent2
            
        # Crossover melody tracks
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
        
        # Add remaining notes from longer parent
        if len(parent1[0]) > min_len:
            child1_melody.extend(parent1[0][min_len:])
        if len(parent2[0]) > min_len:
            child2_melody.extend(parent2[0][min_len:])
            
        # Similar for bass tracks
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
        """Original single-point crossover"""
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
    
    def crossover(self, parent1, parent2):
        """Choose crossover method based on configuration"""
        if self.crossover_type == 'uniform':
            return self.uniform_crossover(parent1, parent2)
        else:
            return self.single_point_crossover(parent1, parent2)
    
    def pitch_mutation(self, individual):
        """Specialized pitch mutation"""
        individual = copy.deepcopy(individual)
        
        # Mutate melody track pitches
        for i in range(len(individual[0])):
            if random.random() < self.mutation_rate:
                base_note = random.choice(self.notes)
                octave = random.choice(self.octaves)
                new_note = base_note + octave
                individual[0][i] = (new_note, individual[0][i][1])
        
        # Mutate bass track pitches
        for i in range(len(individual[1])):
            if random.random() < self.mutation_rate:
                base_note = random.choice(self.notes)
                octave = random.choice(['2', '3'])
                new_note = base_note + octave
                individual[1][i] = (new_note, individual[1][i][1])
                
        return individual
    
    def rhythm_mutation(self, individual):
        """Specialized rhythm mutation"""
        individual = copy.deepcopy(individual)
        
        # Mutate melody track rhythms
        for i in range(len(individual[0])):
            if random.random() < self.mutation_rate:
                new_duration = random.choice([2, 4, 8, 16])
                individual[0][i] = (individual[0][i][0], new_duration)
        
        # Mutate bass track rhythms
        for i in range(len(individual[1])):
            if random.random() < self.mutation_rate:
                new_duration = random.choice([2, 4, 8])
                individual[1][i] = (individual[1][i][0], new_duration)
                
        return individual
    
    def mutate(self, individual):
        """Advanced mutation with strategy selection"""
        # Choose mutation strategy
        strategy = random.choice(['pitch', 'rhythm', 'both'])
        
        if strategy == 'pitch':
            return self.pitch_mutation(individual)
        elif strategy == 'rhythm':
            return self.rhythm_mutation(individual)
        else:  # both
            individual = self.pitch_mutation(individual)
            return self.rhythm_mutation(individual)
    
    def calculate_diversity(self):
        """Calculate population diversity"""
        if len(self.population) < 2:
            return 0
            
        total_distance = 0
        comparisons = 0
        
        for i in range(len(self.population)):
            for j in range(i + 1, len(self.population)):
                # Simple diversity metric: count different notes
                melody1 = [note for note, _ in self.population[i][0]]
                melody2 = [note for note, _ in self.population[j][0]]
                
                min_len = min(len(melody1), len(melody2))
                differences = sum(1 for k in range(min_len) if melody1[k] != melody2[k])
                differences += abs(len(melody1) - len(melody2))
                
                total_distance += differences
                comparisons += 1
        
        return total_distance / comparisons if comparisons > 0 else 0
    
    def run_generation(self):
        """Run one generation with diversity monitoring"""
        print(f"\n=== Generation {self.generation} ===")
        
        # Evaluate fitness for all individuals
        fitness_scores = []
        for i, individual in enumerate(self.population):
            print(f"Evaluating individual {i+1}/{len(self.population)} (automated scoring)")
            fitness = self.automated_fitness_function(individual)
            fitness_scores.append((individual, fitness))
        
        # Sort by fitness
        fitness_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Calculate and track diversity
        diversity = self.calculate_diversity()
        self.diversity_history.append(diversity)
        
        print(f"\nGeneration {self.generation} Results:")
        for i, (_, fitness) in enumerate(fitness_scores[:3]):
            print(f"  Top {i+1}: Fitness = {fitness:.1f}")
        print(f"  Population diversity: {diversity:.1f}")
        
        # Create next generation
        new_population = []
        
        # Keep best individuals (elitism)
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
        
        return fitness_scores[0]
    
    def generate_music(self, individual, filename="composition.wav"):
        """Generate audio file from an individual"""
        try:
            import os
            original_dir = os.getcwd()
            os.chdir('music')
            
            self.muser.generate(individual)
            
            import shutil
            shutil.move('song.wav', f'../{filename}')
            
            os.chdir(original_dir)
            print(f"Generated audio: {filename}")
            
        except Exception as e:
            print(f"Error generating music: {e}")
    
    def get_test_summary(self, fitness_scores):
        """Get test summary with key metrics"""
        fitness_values = [score[1] for score in fitness_scores]
        
        return {
            "crossover": self.crossover_type,
            "mutation": self.mutation_strategy,
            "test_name": f"{self.crossover_type.capitalize()} + {self.mutation_strategy}",
            "best_fitness": round(max(fitness_values), 1),
            "avg_fitness": round(sum(fitness_values) / len(fitness_values), 1),
            "final_diversity": round(self.diversity_history[-1], 2) if self.diversity_history else 0.0
        }
    
    def print_analysis_report(self):
        """Print detailed analysis of the evolutionary process"""
        print(f"\n=== Phase 2 Analysis Report ===")
        print(f"Generations completed: {self.generation}")
        print(f"Population size: {self.population_size}")
        print(f"Mutation rate: {self.mutation_rate}")
        print(f"Crossover rate: {self.crossover_rate}")
        print(f"Crossover type: {self.crossover_type}")
        
        if self.diversity_history:
            avg_diversity = sum(self.diversity_history) / len(self.diversity_history)
            print(f"Average diversity: {avg_diversity:.2f}")
            print(f"Final diversity: {self.diversity_history[-1]:.2f}")
            
            if len(self.diversity_history) > 1:
                diversity_trend = self.diversity_history[-1] - self.diversity_history[0]
                print(f"Diversity trend: {'+' if diversity_trend > 0 else ''}{diversity_trend:.2f}")


def run_experiment(crossover_type, mutation_strategy, population_size=15, mutation_rate=0.12, crossover_rate=0.8, generations=10):
    """Run a single experiment with given parameters"""
    print(f"\n=== Running experiment: {crossover_type.capitalize()} + {mutation_strategy} ===")
    
    # Initialize the genetic algorithm
    ga = MelodyMakerPhase2(population_size, mutation_rate, crossover_rate, crossover_type, mutation_strategy)
    
    print("Initializing population...")
    ga.initialize_population()
    
    best_overall = None
    best_fitness = -1
    all_fitness_scores = []
    
    for gen in range(generations):
        try:
            best_individual = ga.run_generation()
            
            if best_individual[1] > best_fitness:
                best_fitness = best_individual[1]
                best_overall = best_individual[0]
            
            # Collect fitness scores for this generation
            gen_fitness_scores = []
            for individual in ga.population:
                fitness = ga.automated_fitness_function(individual)
                gen_fitness_scores.append((individual, fitness))
            
            all_fitness_scores.extend(gen_fitness_scores)
            
        except Exception as e:
            print(f"Error in generation {gen}: {e}")
            continue
    
    # Get test summary
    test_summary = ga.get_test_summary(all_fitness_scores)
    
    return test_summary, best_overall

def run_full_test_suite(population_size=15, mutation_rate=0.12, crossover_rate=0.8, generations=10):
    """Run the complete test suite with 5 scenarios"""
    print("\n=== RUNNING FULL TEST SUITE (5 SCENARIOS) ===")
    print("Test parameters:")
    print(f"  Population size: {population_size}")
    print(f"  Generations: {generations}")
    print(f"  Mutation rate: {mutation_rate}")
    print(f"  Crossover rate: {crossover_rate}")
    print(f"  Fitness method: volledig automatisch via heuristieken")
    print()
    
    # Define the 5 test scenarios
    test_scenarios = [
        ('uniform', 'pitch'),
        ('uniform', 'rhythm'),
        ('uniform', 'both'),
        ('single', 'both'),
        ('single', 'pitch')
    ]
    
    results = []
    
    for i, (crossover_type, mutation_strategy) in enumerate(test_scenarios, 1):
        print(f"\n--- Test {i}/5: {crossover_type.capitalize()} + {mutation_strategy} ---")
        
        summary, best_individual = run_experiment(
            crossover_type, mutation_strategy, population_size, 
            mutation_rate, crossover_rate, generations
        )
        
        results.append(summary)
        
        # Generate music for best individual
        if best_individual:
            temp_ga = MelodyMakerPhase2()
            filename = f"test_{i}_{crossover_type}_{mutation_strategy}_best.wav"
            temp_ga.generate_music(best_individual, filename)
            print(f"Best composition saved as: {filename}")
    
    return results

def print_test_results(results):
    """Print test results in a formatted table"""
    print("\n=== 5.3 RESULTATEN (SAMENVATTING) ===")
    print(f"{'Test':<20} {'Best Fitness':<15} {'Gemiddelde Fitness':<20} {'Diversiteit (laatste gen.)':<25}")
    print("-" * 80)
    
    for result in results:
        print(f"{result['test_name']:<20} {result['best_fitness']:<15} {result['avg_fitness']:<20} {result['final_diversity']:<25}")
    
    print("\n=== ANALYSIS ===")
    best_overall = max(results, key=lambda x: x['best_fitness'])
    best_avg = max(results, key=lambda x: x['avg_fitness'])
    most_diverse = max(results, key=lambda x: x['final_diversity'])
    
    print(f"Highest fitness: {best_overall['test_name']} ({best_overall['best_fitness']})")
    print(f"Best average fitness: {best_avg['test_name']} ({best_avg['avg_fitness']})")
    print(f"Most diverse: {most_diverse['test_name']} ({most_diverse['final_diversity']})")

def create_fitness_comparison_plot(results):
    """Create bar chart comparing fitness metrics across tests"""
    test_names = [result['test_name'] for result in results]
    best_fitness = [result['best_fitness'] for result in results]
    avg_fitness = [result['avg_fitness'] for result in results]
    
    x = np.arange(len(test_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 8))
    bars1 = ax.bar(x - width/2, best_fitness, width, label='Best Fitness', color='#2E86AB', alpha=0.8)
    bars2 = ax.bar(x + width/2, avg_fitness, width, label='Average Fitness', color='#A23B72', alpha=0.8)
    
    ax.set_xlabel('Test Configuration', fontsize=12, fontweight='bold')
    ax.set_ylabel('Fitness Score', fontsize=12, fontweight='bold')
    ax.set_title('Genetic Algorithm Performance Comparison\nFitness Scores by Crossover Type and Mutation Strategy', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(test_names, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height}', ha='center', va='bottom', fontweight='bold')
    
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"fitness_comparison_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Fitness comparison plot saved as: {filename}")
    plt.close()
    return filename

def create_diversity_plot(results):
    """Create bar chart for diversity comparison"""
    test_names = [result['test_name'] for result in results]
    diversity = [result['final_diversity'] for result in results]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Color bars based on crossover type
    colors = ['#F18F01' if 'Uniform' in name else '#C73E1D' for name in test_names]
    bars = ax.bar(test_names, diversity, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    
    ax.set_xlabel('Test Configuration', fontsize=12, fontweight='bold')
    ax.set_ylabel('Population Diversity (Final Generation)', fontsize=12, fontweight='bold')
    ax.set_title('Population Diversity by Configuration\nHigher Values Indicate More Genetic Variation', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, diversity):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{value}', ha='center', va='bottom', fontweight='bold')
    
    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right')
    
    # Add legend for colors
    uniform_patch = plt.Rectangle((0,0),1,1, facecolor='#F18F01', alpha=0.8, label='Uniform Crossover')
    single_patch = plt.Rectangle((0,0),1,1, facecolor='#C73E1D', alpha=0.8, label='Single-point Crossover')
    ax.legend(handles=[uniform_patch, single_patch], fontsize=11)
    
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"diversity_comparison_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Diversity comparison plot saved as: {filename}")
    plt.close()
    return filename

def create_combined_results_plot(results):
    """Create comprehensive dashboard with multiple metrics"""
    fig = plt.figure(figsize=(16, 12))
    
    # Create a 2x2 grid
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    test_names = [result['test_name'] for result in results]
    best_fitness = [result['best_fitness'] for result in results]
    avg_fitness = [result['avg_fitness'] for result in results]
    diversity = [result['final_diversity'] for result in results]
    
    # 1. Fitness comparison (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    x = np.arange(len(test_names))
    width = 0.35
    ax1.bar(x - width/2, best_fitness, width, label='Best', color='#2E86AB', alpha=0.8)
    ax1.bar(x + width/2, avg_fitness, width, label='Average', color='#A23B72', alpha=0.8)
    ax1.set_title('Fitness Scores Comparison', fontweight='bold')
    ax1.set_ylabel('Fitness Score')
    ax1.set_xticks(x)
    ax1.set_xticklabels([name.replace(' + ', '\\n') for name in test_names], fontsize=9)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. Diversity comparison (top right)
    ax2 = fig.add_subplot(gs[0, 1])
    colors = ['#F18F01' if 'Uniform' in name else '#C73E1D' for name in test_names]
    ax2.bar(range(len(test_names)), diversity, color=colors, alpha=0.8)
    ax2.set_title('Population Diversity', fontweight='bold')
    ax2.set_ylabel('Diversity Score')
    ax2.set_xticks(range(len(test_names)))
    ax2.set_xticklabels([name.replace(' + ', '\\n') for name in test_names], fontsize=9)
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. Fitness vs Diversity scatter plot (middle left)
    ax3 = fig.add_subplot(gs[1, 0])
    scatter_colors = ['#F18F01' if 'Uniform' in name else '#C73E1D' for name in test_names]
    scatter = ax3.scatter(diversity, best_fitness, c=scatter_colors, s=100, alpha=0.8, edgecolors='black')
    for i, name in enumerate(test_names):
        ax3.annotate(name.replace(' + ', '\\n'), (diversity[i], best_fitness[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    ax3.set_xlabel('Population Diversity')
    ax3.set_ylabel('Best Fitness')
    ax3.set_title('Fitness vs Diversity Trade-off', fontweight='bold')
    ax3.grid(alpha=0.3)
    
    # 4. Performance ranking (middle right)
    ax4 = fig.add_subplot(gs[1, 1])
    # Calculate combined score (weighted average)
    combined_scores = [0.5 * best + 0.3 * avg + 0.2 * div for best, avg, div in zip(best_fitness, avg_fitness, diversity)]
    sorted_indices = sorted(range(len(combined_scores)), key=lambda i: combined_scores[i], reverse=True)
    sorted_names = [test_names[i] for i in sorted_indices]
    sorted_scores = [combined_scores[i] for i in sorted_indices]
    
    bars = ax4.barh(range(len(sorted_names)), sorted_scores, color='#4ECDC4', alpha=0.8)
    ax4.set_yticks(range(len(sorted_names)))
    ax4.set_yticklabels([name.replace(' + ', ' + ') for name in sorted_names])
    ax4.set_xlabel('Combined Performance Score')
    ax4.set_title('Overall Performance Ranking\\n(50% Best + 30% Avg + 20% Diversity)', fontweight='bold')
    ax4.grid(axis='x', alpha=0.3)
    
    # Add score labels
    for i, (bar, score) in enumerate(zip(bars, sorted_scores)):
        ax4.text(score + 0.5, bar.get_y() + bar.get_height()/2, 
                f'{score:.1f}', va='center', fontweight='bold')
    
    # 5. Results table (bottom, spanning both columns)
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('off')
    
    # Create table data
    table_data = []
    headers = ['Test Configuration', 'Best Fitness', 'Avg Fitness', 'Diversity', 'Rank']
    for i, (idx, result) in enumerate(zip(sorted_indices, [results[i] for i in sorted_indices])):
        table_data.append([
            result['test_name'],
            f"{result['best_fitness']:.1f}",
            f"{result['avg_fitness']:.1f}",
            f"{result['final_diversity']:.1f}",
            f"#{i+1}"
        ])
    
    table = ax5.table(cellText=table_data, colLabels=headers, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2)
    
    # Style the table
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#E8E8E8')
        table[(0, i)].set_text_props(weight='bold')
    
    plt.suptitle('Genetic Algorithm Performance Analysis Dashboard\\nCrossover Types vs Mutation Strategies', 
                 fontsize=16, fontweight='bold', y=0.95)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"combined_results_dashboard_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Combined results dashboard saved as: {filename}")
    plt.close()
    return filename

def create_all_plots(results):
    """Generate all visualization plots and return filenames"""
    print("\\n=== GENERATING VISUALIZATION PLOTS ===")
    
    plot_files = []
    
    try:
        # Create individual plots
        plot_files.append(create_fitness_comparison_plot(results))
        plot_files.append(create_diversity_plot(results))
        plot_files.append(create_combined_results_plot(results))
        
        print(f"\\nSuccessfully generated {len(plot_files)} visualization plots")
        return plot_files
        
    except Exception as e:
        print(f"Error generating plots: {e}")
        return plot_files

def main():
    print("=== Melody Maker - Genetic Algorithm Phase 2 ===")
    print("Phase 2: Automated fitness with musical heuristics")
    print("Features: Chord progressions, key consistency, advanced mutations\n")
    
    # Ask what mode to run
    print("Choose mode:")
    print("1. Full test suite (5 scenarios with plots)")
    print("2. Simple run (just evolve and generate song.wav)")
    mode = input("Enter choice (1/2, default 1): ").strip() or "1"
    
    if mode == "1":
        # GA Parameters for comparison
        population_size = int(input("Population size (default 15): ") or "15")
        mutation_rate = float(input("Mutation rate 0.0-1.0 (default 0.12): ") or "0.12")
        crossover_rate = float(input("Crossover rate 0.0-1.0 (default 0.8): ") or "0.8")
        generations = int(input("Number of generations (default 10): ") or "10")
        
        print(f"\nRunning comparison with parameters:")
        print(f"  Population: {population_size}")
        print(f"  Mutation rate: {mutation_rate}")
        print(f"  Crossover rate: {crossover_rate}")
        print(f"  Generations: {generations}")
        
        # Run experiments
        results = []
        
        # Run the full test suite (5 scenarios)
        results = run_full_test_suite(population_size, mutation_rate, crossover_rate, generations)
        
        # Print results table
        print_test_results(results)
        
        # Generate and save visualization plots
        plot_files = create_all_plots(results)
        
        if plot_files:
            print(f"\nGenerated {len(plot_files)} visualization files:")
            for plot_file in plot_files:
                print(f"  • {plot_file}")
        
        print(f"\n Generated {len(results)} audio files for best compositions")
        print("All results, plots, and audio files saved in current directory")
    
    elif mode == "2":
        # Simple mode - just evolve and generate song.wav
        print("\n=== SIMPLE MODE: Evolve and Generate Song ===")
        
        # GA Parameters
        print("  Recommended settings for good compositions:")
        print("  Population: 20-50, Generations: 30-100")
        print("  Larger values = better evolution but longer runtime\n")
        
        population_size = int(input("Population size (default 30): ") or "30")
        mutation_rate = float(input("Mutation rate 0.0-1.0 (default 0.12): ") or "0.12")
        crossover_rate = float(input("Crossover rate 0.0-1.0 (default 0.8): ") or "0.8")
        crossover_type = input("Crossover type (single/uniform, default uniform): ") or "uniform"
        mutation_strategy = input("Mutation strategy (pitch/rhythm/both, default both): ") or "both"
        generations = int(input("Number of generations (default 50): ") or "50")
        
        print(f"\nStarting evolution with parameters:")
        print(f"  Population: {population_size}")
        print(f"  Mutation rate: {mutation_rate}")
        print(f"  Crossover rate: {crossover_rate}")
        print(f"  Crossover type: {crossover_type}")
        print(f"  Mutation strategy: {mutation_strategy}")
        print(f"  Generations: {generations}")
        
        # Initialize and run GA
        ga = MelodyMakerPhase2(population_size, mutation_rate, crossover_rate, crossover_type, mutation_strategy)
        
        print("\nInitializing population...")
        ga.initialize_population()
        
        best_overall = None
        best_fitness = -1
        
        print(f"\nEvolving for {generations} generations...")
        for gen in range(generations):
            try:
                best_individual = ga.run_generation()
                
                if best_individual[1] > best_fitness:
                    best_fitness = best_individual[1]
                    best_overall = best_individual[0]
                
                # Show progress every 10 generations (or every 5 for shorter runs)
                progress_interval = 10 if generations > 20 else 5
                if (gen + 1) % progress_interval == 0 or gen == 0:
                    print(f"Generation {gen + 1}/{generations}: Best fitness = {best_fitness:.1f}")
                    
            except KeyboardInterrupt:
                print(f"\nEvolution interrupted at generation {gen + 1}")
                break
            except Exception as e:
                print(f"Error in generation {gen}: {e}")
                continue
        
        # Generate final composition
        if best_overall:
            print(f"\nEvolution complete!")
            print(f"Final best fitness: {best_fitness:.1f}/100")
            print(f"Generating final composition...")
            
            ga.generate_music(best_overall, "song.wav")
            print("Final composition saved as: song.wav")
            
            print(f"\nComposition details:")
            print(f"  Melody track: {len(best_overall[0])} notes")
            print(f"  Bass track: {len(best_overall[1])} notes")
            print(f"  Final diversity: {ga.diversity_history[-1]:.2f}" if ga.diversity_history else "")
        else:
            print("No valid composition generated")
        
        print("\nSimple mode completed!")
    
    else:
        # Original single run mode
        population_size = int(input("Population size (default 15): ") or "15")
        mutation_rate = float(input("Mutation rate 0.0-1.0 (default 0.12): ") or "0.12")
        crossover_rate = float(input("Crossover rate 0.0-1.0 (default 0.8): ") or "0.8")
        crossover_type = input("Crossover type (single/uniform, default uniform): ") or "uniform"
        mutation_strategy = input("Mutation strategy (pitch/rhythm/both, default both): ") or "both"
        generations = int(input("Number of generations (default 10): ") or "10")
        
        print(f"\nStarting Phase 2 GA with parameters:")
        print(f"  Population: {population_size}")
        print(f"  Mutation rate: {mutation_rate}")
        print(f"  Crossover rate: {crossover_rate}")
        print(f"  Crossover type: {crossover_type}")
        print(f"  Mutation strategy: {mutation_strategy}")
        print(f"  Generations: {generations}")
        
        # Initialize the genetic algorithm
        ga = MelodyMakerPhase2(population_size, mutation_rate, crossover_rate, crossover_type, mutation_strategy)
        
        print("\nInitializing population...")
        ga.initialize_population()
        
        best_overall = None
        best_fitness = -1
        all_fitness_scores = []
        
        for gen in range(generations):
            try:
                best_individual = ga.run_generation()
                
                if best_individual[1] > best_fitness:
                    best_fitness = best_individual[1]
                    best_overall = best_individual[0]
                    
                    # Generate music for current best
                    filename = f"phase2_gen_{gen}_best.wav"
                    ga.generate_music(best_overall, filename)
                    print(f"New best composition saved as: {filename}")
                
                print(f"Best fitness so far: {best_fitness:.1f}")
                
                # Ask if user wants to continue
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
        
        # Final results and analysis
        ga.print_analysis_report()
        
        if best_overall:
            print("\nGenerating final best composition...")
            ga.generate_music(best_overall, "phase2_final_best.wav")
            print("Final composition saved as: phase2_final_best.wav")
            
            print(f"\nBest composition structure:")
            print(f"  Melody track: {len(best_overall[0])} notes")
            print(f"  Bass track: {len(best_overall[1])} notes")
            print(f"  Final fitness: {best_fitness:.1f}/100")
        
        print("\nPhase 2 GA completed successfully!")


if __name__ == "__main__":
    main()
