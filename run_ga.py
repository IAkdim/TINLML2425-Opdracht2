#!/usr/bin/env python3

from melody_maker import MelodyMaker
import sys

def main():
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
    
    # Initialize the genetic algorithm
    ga = MelodyMaker(population_size, mutation_rate, crossover_rate)
    
    print("\nInitializing population...")
    ga.initialize_population()
    
    # Evolution loop
    best_overall = None
    best_fitness = -1
    
    for gen in range(generations):
        try:
            best_individual = ga.run_generation()
            
            if best_individual[1] > best_fitness:
                best_fitness = best_individual[1]
                best_overall = best_individual[0]
                
                # Generate music for current best
                filename = f"generation_{gen}_best.wav"
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
    
    # Final results
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

if __name__ == "__main__":
    main()