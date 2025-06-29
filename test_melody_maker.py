#!/usr/bin/env python3
"""
Quick test script for the modified melody_maker_phase2.py
"""

import sys
sys.path.append('.')

from melody_maker_phase2 import run_full_test_suite, print_test_results

def test_small_run():
    """Test with very small parameters to verify it works"""
    print("Testing melody maker with small parameters...")
    
    # Very small test parameters
    population_size = 5
    generations = 2
    mutation_rate = 0.12
    crossover_rate = 0.8
    
    try:
        results = run_full_test_suite(population_size, mutation_rate, crossover_rate, generations)
        print_test_results(results)
        print("\n✅ Test completed successfully!")
        return True
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_small_run()